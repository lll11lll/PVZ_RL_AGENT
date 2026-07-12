using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Il2Cpp;
using MelonLoader;
using UnityEngine;
using Object = UnityEngine.Object;

namespace PvZRLBridge;

public sealed partial class BridgeMod
{
    public override void OnInitializeMelon()
    {
        _config.Port = ReadPortFromEnvironment();
        StartServer();
        LoggerInstance.Msg($"PvZRLBridge listening on 127.0.0.1:{_config.Port}");
        LoggerInstance.Msg("Go/no-go proof command: {\"command\":\"proof\",\"place_test\":true}");
    }

    public override void OnUpdate()
    {
        if (_stopping)
        {
            return;
        }

        _bridgeUpdateLoopCount++;
        ApplyConfiguredGameSpeed();
        LogBoardReadyOnce();

        var processed = 0;
        while (processed < 16 &&
               _pending.TryTakeForDispatch(
                   Stopwatch.GetTimestamp(),
                   CreateRequestTimeoutResponse(),
                   out var request))
        {
            processed++;
            if (request == null)
            {
                continue;
            }

            BridgeResponse response;
            try
            {
                response = HandleRequest(request.Json);
            }
            catch (Exception ex)
            {
                response = BridgeResponse.Fail("request_failed", ex.ToString());
            }

            request.TryComplete(response);
        }
    }

    public override void OnApplicationQuit()
    {
        try { RestoreGameSpeedInternal(); } catch { }
        StopServer();
    }

    private int ReadPortFromEnvironment()
    {
        var raw = Environment.GetEnvironmentVariable("PVZRL_BRIDGE_PORT");
        return int.TryParse(raw, out var port) && port > 0 ? port : DefaultPort;
    }

    private void StartServer()
    {
        var listener = new TcpListener(IPAddress.Loopback, _config.Port);
        listener.Start();
        _pending.StartAccepting();
        _activeClients.StartAccepting();
        _stopping = false;

        var serverThread = new Thread(() => ServerLoop(listener))
        {
            IsBackground = true,
            Name = "PvZRLBridgeServer"
        };
        lock (_serverStateGate)
        {
            _listener = listener;
            _serverThread = serverThread;
        }
        serverThread.Start();
    }

    private void StopServer()
    {
        _stopping = true;
        _pending.StopAcceptingAndCancel(CreateServerStoppingResponse());
        var clients = _activeClients.StopAcceptingAndSnapshot();

        TcpListener? listener;
        Thread? serverThread;
        lock (_serverStateGate)
        {
            listener = _listener;
            _listener = null;
            serverThread = _serverThread;
        }

        try { listener?.Stop(); } catch { }
        foreach (var client in clients)
        {
            SafeCloseClient(client);
        }

        if (serverThread != null && serverThread != Thread.CurrentThread)
        {
            try
            {
                if (!serverThread.Join(ServerThreadShutdownTimeout))
                {
                    LoggerInstance.Warning(
                        $"PvZRLBridge server thread did not stop within {ServerThreadShutdownTimeout.TotalSeconds:0.###} seconds.");
                }
            }
            catch (Exception ex)
            {
                LoggerInstance.Warning("PvZRLBridge server thread join failed: " + ex.Message);
            }
        }

        if (!_activeClients.WaitForDrain(ClientWorkerShutdownTimeout))
        {
            LoggerInstance.Warning(
                $"PvZRLBridge still has {_activeClients.ActiveCount} client worker(s) after " +
                $"{ClientWorkerShutdownTimeout.TotalSeconds:0.###} seconds; sockets were closed and shutdown will continue.");
        }

        lock (_serverStateGate)
        {
            if (ReferenceEquals(_serverThread, serverThread) && (serverThread == null || !serverThread.IsAlive))
            {
                _serverThread = null;
            }
        }
    }

    private void ServerLoop(TcpListener listener)
    {
        while (!_stopping)
        {
            try
            {
                var client = listener.AcceptTcpClient();
                if (!_activeClients.TryRegister(client, out var clientId))
                {
                    SafeCloseClient(client);
                    continue;
                }

                var queued = false;
                try
                {
                    queued = ThreadPool.QueueUserWorkItem(_ => HandleClient(clientId, client));
                }
                catch (Exception ex)
                {
                    if (!_stopping)
                    {
                        LoggerInstance.Warning("PvZRLBridge could not queue a client worker: " + ex.Message);
                    }
                }

                if (!queued)
                {
                    _activeClients.Unregister(clientId);
                    SafeCloseClient(client);
                }
            }
            catch (Exception ex)
            {
                if (!_stopping)
                {
                    LoggerInstance.Warning("PvZRLBridge listener failed: " + ex.Message);
                    Thread.Sleep(250);
                }
            }
        }
    }

    private void HandleClient(long clientId, TcpClient client)
    {
        try
        {
            using (client)
            {
                client.NoDelay = true;
                using var stream = client.GetStream();
                using var reader = new StreamReader(stream, Encoding.UTF8, false, 8192, false);
                using var writer = new StreamWriter(stream, new UTF8Encoding(false), 8192, false)
                {
                    AutoFlush = true
                };

                while (!_stopping && client.Connected)
                {
                    string? line;
                    try
                    {
                        line = reader.ReadLine();
                    }
                    catch
                    {
                        break;
                    }

                    if (line == null)
                    {
                        break;
                    }

                    if (string.IsNullOrWhiteSpace(line))
                    {
                        continue;
                    }

                    var requestId = Interlocked.Increment(ref _nextRequestId);
                    var pending = new PendingRequest(
                        requestId,
                        line,
                        TimeSpan.FromSeconds(Math.Max(0.001, _config.RequestTimeoutSeconds)));
                    _pending.TryEnqueue(pending, CreateServerStoppingResponse());

                    var response = WaitForRequestCompletion(pending);

                    try
                    {
                        writer.WriteLine(JsonSerializer.Serialize(response, _jsonOptions));
                    }
                    catch
                    {
                        break;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            if (!_stopping)
            {
                LoggerInstance.Warning($"PvZRLBridge client {clientId} failed: {ex.Message}");
            }
        }
        finally
        {
            _activeClients.Unregister(clientId);
        }
    }

    private static BridgeResponse WaitForRequestCompletion(PendingRequest pending)
    {
        while (true)
        {
            if (pending.Completion.Task.IsCompleted)
            {
                return pending.Completion.Task.GetAwaiter().GetResult();
            }

            var remaining = pending.RemainingUntilDeadline(Stopwatch.GetTimestamp());
            if (remaining <= TimeSpan.Zero)
            {
                var timeout = CreateRequestTimeoutResponse();
                if (pending.TryCancel(timeout))
                {
                    return timeout;
                }

                // Dispatch owns the request. Waiting for its real completion prevents a
                // timeout response from racing ahead of an in-progress Unity mutation.
                return pending.Completion.Task.GetAwaiter().GetResult();
            }

            if (pending.Completion.Task.Wait(remaining))
            {
                return pending.Completion.Task.GetAwaiter().GetResult();
            }
        }
    }

    private static BridgeResponse CreateRequestTimeoutResponse() =>
        BridgeResponse.Fail("timeout", "Unity main thread did not process the request in time.");

    private static BridgeResponse CreateServerStoppingResponse() =>
        BridgeResponse.Fail("server_stopping", "PvZRLBridge is stopping and cannot accept this request.");

    private static void SafeCloseClient(TcpClient client)
    {
        try { client.Client?.Shutdown(SocketShutdown.Both); } catch { }
        try { client.Close(); } catch { }
    }
}
