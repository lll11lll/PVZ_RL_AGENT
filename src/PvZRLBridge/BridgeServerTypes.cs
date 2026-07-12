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

[assembly: MelonInfo(typeof(PvZRLBridge.BridgeMod), "PvZRLBridge", "0.1.0", "Codex")]
[assembly: MelonGame("LanPiaoPiao", "PlantsVsZombiesRH")]
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("PvZRLBridgeLifecycleHarness")]

namespace PvZRLBridge;


internal sealed class BridgeConfig
{
    public int Port { get; set; } = 32323;
    public int RequestTimeoutSeconds { get; set; } = 10;
    public int StepFrames { get; set; } = 8;
    public float GameSpeed { get; set; } = 1f;
    public string GameSpeedMode { get; set; } = "game_speed";
    public int Seed { get; set; } = 12345;
    public int FallbackRows { get; set; } = 5;
    public int FallbackColumns { get; set; } = 9;
    public int SeedScreenCheckInterval { get; set; } = 100;
    public bool DebugPerformance { get; set; }
    public bool DebugObservation { get; set; }
    public bool DebugSun { get; set; }
    public int DebugSunSampleInterval { get; set; } = 25;
    public List<int> PlantTypes { get; } = new() { (int)PlantType.SunFlower, (int)PlantType.Peashooter };
}

internal enum PendingRequestState
{
    Queued = 0,
    Dispatching = 1,
    Completed = 2,
    Canceled = 3
}

internal sealed class PendingRequest
{
    private readonly long _timestampFrequency;
    private int _state = (int)PendingRequestState.Queued;

    public PendingRequest(long requestId, string json, TimeSpan timeout)
        : this(requestId, json, Stopwatch.GetTimestamp(), Stopwatch.Frequency, timeout)
    {
    }

    internal PendingRequest(long requestId, string json, long createdTimestamp, long timestampFrequency, TimeSpan timeout)
    {
        if (timestampFrequency <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(timestampFrequency));
        }

        RequestId = requestId;
        Json = json;
        CreatedTimestamp = createdTimestamp;
        _timestampFrequency = timestampFrequency;
        DeadlineTimestamp = CalculateDeadline(createdTimestamp, timestampFrequency, timeout);
    }

    public long RequestId { get; }
    public string Json { get; }
    public long CreatedTimestamp { get; }
    public long DeadlineTimestamp { get; }
    public PendingRequestState State => (PendingRequestState)Volatile.Read(ref _state);
    public TaskCompletionSource<BridgeResponse> Completion { get; } =
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    public TimeSpan RemainingUntilDeadline(long nowTimestamp)
    {
        var remainingTicks = DeadlineTimestamp - nowTimestamp;
        if (remainingTicks <= 0)
        {
            return TimeSpan.Zero;
        }

        return TimeSpan.FromSeconds(remainingTicks / (double)_timestampFrequency);
    }

    public bool TryBeginDispatch(long nowTimestamp, BridgeResponse expiredResponse)
    {
        if (nowTimestamp >= DeadlineTimestamp)
        {
            TryCancel(expiredResponse);
            return false;
        }

        return Interlocked.CompareExchange(
                   ref _state,
                   (int)PendingRequestState.Dispatching,
                   (int)PendingRequestState.Queued) == (int)PendingRequestState.Queued;
    }

    public bool TryCancel(BridgeResponse response)
    {
        if (Interlocked.CompareExchange(
                ref _state,
                (int)PendingRequestState.Canceled,
                (int)PendingRequestState.Queued) != (int)PendingRequestState.Queued)
        {
            return false;
        }

        Completion.TrySetResult(response);
        return true;
    }

    public bool TryComplete(BridgeResponse response)
    {
        if (Interlocked.CompareExchange(
                ref _state,
                (int)PendingRequestState.Completed,
                (int)PendingRequestState.Dispatching) != (int)PendingRequestState.Dispatching)
        {
            return false;
        }

        Completion.TrySetResult(response);
        return true;
    }

    private static long CalculateDeadline(long createdTimestamp, long timestampFrequency, TimeSpan timeout)
    {
        var timeoutSeconds = Math.Max(0.0, timeout.TotalSeconds);
        var duration = (long)Math.Ceiling(timeoutSeconds * timestampFrequency);
        if (duration <= 0)
        {
            return createdTimestamp;
        }
        if (createdTimestamp > long.MaxValue - duration)
        {
            return long.MaxValue;
        }
        return createdTimestamp + duration;
    }
}

internal sealed class PendingRequestQueue
{
    private readonly object _gate = new();
    private readonly Queue<PendingRequest> _queue = new();
    private bool _accepting;

    public void StartAccepting()
    {
        lock (_gate)
        {
            if (_queue.Count != 0)
            {
                throw new InvalidOperationException("Cannot start a bridge request queue with stale entries.");
            }
            _accepting = true;
        }
    }

    public bool TryEnqueue(PendingRequest request, BridgeResponse rejectionResponse)
    {
        lock (_gate)
        {
            if (!_accepting)
            {
                request.TryCancel(rejectionResponse);
                return false;
            }

            _queue.Enqueue(request);
            return true;
        }
    }

    public bool TryTakeForDispatch(
        long nowTimestamp,
        BridgeResponse expiredResponse,
        out PendingRequest? request)
    {
        lock (_gate)
        {
            request = null;
            if (!_accepting || _queue.Count == 0)
            {
                return false;
            }

            var candidate = _queue.Dequeue();
            if (candidate.TryBeginDispatch(nowTimestamp, expiredResponse))
            {
                request = candidate;
            }
            return true;
        }
    }

    public int StopAcceptingAndCancel(BridgeResponse cancellationResponse)
    {
        lock (_gate)
        {
            _accepting = false;
            var canceled = 0;
            while (_queue.Count > 0)
            {
                if (_queue.Dequeue().TryCancel(cancellationResponse))
                {
                    canceled++;
                }
            }
            return canceled;
        }
    }
}

internal sealed class ActiveClientRegistry
{
    private readonly object _gate = new();
    private readonly Dictionary<long, TcpClient> _clients = new();
    private readonly ManualResetEventSlim _drained = new(initialState: true);
    private bool _accepting;
    private long _nextClientId;

    public int ActiveCount
    {
        get
        {
            lock (_gate)
            {
                return _clients.Count;
            }
        }
    }

    public void StartAccepting()
    {
        lock (_gate)
        {
            if (_clients.Count != 0)
            {
                throw new InvalidOperationException("Cannot start bridge clients while previous workers remain active.");
            }
            _accepting = true;
            _drained.Set();
        }
    }

    public bool TryRegister(TcpClient client, out long clientId)
    {
        lock (_gate)
        {
            clientId = 0;
            if (!_accepting)
            {
                return false;
            }

            clientId = ++_nextClientId;
            _drained.Reset();
            _clients.Add(clientId, client);
            return true;
        }
    }

    public void Unregister(long clientId)
    {
        lock (_gate)
        {
            _clients.Remove(clientId);
            if (_clients.Count == 0)
            {
                _drained.Set();
            }
        }
    }

    public IReadOnlyList<TcpClient> StopAcceptingAndSnapshot()
    {
        lock (_gate)
        {
            _accepting = false;
            return _clients.Values.ToArray();
        }
    }

    public bool WaitForDrain(TimeSpan timeout) => _drained.Wait(timeout);
}

internal sealed class BridgeResponse
{
    public bool Ok { get; set; }
    public string? Error { get; set; }
    public string? Details { get; set; }
    public object? Data { get; set; }

    public static BridgeResponse Success(object? data) => new() { Ok = true, Data = data };

    public static BridgeResponse Fail(string error, string? details = null) =>
        new() { Ok = false, Error = error, Details = details };
}

