using System;
using System.Collections.Concurrent;
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

namespace PvZRLBridge;

public sealed class BridgeMod : MelonMod
{
    private const int DefaultPort = 32323;
    private const uint MouseEventLeftDown = 0x0002;
    private const uint MouseEventLeftUp = 0x0004;
    private readonly ConcurrentQueue<PendingRequest> _pending = new();
    private readonly BridgeConfig _config = new();
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false
    };

    private TcpListener? _listener;
    private Thread? _serverThread;
    private volatile bool _stopping;
    private int _loggedBoardReady;
    private readonly SeedRuntimeCache _seedRuntimeCache = new();
    private readonly RestartUiCache _restartUiCache = new();
    private int _observationsSinceSeedProbe;
    private bool _hasOriginalSpeed;
    private float _originalGameSpeed = 1f;
    private float _originalTimeScale = 1f;
    private float _originalFixedDeltaTime = 0.02f;
    private long _bridgeUpdateLoopCount;
    private long _speedApplyCount;
    private long _validSpeedModeApplyCount;
    private long _sunSpawnCompensationApplyCount;
    private long _resetCount;
    private long _letsRockClickCount;
    private int _lastSunDebugFrame = -100000;
    private SunDebugSnapshot _sunDebugSnapshot = new();
    private bool _speedConfigDirty = true;
    private string _lastAppliedSpeedMode = "";
    private float _lastRequestedGameSpeed = float.NaN;
    private float _lastAppliedGameSpeed = float.NaN;
    private float _lastAppliedTimeScale = float.NaN;
    private float _lastAppliedFixedDeltaTime = float.NaN;

    [StructLayout(LayoutKind.Sequential)]
    private struct NativePoint
    {
        public int X;
        public int Y;
    }

    [DllImport("user32.dll")]
    private static extern bool ClientToScreen(IntPtr hWnd, ref NativePoint lpPoint);

    [DllImport("user32.dll")]
    private static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    private static extern bool SetCursorPos(int x, int y);

    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);

    public override void OnInitializeMelon()
    {
        _config.Port = ReadPortFromEnvironment();
        StartServer();
        LoggerInstance.Msg($"PvZRLBridge listening on 127.0.0.1:{_config.Port}");
        LoggerInstance.Msg("Go/no-go proof command: {\"command\":\"proof\",\"place_test\":true}");
    }

    public override void OnUpdate()
    {
        _bridgeUpdateLoopCount++;
        ApplyConfiguredGameSpeed();
        LogBoardReadyOnce();

        var processed = 0;
        while (processed < 16 && _pending.TryDequeue(out var request))
        {
            processed++;
            BridgeResponse response;
            try
            {
                response = HandleRequest(request.Json);
            }
            catch (Exception ex)
            {
                response = BridgeResponse.Fail("request_failed", ex.ToString());
            }

            request.Completion.TrySetResult(response);
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
        _listener = new TcpListener(IPAddress.Loopback, _config.Port);
        _listener.Start();
        _serverThread = new Thread(ServerLoop)
        {
            IsBackground = true,
            Name = "PvZRLBridgeServer"
        };
        _serverThread.Start();
    }

    private void StopServer()
    {
        _stopping = true;
        try { _listener?.Stop(); } catch { }
    }

    private void ServerLoop()
    {
        while (!_stopping)
        {
            try
            {
                var client = _listener!.AcceptTcpClient();
                ThreadPool.QueueUserWorkItem(_ => HandleClient(client));
            }
            catch
            {
                if (!_stopping)
                {
                    Thread.Sleep(250);
                }
            }
        }
    }

    private void HandleClient(TcpClient client)
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

                var pending = new PendingRequest(line);
                _pending.Enqueue(pending);

                BridgeResponse response;
                if (pending.Completion.Task.Wait(TimeSpan.FromSeconds(_config.RequestTimeoutSeconds)))
                {
                    response = pending.Completion.Task.Result;
                }
                else
                {
                    response = BridgeResponse.Fail("timeout", "Unity main thread did not process the request in time.");
                }

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

    private BridgeResponse HandleRequest(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        var command = ReadString(root, "command", "cmd")?.Trim().ToLowerInvariant() ?? "observe";

        var data = command switch
        {
            "ping" => new { message = "pong", boardFound = FindBoard() != null },
            "configure" => Configure(root),
            "proof" => Proof(root),
            "observe" => BuildObservation(
                includeDebugArrays: ReadBool(root, _config.DebugObservation, "debug_observation", "debugObservation", "include_debug_arrays", "includeDebugArrays"),
                forceSeedProbe: ReadBool(root, false, "force_seed_probe", "forceSeedProbe"),
                forceRestartProbe: ReadBool(root, false, "force_restart_probe", "forceRestartProbe")),
            "screen_state_fast" => ScreenStateFast(),
            "adventure_screen_state" => AdventureScreenState(root),
            "press_adventure_once" => PressAdventureOnce(root),
            "click_startup_ok_once" => ClickStartupOkOnce(root),
            "dismiss_startup_popup_once" => ClickStartupOkOnce(root),
            "click_trophy_once" => ClickTrophyOnce(root),
            "click_level_complete_reward_once" => ClickTrophyOnce(root),
            "click_reward_continue_once" => ClickRewardContinueOnce(root),
            "click_try_again_once" => ClickTryAgainOnce(root),
            "legal_actions" => LegalActionsCommand(root),
            "teacher_action" => TeacherActionCommand(root),
            "fusion_probe" => FusionProbeCommand(root),
            "fusion_step" => FusionStepCommand(root),
            "step" => ActionCommand(root),
            "reset" => ResetCommand(root),
            "auto_reset" => AutoReset(root),
            "reset_cleanup" => ResetCleanup(root),
            "almanac_probe" => AlmanacProbe(root),
            "seed_probe" => SeedProbe(),
            "ui_probe" => UiProbe(root),
            "auto_select_seeds" => AutoSelectSeeds(root),
            "select_seed_card_once" => SelectSeedCardOnce(root),
            "press_lets_rock_once" => PressLetsRockOnce(root),
            "soft_reset" => SoftReset(root),
            "restore_game_speed" => RestoreGameSpeed(),
            "hard_reset" => new { ok = false, requiresExternalRestart = true, message = "Hard reset is handled by the Python wrapper by restarting the process." },
            _ => throw new InvalidOperationException($"Unknown command '{command}'.")
        };

        return BridgeResponse.Success(data);
    }

    private object Configure(JsonElement root)
    {
        if (TryReadInt(root, out var stepFrames, "step_frames", "stepFrames"))
        {
            _config.StepFrames = Math.Max(1, stepFrames);
        }

        if (TryReadFloat(root, out var gameSpeed, "game_speed", "gameSpeed"))
        {
            var requested = Math.Max(0.01f, gameSpeed);
            if (Math.Abs(_config.GameSpeed - requested) > 0.0001f)
            {
                _config.GameSpeed = requested;
                MarkSpeedConfigDirty();
            }
        }

        var speedMode = ReadString(root, "game_speed_mode", "gameSpeedMode", "speed_mode", "speedMode");
        if (!string.IsNullOrWhiteSpace(speedMode))
        {
            var normalized = NormalizeGameSpeedMode(speedMode);
            if (_config.GameSpeedMode != normalized)
            {
                _config.GameSpeedMode = normalized;
                MarkSpeedConfigDirty();
            }
        }

        if (ReadBool(root, false, "valid_speed_mode", "validSpeedMode"))
        {
            if (_config.GameSpeedMode != "safe")
            {
                _config.GameSpeedMode = "safe";
                MarkSpeedConfigDirty();
            }
        }

        if (TryReadInt(root, out var seed, "seed"))
        {
            _config.Seed = seed;
            UnityEngine.Random.InitState(seed);
        }

        if (TryReadInt(root, out var seedScreenCheckInterval, "seed_screen_check_interval", "seedScreenCheckInterval"))
        {
            _config.SeedScreenCheckInterval = Math.Max(0, seedScreenCheckInterval);
        }

        _config.DebugPerformance = ReadBool(root, _config.DebugPerformance, "debug_performance", "debugPerformance");
        _config.DebugObservation = ReadBool(root, _config.DebugObservation, "debug_observation", "debugObservation");
        _config.DebugSun = ReadBool(root, _config.DebugSun, "debug_sun", "debugSun");
        if (TryReadInt(root, out var debugSunSample, "debug_sun_sample_interval", "debugSunSampleInterval"))
        {
            _config.DebugSunSampleInterval = Math.Max(0, debugSunSample);
        }

        var plantTypes = ReadIntArray(root, "plant_types", "plantTypes");
        if (plantTypes.Count > 0)
        {
            if (!_config.PlantTypes.SequenceEqual(plantTypes))
            {
                _config.PlantTypes.Clear();
                _config.PlantTypes.AddRange(plantTypes);
                InvalidateSeedRuntimeCache("plant_types_changed");
            }
        }

        if (TryReadInt(root, out var rowCount, "row_count", "rowCount"))
        {
            _config.FallbackRows = Math.Max(1, rowCount);
        }

        if (TryReadInt(root, out var columnCount, "column_count", "columnCount"))
        {
            _config.FallbackColumns = Math.Max(1, columnCount);
        }

        return new
        {
            _config.Port,
            _config.StepFrames,
            _config.GameSpeed,
            gameSpeedMode = _config.GameSpeedMode,
            _config.Seed,
            plantTypes = _config.PlantTypes.ToArray(),
            fallbackRows = _config.FallbackRows,
            fallbackColumns = _config.FallbackColumns,
            seedScreenCheckInterval = _config.SeedScreenCheckInterval,
            debugPerformance = _config.DebugPerformance,
            debugObservation = _config.DebugObservation,
            debugSun = _config.DebugSun,
            debugSunSampleInterval = _config.DebugSunSampleInterval,
            unityTimeScale = SafeReadTimeScale(),
            fixedDeltaTime = SafeReadFixedDeltaTime()
        };
    }

    private object Proof(JsonElement root)
    {
        var report = BuildProofReport();
        var placeTest = ReadBool(root, false, "place_test", "placeTest");
        if (placeTest && report.CanReadBoard)
        {
            var plantType = ReadInt(root, _config.PlantTypes[0], "plant_type", "plantType");
            var row = ReadInt(root, 0, "row");
            var column = ReadInt(root, 0, "column", "col");
            report.PlacementAttempt = TryPlacePlant(plantType, row, column, true);
        }

        report.GoNoGo = placeTest && report.CanReadBoard && report.CanReadPlants && report.CanReadZombies &&
                        report.PlacementAttempt?.Success == true;
        report.NextStep = !placeTest
            ? "READ CHECK ONLY: rerun proof with place_test=true before continuing to RL."
            : report.GoNoGo
            ? "GO: continue to the Gym-style environment wrapper."
            : "NO-GO: stop RL work until board read/control succeeds.";
        return report;
    }

    private ProofReport BuildProofReport()
    {
        var board = FindBoard();
        var createPlant = FindCreatePlant();
        var report = new ProofReport
        {
            BoardFound = board != null,
            CreatePlantFound = createPlant != null
        };

        if (board == null)
        {
            return report;
        }

        try
        {
            report.Sun = board.theSun;
            report.Wave = board.theWave;
            report.MaxWave = board.theMaxWave;
            report.PlantCount = board.plantArray?.Count ?? 0;
            report.ZombieCount = board.zombieArray?.Count ?? 0;
            report.CanReadBoard = true;
            report.CanReadPlants = board.plantArray != null;
            report.CanReadZombies = board.zombieArray != null;
        }
        catch (Exception ex)
        {
            report.ReadError = ex.Message;
        }

        return report;
    }

    private object ResetCommand(JsonElement root)
    {
        var mode = ReadString(root, "mode")?.Trim().ToLowerInvariant() ?? "soft";
        return mode == "hard"
            ? new { ok = false, requiresExternalRestart = true, message = "Use Python hard_reset() to restart the game process." }
            : mode == "auto"
            ? AutoReset(root)
            : SoftReset(root);
    }

    private object AutoReset(JsonElement root)
    {
        _resetCount++;
        var before = BuildObservation(forceSeedProbe: true, forceRestartProbe: true);
        var restartInfo = DetectRestartScreenInfo(broadScan: true);
        var actions = new List<string>();
        var allowActiveGameplayReset = AllowActiveGameplayReset(root);
        var requireLossSeedSelectionPath = ReadBool(
            root,
            false,
            "require_loss_seed_selection_path",
            "requireLossSeedSelectionPath",
            "loss_seed_selection_required",
            "lossSeedSelectionRequired");
        var requireSeedSelectionPath = ReadBool(
            root,
            false,
            "require_seed_selection_path",
            "requireSeedSelectionPath",
            "seed_selection_required",
            "seedSelectionRequired");
        var resetReason = ReadString(root, "reset_reason", "resetReason") ?? "";
        var terminalDetected = IsConfirmedTerminalOrTransitionState(before);
        var lossTerminalDetected = terminalDetected &&
                                   HasLossRestartEvidence(before) &&
                                   !HasPostWinEvidence(before);
        var lossSeedSelectionRequired = requireLossSeedSelectionPath || lossTerminalDetected;
        var seedSelectionPathRequired = requireSeedSelectionPath || lossSeedSelectionRequired;
        var allowInGameSeedReplay = seedSelectionPathRequired && allowActiveGameplayReset;
        if (!allowInGameSeedReplay && ShouldBlockActiveGameplayMutation("reset", before, actions, allowActiveGameplayReset, resetReason))
        {
            return new
            {
                ok = false,
                blockedBySafety = true,
                terminalDetected = false,
                lossTerminalDetected = false,
                lossSeedSelectionRequired = false,
                methodUsed = "blocked_active_gameplay",
                invokedUiRestart = false,
                restartClicked = false,
                restartClickMethod = "",
                restartClickTargetName = "",
                restartClickTargetPath = "",
                restartClickError = "",
                actions,
                before,
                observation = before,
                message = "Blocked reset/reinit while active gameplay is still running."
            };
        }
        if (allowInGameSeedReplay && IsActiveGameplayUnsafeForMutation(before))
        {
            actions.Add(
                $"[safety] allowed in-game seed-selection replay boundary reason={resetReason} " +
                $"wave={before.Wave}/{before.MaxWave} zombies={before.ZombieCount} plants={before.PlantCount} " +
                $"mowers={before.LogicalMowerCount}/{before.VisibleMowerObjectCount} gameplayReady={before.GameplayReady} " +
                $"screenState={before.ScreenState} nextStep={before.NextStep}");
        }
        var methodUsed = "soft_reset";
        var invokedUiRestart = false;
        var restartClicked = false;
        var restartClickMethod = "";
        var restartClickTargetName = "";
        var restartClickTargetPath = "";
        var restartClickError = "";

        if (before.SeedSelectionActive && !terminalDetected)
        {
            actions.Add("auto_reset refused while seed-selection UI is active; caller must wait or use seed automation.");
            return new
            {
                ok = false,
                terminalDetected,
                lossTerminalDetected,
                lossSeedSelectionRequired,
                methodUsed = "refused_seed_selection_active",
                invokedUiRestart,
                restartClicked,
                restartClickMethod,
                restartClickTargetName,
                restartClickTargetPath,
                restartClickError,
                actions,
                before,
                observation = before,
                message = "Refused auto reset during seed selection to avoid mixing seed chooser and gameplay UI states."
            };
        }

        if (seedSelectionPathRequired && !terminalDetected)
        {
            TryClearTerminalBoardForSeedReplay(actions);
            if (TryUIMgrEnterGame(actions))
            {
                invokedUiRestart = true;
                methodUsed = "UIMgr.EnterGame.in_game_seed_replay";
            }
            else
            {
                return new
                {
                    ok = false,
                    terminalDetected,
                    lossTerminalDetected,
                    lossSeedSelectionRequired,
                    seedSelectionPathRequired,
                    methodUsed = "in_game_seed_replay_required",
                    invokedUiRestart = false,
                    restartClicked,
                    restartClickMethod,
                    restartClickTargetName,
                    restartClickTargetPath,
                    restartClickError,
                    actions,
                    before,
                    observation = BuildObservation(),
                    message = "Seed-selection-required in-game reset could not invoke UIMgr.EnterGame; hard restart fallback is required."
                };
            }
        }

        if (terminalDetected)
        {
            restartClicked = TryInvokeLoseMenuRestart(
                actions,
                restartInfo,
                out restartClickMethod,
                out restartClickTargetName,
                out restartClickTargetPath,
                out restartClickError);
            invokedUiRestart = restartClicked;
            if (restartClicked)
            {
                methodUsed = restartClickMethod;
            }
            else if (lossSeedSelectionRequired)
            {
                methodUsed = "loss_restart_click_required";
                actions.Add("Loss terminal detected; internal gameplay-start fallbacks are disabled until restart click succeeds.");
            }
            else if (seedSelectionPathRequired && !lossSeedSelectionRequired)
            {
                TryClearTerminalBoardForSeedReplay(actions);
                if (TryUIMgrEnterGame(actions))
                {
                    invokedUiRestart = true;
                    methodUsed = "UIMgr.EnterGame";
                }
                else
                {
                    methodUsed = "seed_selection_replay_required";
                    actions.Add("Seed-selection-required reset could not invoke UIMgr.EnterGame after terminal board cleanup; InitBoard/soft reset fallbacks are disabled.");
                }
            }
            else if (TryUIMgrEnterGame(actions))
            {
                invokedUiRestart = true;
                methodUsed = "UIMgr.EnterGame";
            }
            else if (seedSelectionPathRequired)
            {
                methodUsed = "seed_selection_replay_required";
                actions.Add("Seed-selection-required reset could not invoke UIMgr.EnterGame; InitBoard/soft reset fallbacks are disabled.");
            }
            else if (TryInitBoardQuickReset(actions, allowActiveGameplayReset, ReadString(root, "reset_reason", "resetReason") ?? ""))
            {
                invokedUiRestart = true;
                methodUsed = "InitBoard.QuickInGame/StartInit";
            }
        }

        if (invokedUiRestart)
        {
            UnityEngine.Random.InitState(_config.Seed);
            InvalidateSeedRuntimeCache("auto_reset_ui_restart");
            InvalidateRestartUiCache("auto_reset_ui_restart");
            return new
            {
                ok = true,
                terminalDetected,
                lossTerminalDetected,
                lossSeedSelectionRequired,
                seedSelectionPathRequired,
                methodUsed,
                invokedUiRestart,
                restartClicked,
                restartClickMethod,
                restartClickTargetName,
                restartClickTargetPath,
                restartClickError,
                actions,
                before,
                observation = BuildObservation(),
                message = "Invoked in-game reset hook; Python should wait for board/gameplayReady."
            };
        }

        if (terminalDetected && lossSeedSelectionRequired)
        {
            return new
            {
                ok = false,
                terminalDetected,
                lossTerminalDetected,
                lossSeedSelectionRequired,
                methodUsed = "loss_restart_click_required",
                invokedUiRestart = false,
                restartClicked,
                restartClickMethod,
                restartClickTargetName,
                restartClickTargetPath,
                restartClickError,
                actions,
                before,
                observation = BuildObservation(),
                message = "Loss reset requires a visible UI restart click and must pass through seed selection; internal board/gameplay fallbacks were not invoked."
            };
        }

        if (terminalDetected && seedSelectionPathRequired)
        {
            return new
            {
                ok = false,
                terminalDetected,
                lossTerminalDetected,
                lossSeedSelectionRequired,
                seedSelectionPathRequired,
                methodUsed = "seed_selection_replay_required",
                invokedUiRestart = false,
                restartClicked,
                restartClickMethod,
                restartClickTargetName,
                restartClickTargetPath,
                restartClickError,
                actions,
                before,
                observation = BuildObservation(),
                message = "Seed-selection-required reset could not safely restart through seed selection; internal board/soft reset fallbacks were not invoked."
            };
        }

        InvalidateSeedRuntimeCache("auto_reset_soft_fallback");
        InvalidateRestartUiCache("auto_reset_soft_fallback");
        var soft = SoftReset(root);
        return new
        {
            ok = true,
            terminalDetected,
            lossTerminalDetected,
            lossSeedSelectionRequired,
            seedSelectionPathRequired,
            methodUsed,
            invokedUiRestart,
            restartClicked,
            restartClickMethod,
            restartClickTargetName,
            restartClickTargetPath,
            restartClickError,
            actions,
            before,
            softReset = soft,
            observation = BuildObservation()
        };
    }

    private bool TryClearTerminalBoardForSeedReplay(List<string> actions)
    {
        var board = FindBoard();
        if (board == null)
        {
            actions.Add("PreReplayTerminalClear skipped: Board not found.");
            return false;
        }

        try
        {
            var before = BuildResetCleanupReport(board);
            try
            {
                board.ClearTheBoard();
                actions.Add("PreReplayTerminalClear: Board.ClearTheBoard()");
            }
            catch (Exception ex)
            {
                actions.Add("PreReplayTerminalClear: Board.ClearTheBoard() failed: " + ex.Message);
            }

            board = DestroyDuplicateBoards(board, actions);
            ManualClearDynamicBoardObjects(board, actions);
            var stalePlants = DestroyStalePlants(board, actions);
            var staleMowers = DestroyStaleMowers(board, actions);
            var staleBullets = DestroyStaleComponents(board.bulletArray, actions, "Bullet");
            var staleZombies = DestroyStaleComponents(board.zombieArray, actions, "Zombie");
            var staleGridItems = DestroyStaleComponents(board.griditemArray, actions, "GridItem");
            var rewardObjects = DestroyBlockingRewardObjects(board, actions);
            var duplicateMowers = DestroyDuplicateMowers(board, actions);
            FreezeTerminalBoardBeforeReplay(board, actions);
            RefreshAllBoardBoxes(board, actions);
            var after = BuildResetCleanupReport(board);
            actions.Add(
                "PreReplayTerminalClear(" +
                $"beforePlants={before.LogicalPlantCount}/{before.VisiblePlantObjectCount}, " +
                $"beforeZombies={before.LogicalZombieCount}, beforeBullets={before.LogicalBulletCount}, " +
                $"destroyedStalePlants={stalePlants}, destroyedStaleMowers={staleMowers}, " +
                $"destroyedStaleBullets={staleBullets}, destroyedStaleZombies={staleZombies}, " +
                $"destroyedStaleGridItems={staleGridItems}, rewardObjects={rewardObjects}, " +
                $"duplicateMowers={duplicateMowers}, " +
                $"afterPlants={after.LogicalPlantCount}/{after.VisiblePlantObjectCount}, " +
                $"afterZombies={after.LogicalZombieCount}, afterBullets={after.LogicalBulletCount})");
            return true;
        }
        catch (Exception ex)
        {
            actions.Add("PreReplayTerminalClear failed: " + ex.Message);
            return false;
        }
    }

    private void FreezeTerminalBoardBeforeReplay(Board board, List<string> actions)
    {
        try { board.over = true; } catch { }
        try { board.moreZombiesComing = false; } catch { }
        try { board.newZombieWaveCountDown = 9999f; } catch { }
        try { board.nextZombieWaveCountDown = 9999f; } catch { }
        try { board.hugeWaveCountDown = 0f; } catch { }
        try { board.isHugeWave = false; } catch { }
        actions.Add("PreReplayTerminalClear: froze old terminal board wave timers before UIMgr.EnterGame.");
    }

    private object SoftReset(JsonElement root)
    {
        _resetCount++;
        var actions = new List<string>();
        var beforeObservation = BuildObservation(forceSeedProbe: true, forceRestartProbe: true);
        var allowActiveGameplayReset = AllowActiveGameplayReset(root);
        var resetReason = ReadString(root, "reset_reason", "resetReason") ?? "";
        if (ShouldBlockActiveGameplayMutation("soft reset", beforeObservation, actions, allowActiveGameplayReset, resetReason))
        {
            return new
            {
                ok = false,
                blockedBySafety = true,
                actions,
                before = beforeObservation,
                observation = beforeObservation,
                message = "Blocked soft reset while active gameplay is still running."
            };
        }

        var board = FindBoard();
        if (board == null)
        {
            return new { ok = false, error = "Board not found." };
        }

        var manualClear = ReadBool(root, true, "manual_clear", "manualClear");
        try
        {
            board.ClearTheBoard();
            actions.Add("Board.ClearTheBoard()");
        }
        catch (Exception ex)
        {
            actions.Add("Board.ClearTheBoard() failed: " + ex.Message);
        }

        if (manualClear)
        {
            board = DestroyDuplicateBoards(board, actions);
            ManualClearDynamicBoardObjects(board, actions);
            var rewardObjects = DestroyBlockingRewardObjects(board, actions);
            if (rewardObjects > 0)
            {
                actions.Add($"DestroyBlockingRewardObjects(count={rewardObjects})");
            }

            var duplicateMowers = DestroyDuplicateMowers(board, actions);
            if (duplicateMowers > 0)
            {
                actions.Add($"DestroyDuplicateMowers(count={duplicateMowers})");
            }

            RefreshAllBoardBoxes(board, actions);
        }

        if (TryReadInt(root, out var startSun, "start_sun", "startSun"))
        {
            try
            {
                board.SetSun(startSun);
                actions.Add("Board.SetSun(" + startSun + ")");
            }
            catch (Exception ex)
            {
                actions.Add("Board.SetSun failed: " + ex.Message);
            }
        }

        ResetCardCooldowns(actions);

        if (ReadBool(root, false, "run_init", "runInit"))
        {
            var initBoard = FindInitBoard();
            if (initBoard != null)
            {
                try
                {
                    initBoard.StartInit();
                    actions.Add("InitBoard.StartInit()");
                }
                catch (Exception ex)
                {
                    actions.Add("InitBoard.StartInit() failed: " + ex.Message);
                }
            }
            else
            {
                actions.Add("InitBoard not found.");
            }
        }

        UnityEngine.Random.InitState(_config.Seed);
        InvalidateSeedRuntimeCache("soft_reset");
        InvalidateRestartUiCache("soft_reset");
        return new { ok = true, actions, observation = BuildObservation() };
    }

    private object ResetCleanup(JsonElement root)
    {
        var board = FindBoard();
        if (board == null)
        {
            return new { ok = false, error = "Board not found." };
        }

        var actions = new List<string>();
        var guardObservation = BuildObservation(forceSeedProbe: true, forceRestartProbe: true);
        var allowActiveGameplayReset = AllowActiveGameplayReset(root);
        var resetReason = ReadString(root, "reset_reason", "resetReason") ?? "";
        if (ShouldBlockActiveGameplayMutation("cleanup_reward_ui", guardObservation, actions, allowActiveGameplayReset, resetReason))
        {
            return new
            {
                ok = false,
                blockedBySafety = true,
                cleanupSuccess = false,
                actions,
                observation = guardObservation,
                message = "Blocked reset cleanup while active gameplay is still running."
            };
        }

        board = DestroyDuplicateBoards(board, actions);
        var before = BuildResetCleanupReport(board);

        if (ReadBool(root, true, "destroy_stale", "destroyStale"))
        {
            var stalePlants = DestroyStalePlants(board, actions);
            var staleMowers = DestroyStaleMowers(board, actions);
            var staleBullets = DestroyStaleComponents(board.bulletArray, actions, "Bullet");
            var staleZombies = DestroyStaleComponents(board.zombieArray, actions, "Zombie");
            var staleGridItems = DestroyStaleComponents(board.griditemArray, actions, "GridItem");
            var rewardObjects = DestroyBlockingRewardObjects(board, actions);
            var duplicateMowers = DestroyDuplicateMowers(board, actions);
            actions.Add($"DestroyStaleSceneObjects(plants={stalePlants}, mowers={staleMowers}, duplicateMowers={duplicateMowers}, bullets={staleBullets}, zombies={staleZombies}, gridItems={staleGridItems}, rewardObjects={rewardObjects})");
        }

        if (ReadBool(root, true, "ensure_mowers", "ensureMowers"))
        {
            TryEnsureMowersInitialized(board, actions, allowActiveGameplayReset, resetReason);
        }

        if (ReadBool(root, true, "refresh_boxes", "refreshBoxes"))
        {
            RefreshAllBoardBoxes(board, actions);
        }

        if (ReadBool(root, true, "reset_card_cooldowns", "resetCardCooldowns"))
        {
            ResetCardCooldowns(actions);
        }

        if (ReadBool(root, false, "reset_counters", "resetCounters"))
        {
            TrySetBoardCounters(board, actions);
        }

        var after = BuildResetCleanupReport(board);
        if (ReadBool(root, true, "destroy_stale", "destroyStale") || ReadBool(root, true, "reset_card_cooldowns", "resetCardCooldowns"))
        {
            InvalidateSeedRuntimeCache("reset_cleanup");
            InvalidateRestartUiCache("reset_cleanup");
        }
        var cleanupSuccess = after.StaleVisiblePlantObjectCount == 0 &&
                             after.VisiblePlantObjectCount == after.LogicalPlantCount &&
                             after.StaleVisibleMowerObjectCount == 0 &&
                             after.DuplicateMowerRowCount == 0 &&
                             after.VisibleMowerObjectCount == after.LogicalMowerCount;

        return new
        {
            ok = true,
            before,
            after,
            actions,
            cleanupSuccess,
            message = "Unity destroys objects at frame end; Python re-observes after a short delay for final validation."
        };
    }

    private static bool IsPostWinScreenState(string? screenState) =>
        screenState is "level_complete_trophy" or "reward_unlock" or "reward_screen";

    private static bool IsLossScreenState(string? screenState) =>
        screenState is "game_over" or "game_over_restart_screen";

    private static bool HasPostWinEvidence(ObservationDto obs) =>
        obs.IsRewardScreen ||
        obs.IsNewPlantUnlockedScreen ||
        IsPostWinScreenState(obs.ScreenState);

    private static bool IsConfirmedTerminalOrTransitionState(ObservationDto obs)
    {
        if (HasPostWinEvidence(obs) && obs.TerminalHint != "running")
        {
            return true;
        }

        if (obs.Over || (!HasPostWinEvidence(obs) && (HasLossRestartEvidence(obs) || obs.NextStep == "click_restart")))
        {
            return true;
        }

        if (obs.TerminalHint == "game_over_or_loss")
        {
            return true;
        }

        if (!HasPostWinEvidence(obs) && IsLossScreenState(obs.ScreenState))
        {
            return true;
        }

        return obs.IsGameOverScreen && !HasPostWinEvidence(obs);
    }

    private static bool IsActiveGameplayUnsafeForMutation(ObservationDto obs)
    {
        if (!obs.BoardFound || obs.SeedSelectionActive || obs.OnSeedSelectionScreen || obs.Over)
        {
            return false;
        }

        if (IsConfirmedTerminalOrTransitionState(obs))
        {
            return false;
        }

        return obs.BoardStartMove &&
               HasActiveGameplayProgress(obs) &&
               (obs.GameplayReady ||
                obs.ActualGameplayReady ||
                obs.TerminalHint == "running" ||
                obs.NextStep == "play" ||
                obs.ScreenState == "gameplay");
    }

    private static bool HasActiveGameplayProgress(ObservationDto obs) =>
        obs.GameplayReady ||
        obs.ActualGameplayReady ||
        obs.Wave > 0 ||
        obs.KillCount > 0 ||
        obs.PlantCount > 0 ||
        obs.VisiblePlantObjectCount > 0 ||
        obs.ZombieCount > 0 ||
        obs.BulletCount > 0 ||
        obs.SeedSlotCount > 0 ||
        obs.ActiveGameplayCardBankCount > 0;

    private static bool IsLiveBoardRunning(ObservationDto obs) =>
        obs.BoardFound &&
        obs.TerminalHint == "running" &&
        !obs.Done &&
        !obs.Over &&
        (obs.Wave > 0 ||
         obs.PlantCount > 0 ||
         obs.VisiblePlantObjectCount > 0 ||
         obs.ZombieCount > 0 ||
         obs.BulletCount > 0 ||
         obs.KillCount > 0);

    private static bool AllowActiveGameplayReset(JsonElement root) =>
        ReadBool(
            root,
            false,
            "allow_active_gameplay_reset",
            "allowActiveGameplayReset",
            "allow_active_gameplay_mutation",
            "allowActiveGameplayMutation");

    private bool ShouldBlockActiveGameplayMutation(
        string actionName,
        ObservationDto obs,
        List<string> actions,
        bool allowActiveGameplayMutation = false,
        string resetReason = "")
    {
        var timeoutResetShortcut = IsTimeoutResetReason(resetReason) &&
                                   IsTimeoutResetShortcutMutation(actionName);
        if (!IsActiveGameplayUnsafeForMutation(obs))
        {
            return false;
        }

        if (timeoutResetShortcut)
        {
            var timeoutMessage =
                $"[safety] blocked {actionName} for timeout reset during active gameplay; " +
                "timeout truncation must pass through seed selection or Python hard reset. " +
                $"wave={obs.Wave}/{obs.MaxWave} zombies={obs.ZombieCount} plants={obs.PlantCount} " +
                $"mowers={obs.LogicalMowerCount}/{obs.VisibleMowerObjectCount} gameplayReady={obs.GameplayReady} " +
                $"screenState={obs.ScreenState} nextStep={obs.NextStep} " +
                $"done={obs.Done} over={obs.Over} terminalHint={obs.TerminalHint}";
            actions.Add(timeoutMessage);
            LoggerInstance.Msg(timeoutMessage);
            return true;
        }

        if (allowActiveGameplayMutation)
        {
            actions.Add(
                $"[safety] allowed {actionName} for explicit training reset boundary reason={resetReason} " +
                $"wave={obs.Wave}/{obs.MaxWave} zombies={obs.ZombieCount} plants={obs.PlantCount} " +
                $"mowers={obs.LogicalMowerCount}/{obs.VisibleMowerObjectCount} gameplayReady={obs.GameplayReady} " +
                $"screenState={obs.ScreenState} nextStep={obs.NextStep}");
            return false;
        }

        var message =
            $"[safety] blocked {actionName} during active gameplay " +
            $"wave={obs.Wave}/{obs.MaxWave} zombies={obs.ZombieCount} plants={obs.PlantCount} " +
            $"mowers={obs.LogicalMowerCount}/{obs.VisibleMowerObjectCount} gameplayReady={obs.GameplayReady} " +
            $"screenState={obs.ScreenState} nextStep={obs.NextStep} " +
            $"done={obs.Done} over={obs.Over} terminalHint={obs.TerminalHint}";
        actions.Add(message);
        LoggerInstance.Msg(message);
        return true;
    }

    private static bool IsTimeoutResetReason(string resetReason) =>
        string.Equals(resetReason?.Trim(), "timeout", StringComparison.OrdinalIgnoreCase);

    private static bool IsTimeoutResetShortcutMutation(string actionName)
    {
        var normalized = (actionName ?? string.Empty).Trim().ToLowerInvariant();
        return normalized == "reset" ||
               normalized == "soft reset" ||
               normalized == "board reinit";
    }

    private Board DestroyDuplicateBoards(Board primaryHint, List<string> actions)
    {
        List<Board> boards;
        try
        {
            boards = Object.FindObjectsOfType<Board>()
                .Where(candidate => candidate != null)
                .ToList();
        }
        catch (Exception ex)
        {
            actions.Add("DestroyDuplicateBoards scan failed: " + ex.Message);
            return primaryHint;
        }

        if (boards.Count <= 1)
        {
            return primaryHint;
        }

        var primary = SelectPrimaryBoard(boards, primaryHint);
        var primaryId = SafeInstanceId(primary);
        var destroyed = 0;
        foreach (var duplicate in boards)
        {
            if (SafeInstanceId(duplicate) == primaryId)
            {
                continue;
            }

            try
            {
                Object.Destroy(duplicate.gameObject);
                destroyed++;
            }
            catch (Exception ex)
            {
                actions.Add($"DestroyDuplicateBoard({SafeInstanceId(duplicate)}) failed: {ex.Message}");
            }
        }

        actions.Add($"DestroyDuplicateBoards(total={boards.Count}, kept={primaryId}, destroyed={destroyed})");
        return primary;
    }

    private Board SelectPrimaryBoard(List<Board> boards, Board primaryHint)
    {
        var hintId = SafeInstanceId(primaryHint);
        return boards
            .OrderByDescending(board => ScoreBoardForRetention(board, hintId))
            .ThenByDescending(board => SafeInstanceId(board))
            .First();
    }

    private int ScoreBoardForRetention(Board board, int hintId)
    {
        var score = SafeInstanceId(board) == hintId ? 10 : 0;
        try { score += board.gameObject != null && board.gameObject.activeInHierarchy ? 5 : 0; } catch { }
        try { score += board.over ? -100 : 100; } catch { }
        try { score += board.theWave <= 1 ? 25 : 0; } catch { }
        try { score += board.startMove ? 10 : 0; } catch { }
        try { score += (board.plantArray?.Count ?? 0) == 0 ? 10 : 0; } catch { }
        try { score += (board.zombieArray?.Count ?? 0) == 0 ? 5 : 0; } catch { }
        return score;
    }

    private static int SafeInstanceId(Object? unityObject)
    {
        try { return unityObject != null ? unityObject.GetInstanceID() : 0; }
        catch { return 0; }
    }

    private void TryEnsureMowersInitialized(
        Board board,
        List<string> actions,
        bool allowActiveGameplayMutation = false,
        string resetReason = "")
    {
        var guardObservation = BuildObservation(forceSeedProbe: true, forceRestartProbe: true);
        if (ShouldBlockActiveGameplayMutation("mower reinit", guardObservation, actions, allowActiveGameplayMutation, resetReason))
        {
            return;
        }

        var rows = SafePositive(board.rowNum, _config.FallbackRows);
        var mowerIds = GetMowerArrayIds(board, out var logicalMowerCount);
        var visibleMowerCount = ScanVisibleMowers(board, mowerIds)
            .Count(mower => mower.ActiveInHierarchy && mower.InBoardBounds);

        if (logicalMowerCount >= rows && visibleMowerCount >= rows)
        {
            actions.Add($"Mowers already initialized(logical={logicalMowerCount}, visible={visibleMowerCount}, rows={rows})");
            return;
        }

        if (logicalMowerCount > 0 || visibleMowerCount > 0)
        {
            actions.Add($"Partial mower state exists(logical={logicalMowerCount}, visible={visibleMowerCount}, rows={rows}); attempting InitMower refill.");
            var partialInitBoard = FindInitBoard();
            if (partialInitBoard == null)
            {
                actions.Add("InitBoard not found for partial mower refill.");
                return;
            }

            if (TryInvokeMethod(partialInitBoard, "InitMower", Array.Empty<object>(), actions))
            {
                actions.Add($"InitBoard.InitMower() invoked after partial mower reset state(rows={rows})");
                var duplicateMowers = DestroyDuplicateMowers(board, actions);
                if (duplicateMowers > 0)
                {
                    actions.Add($"DestroyDuplicateMowers(after partial refill count={duplicateMowers})");
                }
            }
            else
            {
                actions.Add("InitBoard.InitMower() was not invoked for partial mower refill.");
            }
            return;
        }

        var initBoard = FindInitBoard();
        if (initBoard == null)
        {
            actions.Add("InitBoard not found for InitMower().");
            return;
        }

        if (TryInvokeMethod(initBoard, "InitMower", Array.Empty<object>(), actions))
        {
            actions.Add($"InitBoard.InitMower() invoked after empty mower reset state(rows={rows})");
        }
        else
        {
            actions.Add("InitBoard.InitMower() was not invoked.");
        }
    }

    private void ManualClearDynamicBoardObjects(Board board, List<string> actions)
    {
        var destroyed = new HashSet<int>();

        var plants = DestroyPlantList(board.plantArray, board, destroyed);
        plants += DestroyPlantList(board.hiddenPlants, board, destroyed);
        plants += DestroyPlantList(board.plantHead, board, destroyed);
        var bullets = DestroyComponentList(board.bulletArray, destroyed);
        var zombies = DestroyComponentList(board.zombieArray, destroyed);
        zombies += DestroyComponentList(board.zombieHead, destroyed);
        var obstacles = DestroyComponentList(board.zombieBalls, destroyed);
        var gridItems = DestroyComponentList(board.griditemArray, destroyed);

        TrySetBoardCounters(board, actions);
        actions.Add($"ManualClear(plants={plants}, bullets={bullets}, zombies={zombies}, obstacles={obstacles}, gridItems={gridItems})");
    }

    private ResetCleanupReport BuildResetCleanupReport(Board board)
    {
        var report = new ResetCleanupReport
        {
            BoardFound = board != null
        };

        if (board == null)
        {
            return report;
        }

        report.LogicalPlantCount = board.plantArray?.Count ?? 0;
        report.LogicalZombieCount = board.zombieArray?.Count ?? 0;
        report.LogicalBulletCount = board.bulletArray?.Count ?? 0;
        report.LogicalGridItemCount = board.griditemArray?.Count ?? 0;
        var mowerIds = GetMowerArrayIds(board, out var logicalMowerCount);
        report.LogicalMowerCount = logicalMowerCount;

        var primaryPlantIds = new HashSet<int>();
        AddComponentIds(primaryPlantIds, board.plantArray);
        foreach (var visiblePlant in ScanVisiblePlants(board, primaryPlantIds))
        {
            report.VisiblePlants.Add(visiblePlant);
        }
        foreach (var visibleMower in ScanVisibleMowers(board, mowerIds))
        {
            report.VisibleMowers.Add(visibleMower);
        }

        report.VisiblePlantObjectCount = report.VisiblePlants.Count(p => p.ActiveInHierarchy && p.InBoardBounds);
        report.StaleVisiblePlantObjectCount = report.VisiblePlants.Count(p => p.ActiveInHierarchy && p.InBoardBounds && !p.InPlantArray);
        report.VisibleMowerObjectCount = report.VisibleMowers.Count(m => m.ActiveInHierarchy && m.InBoardBounds);
        report.StaleVisibleMowerObjectCount = report.VisibleMowers.Count(m => m.ActiveInHierarchy && m.InBoardBounds && !m.InMowerArray);
        foreach (var group in report.VisibleMowers.Where(m => m.ActiveInHierarchy && m.InBoardBounds).GroupBy(m => m.Row))
        {
            if (group.Count() > 1)
            {
                report.DuplicateMowerRows.Add(group.Key);
            }
        }
        report.DuplicateMowerRowCount = report.DuplicateMowerRows.Count;
        report.SceneZombieObjectCount = CountActiveComponents<Zombie>();
        report.SceneBulletObjectCount = CountActiveComponents<Bullet>();
        report.SceneGridItemObjectCount = CountActiveComponents<GridItem>();
        report.SceneMowerObjectCount = CountActiveComponents<Mower>();
        return report;
    }

    private List<VisiblePlantDto> ScanVisiblePlants(Board board, HashSet<int>? primaryPlantIds = null)
    {
        primaryPlantIds ??= new HashSet<int>();
        var result = new List<VisiblePlantDto>();
        try
        {
            var plants = Object.FindObjectsOfType<Plant>();
            foreach (var plant in plants)
            {
                try
                {
                    if (plant == null || plant.gameObject == null || !plant.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    var id = plant.GetInstanceID();
                    var position = plant.transform.position;
                    var row = plant.thePlantRow;
                    var column = plant.thePlantColumn;
                    var inBounds = IsPlantInBoardBounds(plant, board);
                    result.Add(new VisiblePlantDto
                    {
                        InstanceId = id,
                        Type = (int)plant.thePlantType,
                        TypeName = plant.thePlantType.ToString(),
                        Row = row,
                        Column = column,
                        X = position.x,
                        Y = position.y,
                        ActiveInHierarchy = true,
                        InBoardBounds = inBounds,
                        InPlantArray = primaryPlantIds.Contains(id)
                    });
                }
                catch
                {
                    // Ignore stale wrappers while scenes are changing.
                }
            }
        }
        catch
        {
            // Scene scans are diagnostics; observation should still succeed without them.
        }

        return result;
    }

    private bool IsPlantInBoardBounds(Plant plant, Board board)
    {
        try
        {
            var row = plant.thePlantRow;
            var column = plant.thePlantColumn;
            var rows = SafePositive(board.rowNum, _config.FallbackRows);
            var columns = SafePositive(board.columnNum, _config.FallbackColumns);
            return row >= 0 && row < rows && column >= 0 && column < columns;
        }
        catch
        {
            return false;
        }
    }

    private int DestroyStalePlants(Board board, List<string> actions)
    {
        var knownIds = new HashSet<int>();
        AddComponentIds(knownIds, board.plantArray);
        AddComponentIds(knownIds, board.hiddenPlants);
        AddComponentIds(knownIds, board.plantHead);

        var destroyed = 0;
        try
        {
            var plants = Object.FindObjectsOfType<Plant>();
            foreach (var plant in plants)
            {
                try
                {
                    if (plant == null || plant.gameObject == null || !plant.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    var id = plant.GetInstanceID();
                    if (knownIds.Contains(id) || !IsPlantInBoardBounds(plant, board))
                    {
                        continue;
                    }

                    try { plant.Die(Plant.DieReason.Default); } catch { }
                    try { plant.RemoveFromList(); } catch { }
                    try { board.OnPlantDie(plant); } catch { }
                    try { board.UpdateBox(plant.thePlantColumn, plant.thePlantRow); } catch { }
                    Object.Destroy(plant.gameObject);
                    destroyed++;
                }
                catch
                {
                    // Continue clearing other stale objects.
                }
            }
        }
        catch (Exception ex)
        {
            actions.Add("FindObjectsOfType<Plant>() failed during cleanup: " + ex.Message);
        }

        return destroyed;
    }

    private List<VisibleMowerDto> ScanVisibleMowers(Board board, HashSet<int>? mowerArrayIds = null)
    {
        mowerArrayIds ??= new HashSet<int>();
        var result = new List<VisibleMowerDto>();
        try
        {
            var mowers = Object.FindObjectsOfType<Mower>();
            foreach (var mower in mowers)
            {
                try
                {
                    if (mower == null || mower.gameObject == null || !mower.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    var id = mower.GetInstanceID();
                    var position = mower.transform.position;
                    var row = SafeReadMowerRow(mower);
                    result.Add(new VisibleMowerDto
                    {
                        InstanceId = id,
                        Type = SafeReadMowerType(mower),
                        TypeName = SafeReadMowerTypeName(mower),
                        Row = row,
                        X = position.x,
                        Y = position.y,
                        MowerX = SafeReadMowerX(mower),
                        ActiveInHierarchy = true,
                        InBoardBounds = IsMowerInBoardBounds(mower, board),
                        InMowerArray = mowerArrayIds.Contains(id)
                    });
                }
                catch
                {
                    // Ignore stale wrappers while scenes are changing.
                }
            }
        }
        catch
        {
            // Scene scans are diagnostics; observation should still succeed without them.
        }

        return result;
    }

    private bool IsMowerInBoardBounds(Mower mower, Board board)
    {
        try
        {
            var row = mower.theMowerRow;
            var rows = SafePositive(board.rowNum, _config.FallbackRows);
            return row >= 0 && row < rows;
        }
        catch
        {
            return false;
        }
    }

    private int DestroyStaleMowers(Board board, List<string> actions)
    {
        var mowerIds = GetMowerArrayIds(board, out _);
        var destroyed = 0;
        try
        {
            var mowers = Object.FindObjectsOfType<Mower>();
            foreach (var mower in mowers)
            {
                try
                {
                    if (mower == null || mower.gameObject == null || !mower.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    var id = mower.GetInstanceID();
                    if (!IsMowerInBoardBounds(mower, board))
                    {
                        continue;
                    }

                    var stale = !mowerIds.Contains(id);
                    if (!stale)
                    {
                        continue;
                    }

                    Object.Destroy(mower.gameObject);
                    destroyed++;
                }
                catch
                {
                    // Continue clearing other stale mowers.
                }
            }
        }
        catch (Exception ex)
        {
            actions.Add("FindObjectsOfType<Mower>() failed during cleanup: " + ex.Message);
        }

        return destroyed;
    }

    private int DestroyDuplicateMowers(Board board, List<string> actions)
    {
        var rows = SafePositive(board.rowNum, _config.FallbackRows);
        var keptRows = new HashSet<int>();
        var removeIndexes = new List<int>();
        var removeObjects = new List<Mower>();
        try
        {
            var mowers = board.mowerArray;
            if (mowers == null)
            {
                return 0;
            }

            for (var i = 0; i < mowers.Count; i++)
            {
                try
                {
                    var mower = mowers[i];
                    if (mower == null)
                    {
                        removeIndexes.Add(i);
                        continue;
                    }

                    var row = SafeReadMowerRow(mower);
                    if (row < 0 || row >= rows || keptRows.Contains(row))
                    {
                        removeIndexes.Add(i);
                        removeObjects.Add(mower);
                        continue;
                    }

                    keptRows.Add(row);
                }
                catch
                {
                    removeIndexes.Add(i);
                }
            }

            foreach (var index in removeIndexes.OrderByDescending(index => index))
            {
                try
                {
                    mowers.RemoveAt(index);
                }
                catch (Exception ex)
                {
                    actions.Add($"mowerArray.RemoveAt({index}) failed: {ex.Message}");
                }
            }

            var destroyedIds = new HashSet<int>();
            foreach (var mower in removeObjects)
            {
                try
                {
                    if (mower == null || mower.gameObject == null)
                    {
                        continue;
                    }

                    var id = mower.GetInstanceID();
                    if (!destroyedIds.Add(id))
                    {
                        continue;
                    }

                    Object.Destroy(mower.gameObject);
                }
                catch
                {
                    // Continue destroying duplicate mower objects.
                }
            }
        }
        catch (Exception ex)
        {
            actions.Add("DestroyDuplicateMowers failed: " + ex.Message);
        }

        return removeObjects.Count;
    }

    private int DestroyBlockingRewardObjects(Board board, List<string> actions)
    {
        var destroyed = 0;
        var targets = new HashSet<GameObject>();
        try
        {
            foreach (var transform in Object.FindObjectsOfType<Transform>())
            {
                try
                {
                    if (transform == null || transform.gameObject == null || !transform.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    WorldToScreen(transform.position, out var inBounds);
                    if (!inBounds)
                    {
                        continue;
                    }

                    var gameObject = transform.gameObject;
                    var signal = new UiProbeEntryDto
                    {
                        Name = SafeObjectName(gameObject),
                        ClassName = string.Join(",", SafeComponentTypeNames(gameObject)),
                        Text = SafeReadGameObjectText(gameObject),
                        ActiveSelf = gameObject.activeSelf,
                        ActiveInHierarchy = gameObject.activeInHierarchy,
                        RendererVisible = HasVisibleRenderer(gameObject),
                        UiVisible = true,
                        ParentName = transform.parent != null ? SafeObjectName(transform.parent.gameObject) : null,
                        RootName = transform.root != null ? SafeObjectName(transform.root.gameObject) : null,
                        HierarchyPath = BuildHierarchyPath(transform),
                        InScreenBounds = true
                    };
                    if (!IsBlockingRewardUiSignal(signal))
                    {
                        continue;
                    }

                    var target = FindBlockingRewardRoot(gameObject, board);
                    if (target != null)
                    {
                        targets.Add(target);
                    }
                }
                catch { }
            }

            foreach (var target in targets)
            {
                try
                {
                    actions.Add($"DestroyBlockingRewardObject({BuildHierarchyPath(target.transform)})");
                    Object.Destroy(target);
                    destroyed++;
                }
                catch (Exception ex)
                {
                    actions.Add($"DestroyBlockingRewardObject failed for {SafeObjectName(target)}: {ex.Message}");
                }
            }
        }
        catch (Exception ex)
        {
            actions.Add("FindObjectsOfType<Transform>() failed during reward cleanup: " + ex.Message);
        }

        return destroyed;
    }

    private static GameObject? FindBlockingRewardRoot(GameObject gameObject, Board board)
    {
        GameObject? candidate = null;
        try
        {
            var boardObject = board.gameObject;
            var current = gameObject.transform;
            var guard = 0;
            while (current != null && guard++ < 32)
            {
                var currentObject = current.gameObject;
                if (currentObject == boardObject)
                {
                    break;
                }

                var text = NormalizeUiText($"{SafeObjectName(currentObject)} {BuildHierarchyPath(current)}");
                if (text.Contains("trophy") ||
                    text.Contains("reward") ||
                    text.Contains("prize") ||
                    text.Contains("award") ||
                    text.Contains("levelcomplete") ||
                    text.Contains("levelcompleted") ||
                    text.Contains("newplant"))
                {
                    candidate = currentObject;
                }

                current = current.parent;
            }
        }
        catch
        {
            return gameObject;
        }

        return candidate ?? gameObject;
    }

    private HashSet<int> GetMowerArrayIds(Board board, out int logicalCount)
    {
        logicalCount = 0;
        var ids = new HashSet<int>();
        try
        {
            var mowers = board.mowerArray;
            if (mowers == null)
            {
                return ids;
            }

            logicalCount = mowers.Count;
            for (var i = 0; i < mowers.Count; i++)
            {
                try
                {
                    var mower = mowers[i];
                    if (mower != null)
                    {
                        ids.Add(mower.GetInstanceID());
                    }
                }
                catch
                {
                    // Ignore stale wrappers.
                }
            }
        }
        catch
        {
            logicalCount = ids.Count;
        }

        return ids;
    }

    private static int SafeReadMowerRow(Mower mower)
    {
        try { return mower.theMowerRow; }
        catch { return -1; }
    }

    private static int SafeReadMowerType(Mower mower)
    {
        try { return (int)mower.theMowerType; }
        catch { return -1; }
    }

    private static string SafeReadMowerTypeName(Mower mower)
    {
        try { return mower.theMowerType.ToString(); }
        catch { return "unknown"; }
    }

    private static float SafeReadMowerX(Mower mower)
    {
        try { return mower.transform.position.x; }
        catch { return 0f; }
    }

    private int DestroyStaleComponents<T>(Il2CppSystem.Collections.Generic.List<T>? liveList, List<string> actions, string label)
        where T : Component
    {
        var knownIds = new HashSet<int>();
        AddComponentIds(knownIds, liveList);
        var destroyed = 0;
        try
        {
            var components = Object.FindObjectsOfType<T>();
            foreach (var item in components)
            {
                try
                {
                    if (item == null || item.gameObject == null || !item.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    var id = item.GetInstanceID();
                    if (knownIds.Contains(id))
                    {
                        continue;
                    }

                    Object.Destroy(item.gameObject);
                    destroyed++;
                }
                catch
                {
                    // Continue clearing other stale objects.
                }
            }
        }
        catch (Exception ex)
        {
            actions.Add($"FindObjectsOfType<{label}>() failed during cleanup: " + ex.Message);
        }

        return destroyed;
    }

    private void AddComponentIds<T>(HashSet<int> ids, Il2CppSystem.Collections.Generic.List<T>? list)
        where T : Component
    {
        if (list == null)
        {
            return;
        }

        for (var i = 0; i < list.Count; i++)
        {
            try
            {
                var item = list[i];
                if (item != null)
                {
                    ids.Add(item.GetInstanceID());
                }
            }
            catch
            {
                // Ignore stale wrappers.
            }
        }
    }

    private int CountActiveComponents<T>()
        where T : Component
    {
        var count = 0;
        try
        {
            var components = Object.FindObjectsOfType<T>();
            foreach (var item in components)
            {
                try
                {
                    if (item != null && item.gameObject != null && item.gameObject.activeInHierarchy)
                    {
                        count++;
                    }
                }
                catch
                {
                    // Ignore stale wrappers.
                }
            }
        }
        catch
        {
            // Diagnostics only.
        }

        return count;
    }

    private int DestroyPlantList(Il2CppSystem.Collections.Generic.List<Plant>? list, Board board, HashSet<int> destroyed)
    {
        if (list == null)
        {
            return 0;
        }

        var removed = list.Count;
        for (var i = list.Count - 1; i >= 0; i--)
        {
            try
            {
                var plant = list[i];
                if (plant == null)
                {
                    continue;
                }

                try { plant.Die(Plant.DieReason.Default); } catch { }
                try { plant.RemoveFromList(); } catch { }
                try { board.OnPlantDie(plant); } catch { }
                try { board.UpdateBox(plant.thePlantColumn, plant.thePlantRow); } catch { }

                var id = plant.GetInstanceID();
                if (destroyed.Add(id))
                {
                    Object.Destroy(plant.gameObject);
                }
            }
            catch
            {
                // Continue clearing other wrappers; stale IL2CPP wrappers can throw during scene transitions.
            }
        }

        try { list.Clear(); } catch { }
        return removed;
    }

    private int DestroyComponentList<T>(Il2CppSystem.Collections.Generic.List<T>? list, HashSet<int> destroyed)
        where T : Component
    {
        if (list == null)
        {
            return 0;
        }

        var removed = list.Count;
        for (var i = list.Count - 1; i >= 0; i--)
        {
            try
            {
                var item = list[i];
                if (item == null)
                {
                    continue;
                }

                var id = item.GetInstanceID();
                if (destroyed.Add(id))
                {
                    Object.Destroy(item.gameObject);
                }
            }
            catch
            {
                // Continue clearing other wrappers; stale IL2CPP wrappers can throw during scene transitions.
            }
        }

        try { list.Clear(); } catch { }
        return removed;
    }

    private void TrySetBoardCounters(Board board, List<string> actions)
    {
        try { board.killZombieCount = 0; } catch { }
        try { board.theTotalNumOfPlant = 0; } catch { }
        try { board.plantedCount = 0; } catch { }
        try { board.theCurrentPlantCount = 0; } catch { }
        try { board.currentBulletNum = 0; } catch { }
        try { board.theWave = 0; } catch { }
        try { board.moreZombiesComing = false; } catch { }
        try { board.newZombieWaveCountDown = 5f; } catch { }
        try { board.nextZombieWaveCountDown = 5f; } catch { }
        try { board.hugeWaveCountDown = 0f; } catch { }
        try { board.isHugeWave = false; } catch { }
        try { board.over = false; } catch { }
        try { board.time = 0f; } catch { }
        actions.Add("ResetCounters(killZombieCount,theTotalNumOfPlant,plantedCount,theCurrentPlantCount,currentBulletNum,theWave,waveTimers,over,time)");
    }

    private void RefreshAllBoardBoxes(Board board, List<string> actions)
    {
        var rows = SafePositive(board.rowNum, _config.FallbackRows);
        var columns = SafePositive(board.columnNum, _config.FallbackColumns);
        var refreshed = 0;
        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                try
                {
                    board.UpdateBox(column, row);
                    refreshed++;
                }
                catch
                {
                    // Keep reset best-effort; some scenes may not have initialized box info yet.
                }
            }
        }

        actions.Add($"Board.UpdateBox(all cells refreshed={refreshed})");
    }

    private bool TryInvokeLoseMenuRestart(
        List<string> actions,
        RestartScreenInfo? restartInfo,
        out string methodUsed,
        out string targetName,
        out string targetPath,
        out string error)
    {
        methodUsed = "";
        targetName = "";
        targetPath = "";
        error = "";
        try
        {
            var buttons = Object.FindObjectsOfType<LoseMenuBtn>();
            foreach (var button in buttons)
            {
                try
                {
                    if (button == null)
                    {
                        continue;
                    }

                    if (button.type != LoseMenuBtn.LoseBtnType.Restart &&
                        button.type != LoseMenuBtn.LoseBtnType.TryAgain)
                    {
                        continue;
                    }

                    try { button.OnMouseDown(); } catch { }
                    button.OnMouseUp();
                    methodUsed = $"LoseMenuBtn.{button.type}.OnMouseUp";
                    targetName = SafeObjectName(button.gameObject) ?? button.type.ToString();
                    targetPath = BuildHierarchyPath(button.transform);
                    actions.Add($"{methodUsed}({targetPath})");
                    return true;
                }
                catch (Exception ex)
                {
                    actions.Add("LoseMenuBtn restart candidate failed: " + ex.Message);
                    error = ex.Message;
                }
            }

            actions.Add("LoseMenuBtn restart button not found.");
        }
        catch (Exception ex)
        {
            actions.Add("FindObjectsOfType<LoseMenuBtn>() failed: " + ex.Message);
            error = ex.Message;
        }

        if (TryClickRestartLikeObject(restartInfo, actions, out var fallbackMethod, out var fallbackName, out var fallbackPath, out var fallbackError))
        {
            methodUsed = fallbackMethod;
            targetName = fallbackName;
            targetPath = fallbackPath;
            actions.Add($"Restart fallback clicked via {fallbackMethod}: {fallbackPath}");
            return true;
        }

        if (string.IsNullOrWhiteSpace(error))
        {
            error = fallbackError;
        }

        return false;
    }

    private bool IsLossScreenActive(bool broadScan = false)
    {
        var info = DetectRestartScreenInfo(broadScan);
        return HasLossRestartEvidence(info);
    }

    private bool IsRestartButtonActive(bool broadScan = false)
    {
        return DetectRestartScreenInfo(broadScan).RestartButtonActive;
    }

    private bool IsLoseMenuButtonVisible()
    {
        try
        {
            foreach (var button in Object.FindObjectsOfType<LoseMenuBtn>())
            {
                try
                {
                    if (button == null || button.gameObject == null || !button.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    WorldToScreen(button.transform.position, out var inBounds);
                    if (inBounds)
                    {
                        return true;
                    }
                }
                catch { }
            }
        }
        catch { }

        return false;
    }

    private RestartScreenInfo DetectRestartScreenInfo(bool broadScan = false) =>
        broadScan ? DetectRestartScreenSlowDiagnostic() : DetectRestartScreenFastCached();

    private RestartScreenInfo DetectRestartScreenFastCached()
    {
        var watch = Stopwatch.StartNew();
        var info = new RestartScreenInfo { RestartDetectionMode = "fast_cached" };
        RefreshRestartUiCacheTypedIfNeeded("fast_cached");
        ApplyCachedRestartInfo(info);
        FinalizeRestartScreenInfo(info);
        watch.Stop();
        info.screen_check_ms = Math.Round(watch.Elapsed.TotalMilliseconds, 3);
        return info;
    }

    private RestartScreenInfo DetectRestartScreenSlowDiagnostic()
    {
        var watch = Stopwatch.StartNew();
        var info = new RestartScreenInfo { RestartDetectionMode = "slow_diagnostic" };
        RefreshRestartUiCacheTyped("slow_diagnostic");
        ApplyCachedRestartInfo(info);

        GameObject? restartButton = null;
        string restartName = "";
        string restartPath = "";
        string restartReason = "";
        var restartButtonTerminalContext = false;
        var gameOverDetected = false;
        var gameOverReason = "";
        var pauseDetected = info.PauseMenuActive;
        var pauseRestartDetected = info.PauseRestartButtonActive;

        try
        {
            foreach (var transform in Object.FindObjectsOfType<Transform>())
            {
                try
                {
                    if (transform == null || transform.gameObject == null || !transform.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    WorldToScreen(transform.position, out var inBounds);
                    if (!inBounds)
                    {
                        continue;
                    }

                    var path = BuildHierarchyPath(transform);
                    var text = SafeReadGameObjectText(transform.gameObject);
                    var componentTypes = string.Join(",", SafeComponentTypeNames(transform.gameObject));
                    var raw = $"{SafeObjectName(transform.gameObject)} {path} {text} {componentTypes}";
                    var normalized = NormalizeUiText(raw);
                    var pauseContext = LooksLikePauseContext(normalized);
                    pauseDetected = pauseDetected || pauseContext;

                    if (!gameOverDetected &&
                        (normalized.Contains("gameover") ||
                         normalized.Contains("youlose") ||
                         normalized.Contains("defeat") ||
                         normalized.Contains("failed")))
                    {
                        gameOverDetected = true;
                        gameOverReason = $"game-over text/object '{SafeObjectName(transform.gameObject)}' at {path}";
                        _restartUiCache.GameOverTextObject = transform.gameObject;
                    }

                    if (LooksLikeRestartControl(normalized))
                    {
                        if (pauseContext)
                        {
                            pauseRestartDetected = true;
                            continue;
                        }

                        var terminalContext = normalized.Contains("lose") ||
                                              normalized.Contains("loss") ||
                                              normalized.Contains("gameover") ||
                                              normalized.Contains("gamefail") ||
                                              normalized.Contains("failed") ||
                                              normalized.Contains("defeat");
                        if (restartButton == null && (terminalContext || gameOverDetected || info.LossMenuActive))
                        {
                            restartButton = transform.gameObject;
                            restartName = SafeObjectName(transform.gameObject) ?? "";
                            restartPath = path;
                            restartReason = $"loss restart-like visible object '{restartName}' at {restartPath}";
                            restartButtonTerminalContext = terminalContext;
                        }
                    }
                }
                catch { }
            }
        }
        catch (Exception ex)
        {
            info.RestartDetectionReason = "UI restart scan failed: " + ex.Message;
        }

        info.GameOverTextVisible = info.GameOverTextVisible || gameOverDetected;
        info.PauseMenuActive = info.PauseMenuActive || pauseDetected;
        info.OnPauseMenu = info.OnPauseMenu || pauseDetected;
        info.PauseRestartButtonActive = info.PauseRestartButtonActive || pauseRestartDetected;
        info.OnGameOverScreen = info.OnGameOverScreen || gameOverDetected;
        if (restartButton != null && (restartButtonTerminalContext || gameOverDetected || info.LossMenuActive))
        {
            info.LossMenuActive = info.LossMenuActive || restartButtonTerminalContext;
            info.RestartButtonActive = true;
            info.RestartButtonObject = restartButton;
            info.RestartButtonName = restartName;
            info.RestartButtonPath = restartPath;
        }

        info.RestartDetectionReason = string.Join(
            "; ",
            new[] { gameOverReason, restartReason, info.RestartDetectionReason }
                .Where(part => !string.IsNullOrWhiteSpace(part)));
        FinalizeRestartScreenInfo(info);
        watch.Stop();
        info.screen_check_ms = Math.Round(watch.Elapsed.TotalMilliseconds, 3);
        return info;
    }

    private void RefreshRestartUiCacheTypedIfNeeded(string reason)
    {
        var sceneKey = GetActiveSceneKey();
        var sceneChanged = !string.Equals(_restartUiCache.SceneKey, sceneKey, StringComparison.Ordinal);
        if (sceneChanged ||
            !_restartUiCache.Valid ||
            !IsUnityObjectAlive(_restartUiCache.LossRestartButton) ||
            !IsUnityObjectAlive(_restartUiCache.PauseRestartButton))
        {
            RefreshRestartUiCacheTyped(reason);
        }
    }

    private void RefreshRestartUiCacheTyped(string reason)
    {
        _restartUiCache.Valid = true;
        _restartUiCache.SceneKey = GetActiveSceneKey();
        _restartUiCache.InvalidReason = reason;
        _restartUiCache.LossRestartButton = null;
        _restartUiCache.LossMenuRoot = null;
        _restartUiCache.PauseRestartButton = null;
        _restartUiCache.PauseMenuRoot = null;

        try
        {
            foreach (var button in Object.FindObjectsOfType<LoseMenuBtn>())
            {
                try
                {
                    if (button == null || button.gameObject == null)
                    {
                        continue;
                    }

                    if (button.type != LoseMenuBtn.LoseBtnType.Restart &&
                        button.type != LoseMenuBtn.LoseBtnType.TryAgain)
                    {
                        continue;
                    }

                    _restartUiCache.LossRestartButton = button;
                    _restartUiCache.LossMenuRoot = button.transform?.root?.gameObject;
                    break;
                }
                catch { }
            }
        }
        catch { }

        try
        {
            foreach (var button in Object.FindObjectsOfType<PauseMenu_Btn>())
            {
                try
                {
                    if (button == null || button.gameObject == null)
                    {
                        continue;
                    }

                    _restartUiCache.PauseRestartButton = button;
                    _restartUiCache.PauseMenuRoot = button.transform?.root?.gameObject;
                    break;
                }
                catch { }
            }
        }
        catch { }
    }

    private void InvalidateRestartUiCache(string reason)
    {
        _restartUiCache.Valid = false;
        _restartUiCache.InvalidReason = reason;
        _restartUiCache.LossRestartButton = null;
        _restartUiCache.LossMenuRoot = null;
        _restartUiCache.PauseRestartButton = null;
        _restartUiCache.PauseMenuRoot = null;
        _restartUiCache.GameOverTextObject = null;
    }

    private void ApplyCachedRestartInfo(RestartScreenInfo info)
    {
        var lossButton = _restartUiCache.LossRestartButton;
        if (IsActiveComponent(lossButton))
        {
            SetLossButtonInfo(info, lossButton!);
        }

        if (IsActiveGameObject(_restartUiCache.GameOverTextObject))
        {
            info.GameOverTextVisible = true;
            info.OnGameOverScreen = true;
        }

        var pauseButton = _restartUiCache.PauseRestartButton;
        if (IsActiveComponent(pauseButton))
        {
            SetPauseButtonInfo(info, pauseButton!);
        }
    }

    private void SetLossButtonInfo(RestartScreenInfo info, LoseMenuBtn button)
    {
        info.LossMenuActive = true;
        info.OnGameOverScreen = true;
        info.OnRestartScreen = true;
        info.RestartButtonActive = true;
        info.RestartButtonObject = button.gameObject;
        info.RestartButtonName = SafeObjectName(button.gameObject) ?? button.type.ToString();
        info.RestartButtonPath = button.transform != null ? BuildHierarchyPath(button.transform) : null;
        info.RestartDetectionReason = $"LoseMenuBtn.{button.type}";
    }

    private static void SetPauseButtonInfo(RestartScreenInfo info, PauseMenu_Btn button)
    {
        info.OnPauseMenu = true;
        info.PauseMenuActive = true;
        info.PauseRestartButtonActive = true;
    }

    private static void FinalizeRestartScreenInfo(RestartScreenInfo info)
    {
        var hasLossEvidence = HasLossRestartEvidence(info);
        if (info.PauseMenuActive && !hasLossEvidence)
        {
            info.OnGameOverScreen = false;
            info.OnRestartScreen = false;
            info.LossMenuActive = false;
            info.GameOverTextVisible = false;
            info.RestartButtonActive = false;
            info.RestartButtonObject = null;
            info.RestartButtonName = null;
            info.RestartButtonPath = null;
            return;
        }

        info.OnGameOverScreen = hasLossEvidence;
        info.OnRestartScreen = info.RestartButtonActive && hasLossEvidence;
    }

    private static bool HasLossRestartEvidence(RestartScreenInfo info) =>
        info.OnGameOverScreen ||
        info.LossMenuActive ||
        (info.RestartButtonActive && info.GameOverTextVisible);

    private static bool HasLossRestartEvidence(ObservationDto obs) =>
        obs.OnGameOverScreen ||
        obs.LossMenuActive ||
        (obs.RestartButtonActive && obs.GameOverTextVisible);

    private static bool IsUnityObjectAlive(Object? target)
    {
        try { return target != null; }
        catch { return false; }
    }

    private static bool IsActiveComponent(Component? component)
    {
        try
        {
            return component != null &&
                   component.gameObject != null &&
                   component.gameObject.activeInHierarchy;
        }
        catch
        {
            return false;
        }
    }

    private static bool IsActiveGameObject(GameObject? gameObject)
    {
        try { return gameObject != null && gameObject.activeInHierarchy; }
        catch { return false; }
    }

    private static string GetActiveSceneKey()
    {
        try
        {
            var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
            return $"{scene.buildIndex}:{scene.name}";
        }
        catch
        {
            return "unknown";
        }
    }

    private static bool LooksLikeRestartControl(string normalized)
    {
        return normalized.Contains("restart") ||
               normalized.Contains("reset") ||
               normalized.Contains("tryagain") ||
               normalized.Contains("retry") ||
               normalized.Contains("playagain");
    }

    private static bool LooksLikePauseContext(string normalized)
    {
        return normalized.Contains("pausemenu") ||
               normalized.Contains("pause_menu") ||
               (normalized.Contains("pause") &&
                (normalized.Contains("resume") ||
                 normalized.Contains("mainmenu") ||
                 normalized.Contains("options") ||
                 normalized.Contains("restart") ||
                 normalized.Contains("menu")));
    }

    private bool TryFindRestartLikeObject(out GameObject? gameObject, out string hierarchyPath, bool requireTerminalContext = true)
    {
        gameObject = null;
        hierarchyPath = "";
        var info = DetectRestartScreenInfo(broadScan: true);
        if (info.RestartButtonObject != null && (!requireTerminalContext || HasLossRestartEvidence(info)))
        {
            gameObject = info.RestartButtonObject;
            hierarchyPath = info.RestartButtonPath ?? "";
            return true;
        }

        try
        {
            foreach (var transform in Object.FindObjectsOfType<Transform>())
            {
                try
                {
                    if (transform == null || transform.gameObject == null || !transform.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    WorldToScreen(transform.position, out var inBounds);
                    if (!inBounds)
                    {
                        continue;
                    }

                    var path = BuildHierarchyPath(transform);
                    var text = SafeReadGameObjectText(transform.gameObject);
                    var normalized = NormalizeUiText($"{SafeObjectName(transform.gameObject)} {path} {text} {string.Join(",", SafeComponentTypeNames(transform.gameObject))}");
                    var looksRestart = LooksLikeRestartControl(normalized) || normalized.Contains("again");
                    if (!looksRestart || LooksLikePauseContext(normalized))
                    {
                        continue;
                    }

                    var looksTerminal = normalized.Contains("lose") ||
                                        normalized.Contains("loss") ||
                                        normalized.Contains("gameover") ||
                                        normalized.Contains("gamefail") ||
                                        normalized.Contains("failed") ||
                                        normalized.Contains("defeat") ||
                                        normalized.Contains("terminal");
                    if (requireTerminalContext && !looksTerminal)
                    {
                        continue;
                    }

                    gameObject = transform.gameObject;
                    hierarchyPath = path;
                    return true;
                }
                catch { }
            }
        }
        catch { }

        return false;
    }

    private bool TryClickRestartLikeObject(
        RestartScreenInfo? restartInfo,
        List<string> actions,
        out string methodUsed,
        out string targetName,
        out string hierarchyPath,
        out string error)
    {
        methodUsed = "";
        targetName = "";
        hierarchyPath = "";
        error = "";
        var gameObject = restartInfo != null && HasLossRestartEvidence(restartInfo)
            ? restartInfo.RestartButtonObject
            : null;
        if (gameObject == null && (!TryFindRestartLikeObject(out gameObject, out hierarchyPath, requireTerminalContext: true) || gameObject == null))
        {
            error = "Restart fallback scan found no visible loss-specific restart-like object.";
            actions.Add(error);
            return false;
        }

        for (var current = gameObject.transform; current != null; current = current.parent)
        {
            targetName = SafeObjectName(current.gameObject) ?? "";
            hierarchyPath = BuildHierarchyPath(current);
            if (TryInvokeVisibleButtonObject(current.gameObject, actions, out methodUsed) ||
                TryNativeMouseClickGameObject(current.gameObject, actions, out methodUsed))
            {
                return true;
            }
        }

        error = $"Visible restart-like object was found but could not be clicked: {SafeObjectName(gameObject)}";
        return false;
    }

    private bool TryInvokeVisibleButtonObject(GameObject gameObject, List<string> actions, out string methodUsed)
    {
        methodUsed = "";
        foreach (var component in gameObject.GetComponents<Component>())
        {
            if (component == null)
            {
                continue;
            }

            var typeName = component.GetType().FullName ?? component.GetType().Name;
            if (TryInvokeNoArg(component, "OnMouseDown", out _))
            {
                actions.Add($"{typeName}.OnMouseDown({SafeObjectName(gameObject)})");
            }

            if (TryInvokeNoArg(component, "OnMouseUpAsButton", out var upAsButtonError))
            {
                methodUsed = $"{typeName}.OnMouseUpAsButton";
                actions.Add($"{methodUsed}({SafeObjectName(gameObject)})");
                return true;
            }

            if (TryInvokeNoArg(component, "OnMouseUp", out _))
            {
                methodUsed = $"{typeName}.OnMouseUp";
                actions.Add($"{methodUsed}({SafeObjectName(gameObject)})");
                return true;
            }

            if (TryInvokeUnityEvent(component, "clickEvent", out methodUsed) ||
                TryInvokeUnityEvent(component, "onClick", out methodUsed))
            {
                actions.Add($"{methodUsed}({SafeObjectName(gameObject)})");
                return true;
            }

            if (!string.IsNullOrWhiteSpace(upAsButtonError))
            {
                actions.Add($"{typeName}.OnMouseUpAsButton unavailable on {SafeObjectName(gameObject)}: {upAsButtonError}");
            }
        }

        return false;
    }

    private bool TryUIMgrEnterGame(List<string> actions)
    {
        try
        {
            var levelType = GameAPP.theBoardType;
            var levelNumber = GameAPP.theBoardLevel;
            var levelName = GameAPP.theIZLevelName ?? string.Empty;
            UIMgr.EnterGame(levelType, levelNumber, 0, levelName);
            actions.Add($"UIMgr.EnterGame({levelType},{levelNumber},0,{levelName})");
            return true;
        }
        catch (Exception ex)
        {
            actions.Add("UIMgr.EnterGame() failed: " + ex.Message);
            return false;
        }
    }

    private bool TryInitBoardQuickReset(
        List<string> actions,
        bool allowActiveGameplayMutation = false,
        string resetReason = "")
    {
        var guardObservation = BuildObservation(forceSeedProbe: true, forceRestartProbe: true);
        if (ShouldBlockActiveGameplayMutation("board reinit", guardObservation, actions, allowActiveGameplayMutation, resetReason))
        {
            return false;
        }

        var initBoard = FindInitBoard();
        if (initBoard == null)
        {
            actions.Add("InitBoard not found for quick reset.");
            return false;
        }

        var invoked = false;
        try
        {
            initBoard.QuickInGame();
            actions.Add("InitBoard.QuickInGame()");
            invoked = true;
        }
        catch (Exception ex)
        {
            actions.Add("InitBoard.QuickInGame() failed: " + ex.Message);
        }

        try
        {
            initBoard.StartInit();
            actions.Add("InitBoard.StartInit()");
            invoked = true;
        }
        catch (Exception ex)
        {
            actions.Add("InitBoard.StartInit() failed: " + ex.Message);
        }

        return invoked;
    }

    private object ActionCommand(JsonElement root)
    {
        var watch = Stopwatch.StartNew();
        var action = ReadInt(root, 0, "action", "action_id", "actionId");
        var returnObservation = ReadBool(root, true, "return_observation", "returnObservation");
        var before = BuildObservation();
        var decoded = DecodeAction(action, before);
        PlacementResult? placement = null;
        var illegalAction = decoded.Kind == "invalid";
        var illegalReason = decoded.Kind == "invalid" ? decoded.Error : null;

        var lossEvidence = HasLossRestartEvidence(before) &&
                           !HasPostWinEvidence(before) &&
                           before.TerminalHint != "possible_win";
        if (decoded.Kind == "plant" && lossEvidence)
        {
            illegalAction = true;
            illegalReason = "game_over_restart_screen";
            placement = PlacementResult.Fail(
                decoded.PlantType,
                decoded.Row,
                decoded.Column,
                "Game Over / Restart UI is active; plant actions are disabled until reset completes.",
                "game_over_restart_screen");
        }
        else if (decoded.Kind == "plant" && before.SeedSelectionActive)
        {
            illegalAction = true;
            illegalReason = "seed_selection_active";
            placement = PlacementResult.Fail(
                decoded.PlantType,
                decoded.Row,
                decoded.Column,
                "Seed selection UI is active; plant actions are disabled until Let's Rock has completed.",
                "seed_selection_active");
        }
        else if (decoded.Kind == "plant")
        {
            placement = TryPlaceSeedSlot(decoded.SeedSlotIndex, decoded.Row, decoded.Column, true, before);
            illegalAction = placement is { Success: false };
            illegalReason = placement?.IllegalReason;
        }

        var afterObservation = returnObservation ? BuildObservation() : null;
        watch.Stop();
        return new
        {
            action,
            decoded,
            legalBefore = before.LegalActions.Contains(action),
            legalActionReasonBefore = before.LegalActionReason,
            illegalAction,
            illegalReason,
            plantPlaced = placement?.PlantPlaced ?? false,
            costPaid = placement?.CostPaid ?? false,
            cooldownStarted = placement?.CooldownStarted ?? false,
            plantCost = placement?.PlantCost,
            costSource = placement?.CostSource,
            cardCooldown = placement?.CardCooldown,
            sunBefore = placement?.SunBefore,
            sunAfter = placement?.SunAfter,
            placement,
            observation = afterObservation,
            step_ms = _config.DebugPerformance ? Math.Round(watch.Elapsed.TotalMilliseconds, 3) : 0.0,
            observe_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.observe_ms : 0.0,
            bridge_observe_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.bridge_observe_ms : 0.0,
            screen_check_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.screen_check_ms : before.screen_check_ms,
            seed_probe_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.seed_probe_ms : 0.0,
            ui_scan_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.ui_scan_ms : 0.0,
            restartDetectionMode = afterObservation?.RestartDetectionMode ?? before.RestartDetectionMode
        };
    }

    private object FusionProbeCommand(JsonElement root)
    {
        var watch = Stopwatch.StartNew();
        var before = BuildObservation(forceSeedProbe: true);
        var candidates = new List<FusionCandidateDto>();
        var sourceRow = ReadNullableInt(root, "source_row", "sourceRow");
        var sourceCol = ReadNullableInt(root, "source_col", "sourceCol", "source_column", "sourceColumn");
        var sourcePlantType = ReadNullableInt(root, "source_plant_type", "sourcePlantType");
        var ingredientSeedSlotIndex = ReadNullableInt(root, "ingredient_seed_slot_index", "ingredientSeedSlotIndex", "seed_slot_index", "seedSlotIndex");
        foreach (var source in before.Plants)
        {
            if (sourceRow.HasValue && source.Row != sourceRow.Value)
            {
                continue;
            }
            if (sourceCol.HasValue && source.Column != sourceCol.Value)
            {
                continue;
            }
            if (sourcePlantType.HasValue && source.Type != sourcePlantType.Value)
            {
                continue;
            }
            foreach (var slot in before.SeedSlots.OrderBy(slot => slot.SlotIndex))
            {
                if (ingredientSeedSlotIndex.HasValue && slot.SlotIndex != ingredientSeedSlotIndex.Value)
                {
                    continue;
                }
                candidates.Add(ProbeFusionCandidate(before, source, slot));
            }
        }

        watch.Stop();
        return new
        {
            fusionAvailable = candidates.Any(candidate => candidate.FusionLegal),
            fusionCandidateCount = candidates.Count,
            fusionCandidates = candidates,
            gameplayReady = before.GameplayReady,
            seedSelectionActive = before.SeedSelectionActive,
            screenState = before.ScreenState,
            step_ms = _config.DebugPerformance ? Math.Round(watch.Elapsed.TotalMilliseconds, 3) : 0.0
        };
    }

    private object FusionStepCommand(JsonElement root)
    {
        var watch = Stopwatch.StartNew();
        var returnObservation = ReadBool(root, true, "return_observation", "returnObservation");
        var sourceInstanceId = ReadInt(root, 0, "source_instance_id", "sourceInstanceId");
        var sourceRow = ReadInt(root, -1, "source_row", "sourceRow");
        var sourceCol = ReadInt(root, -1, "source_col", "sourceCol", "source_column", "sourceColumn");
        var sourcePlantType = ReadInt(root, -1, "source_plant_type", "sourcePlantType");
        var ingredientSeedSlotIndex = ReadInt(root, -1, "ingredient_seed_slot_index", "ingredientSeedSlotIndex", "seed_slot_index", "seedSlotIndex");
        var ingredientPlantType = ReadInt(root, -1, "ingredient_plant_type", "ingredientPlantType", "target_plant_type", "targetPlantType");
        var predictedResultType = ReadInt(root, -1, "predicted_result_type", "predictedResultType");
        var predictedResultName = ReadString(root, "predicted_result_name", "predictedResultName") ?? "";
        var before = BuildObservation(forceSeedProbe: true);
        var rejection = ValidateFusionState(before);
        PlacementResult? placement = null;
        var candidate = new FusionCandidateDto
        {
            SourceInstanceId = sourceInstanceId,
            SourcePlantType = sourcePlantType,
            SourcePlantName = ((PlantType)sourcePlantType).ToString(),
            SourceRow = sourceRow,
            SourceCol = sourceCol,
            IngredientSeedSlotIndex = ingredientSeedSlotIndex,
            IngredientPlantType = ingredientPlantType,
            IngredientPlantName = ingredientPlantType >= 0 ? ((PlantType)ingredientPlantType).ToString() : "",
            PredictedResultType = predictedResultType,
            PredictedResultName = predictedResultName
        };

        if (string.IsNullOrWhiteSpace(rejection))
        {
            var source = FindSourcePlant(before, sourceInstanceId, sourceRow, sourceCol, sourcePlantType);
            if (source == null)
            {
                rejection = "source_not_found";
            }
            else if (!TryGetSeedSlotForPlacement(ingredientSeedSlotIndex, out var slot, out var card))
            {
                rejection = "target_not_available";
            }
            else if (ingredientPlantType >= 0 && slot.PlantType != ingredientPlantType)
            {
                rejection = "target_not_available";
            }
            else
            {
                candidate = ProbeFusionCandidate(before, source, slot);
                candidate.PredictedResultType = predictedResultType;
                candidate.PredictedResultName = predictedResultName;
                if (!candidate.FusionLegal)
                {
                    rejection = candidate.FusionBlockedReason ?? "bridge_rejected";
                }
                else
                {
                    placement = TryFuseSeedSlot(slot, card, source, before);
                    if (placement is { Success: false })
                    {
                        rejection = placement.IllegalReason ?? "bridge_rejected";
                    }
                }
            }
        }

        var afterObservation = returnObservation ? BuildObservation() : null;
        watch.Stop();
        var success = placement?.Success ?? false;
        return new
        {
            action = -1,
            decoded = new
            {
                kind = "fusion",
                sourcePlantType,
                ingredientPlantType,
                resultPlantType = predictedResultType,
                row = sourceRow,
                column = sourceCol
            },
            fusionAttempted = placement != null,
            fusionSucceeded = success,
            fusionRejectedReason = success ? null : rejection,
            illegalAction = !success,
            illegalReason = success ? null : rejection,
            plantPlaced = placement?.PlantPlaced ?? false,
            costPaid = placement?.CostPaid ?? false,
            cooldownStarted = placement?.CooldownStarted ?? false,
            plantCost = placement?.PlantCost,
            costSource = placement?.CostSource,
            cardCooldown = placement?.CardCooldown,
            sunBefore = placement?.SunBefore,
            sunAfter = placement?.SunAfter,
            sourceInstanceId,
            sourceRow,
            sourceCol,
            sourcePlantType,
            ingredientSeedSlotIndex,
            ingredientPlantType,
            predictedResultType,
            predictedResultName,
            fusionCandidate = candidate,
            placement,
            observation = afterObservation,
            step_ms = _config.DebugPerformance ? Math.Round(watch.Elapsed.TotalMilliseconds, 3) : 0.0,
            observe_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.observe_ms : 0.0,
            bridge_observe_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.bridge_observe_ms : 0.0,
            screen_check_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.screen_check_ms : before.screen_check_ms,
            seed_probe_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.seed_probe_ms : 0.0,
            ui_scan_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.ui_scan_ms : 0.0,
            restartDetectionMode = afterObservation?.RestartDetectionMode ?? before.RestartDetectionMode
        };
    }

    private string ValidateFusionState(ObservationDto obs)
    {
        if (obs.Done || obs.Over)
        {
            return "terminal_or_transition_state";
        }
        if (obs.SeedSelectionActive || obs.ScreenState == "seed_selection")
        {
            return "seed_selection_active";
        }
        if (obs.BlockingRewardUiActive || obs.ScreenState is "reward_unlock" or "reward_screen" or "level_complete_trophy")
        {
            return "reward_or_unlock_screen_active";
        }
        if (!obs.BoardFound)
        {
            return "not_gameplay";
        }
        if (!obs.GameplayReady)
        {
            return "gameplay_not_ready";
        }
        return "";
    }

    private PlantDto? FindSourcePlant(ObservationDto obs, int instanceId, int row, int column, int plantType)
    {
        foreach (var plant in obs.Plants)
        {
            if (plant.Row != row || plant.Column != column || plant.Type != plantType)
            {
                continue;
            }
            if (instanceId > 0 && plant.InstanceId > 0 && plant.InstanceId != instanceId)
            {
                continue;
            }
            return plant;
        }
        return null;
    }

    private FusionCandidateDto ProbeFusionCandidate(ObservationDto obs, PlantDto source, SeedSlotDto slot)
    {
        var candidate = new FusionCandidateDto
        {
            SourceInstanceId = source.InstanceId,
            SourcePlantType = source.Type,
            SourcePlantName = source.TypeName,
            SourceRow = source.Row,
            SourceCol = source.Column,
            IngredientSeedSlotIndex = slot.SlotIndex,
            IngredientPlantType = slot.PlantType,
            IngredientPlantName = slot.PlantTypeName,
            FusionLegal = false,
            FusionBlockedReason = ""
        };
        var stateRejection = ValidateFusionState(obs);
        if (!string.IsNullOrWhiteSpace(stateRejection))
        {
            candidate.FusionBlockedReason = stateRejection;
            return candidate;
        }
        if (!slot.Usable || slot.Disabled || !slot.Ready || obs.Sun < slot.SeedCost)
        {
            candidate.FusionBlockedReason = "target_not_available";
            return candidate;
        }
        var createPlant = FindCreatePlant();
        if (createPlant == null)
        {
            candidate.FusionBlockedReason = "fusion_not_available";
            return candidate;
        }
        try
        {
            var plantType = (PlantType)slot.PlantType;
            if (!createPlant.CheckBox(source.Column, source.Row, plantType))
            {
                candidate.FusionBlockedReason = "bridge_rejected";
                return candidate;
            }
        }
        catch
        {
            candidate.FusionBlockedReason = "bridge_rejected";
            return candidate;
        }
        candidate.FusionLegal = true;
        candidate.FusionBlockedReason = "";
        return candidate;
    }

    private PlacementResult TryFuseSeedSlot(SeedSlotDto slot, CardUI? card, PlantDto source, ObservationDto precheckedObservation)
    {
        var plantTypeId = slot.PlantType;
        var board = FindBoard();
        if (board == null)
        {
            return PlacementResult.Fail(plantTypeId, source.Row, source.Column, "Board not found.", "board_not_found")
                .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId);
        }
        var createPlant = FindCreatePlant();
        if (createPlant == null)
        {
            return PlacementResult.Fail(plantTypeId, source.Row, source.Column, "CreatePlant.Instance not found.", "create_plant_not_found")
                .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId);
        }
        var stateRejection = ValidateFusionState(precheckedObservation);
        if (!string.IsNullOrWhiteSpace(stateRejection))
        {
            return PlacementResult.Fail(plantTypeId, source.Row, source.Column, "Fusion blocked by unsafe state.", stateRejection)
                .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId);
        }
        var costInfo = new PlantCostInfo
        {
            PlantType = plantTypeId,
            Cost = Math.Max(0, slot.SeedCost),
            Source = slot.Source ?? "seed_slot",
            Warning = slot.Warning
        };
        var sunBefore = board.theSun;
        var cooldownInfo = card != null
            ? BuildCardCooldownDto(card, plantTypeId, $"seed_slot[{slot.SlotIndex}]")
            : SlotCooldownFromDto(slot);
        if (!slot.Usable || slot.Disabled)
        {
            return PlacementResult.Fail(plantTypeId, source.Row, source.Column, "Seed slot is not usable.", slot.Disabled ? "slot_disabled" : "slot_not_usable", costInfo, sunBefore, sunBefore, cooldownInfo)
                .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId);
        }
        if (sunBefore < costInfo.Cost)
        {
            return PlacementResult.Fail(plantTypeId, source.Row, source.Column, "Insufficient sun for fusion.", "target_not_available", costInfo, sunBefore, sunBefore, cooldownInfo)
                .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId);
        }
        if (!cooldownInfo.Ready)
        {
            return PlacementResult.Fail(plantTypeId, source.Row, source.Column, "Seed slot is on cooldown.", "target_not_available", costInfo, sunBefore, sunBefore, cooldownInfo)
                .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId);
        }
        try
        {
            if (!createPlant.CheckBox(source.Column, source.Row, (PlantType)plantTypeId))
            {
                return PlacementResult.Fail(plantTypeId, source.Row, source.Column, "CreatePlant.CheckBox returned false for fusion.", "bridge_rejected", costInfo, sunBefore, sunBefore, cooldownInfo)
                    .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId);
            }
        }
        catch (Exception ex)
        {
            return PlacementResult.Fail(plantTypeId, source.Row, source.Column, "CreatePlant.CheckBox failed: " + ex.Message, "bridge_rejected", costInfo, sunBefore, sunBefore, cooldownInfo)
                .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId);
        }
        try
        {
            var obj = createPlant.SetPlant(source.Column, source.Row, (PlantType)plantTypeId, null, Vector2.zero, true, true, null);
            var plantPlaced = obj != null;
            var costPaid = false;
            var paymentSource = "not_paid";
            var sunAfter = board.theSun;
            if (plantPlaced)
            {
                costPaid = TryPaySun(board, costInfo.Cost, sunBefore, out sunAfter, out paymentSource);
            }
            var cooldownStarted = false;
            var cooldownSource = "not_started";
            if (plantPlaced && costPaid && card != null)
            {
                cooldownStarted = TryStartSeedSlotCooldown(card, slot.SlotIndex, out cooldownSource);
                InvalidateSeedRuntimeCache($"fusion_seed_slot[{slot.SlotIndex}] placed");
            }
            return new PlacementResult
            {
                Success = plantPlaced && costPaid,
                PlantPlaced = plantPlaced,
                CostPaid = costPaid,
                CooldownStarted = cooldownStarted,
                PlantCost = costInfo.Cost,
                CostSource = costInfo.Source,
                CostWarning = costInfo.Warning,
                PaymentSource = paymentSource,
                CooldownSource = cooldownSource,
                CardCooldown = card != null ? BuildCardCooldownDto(card, plantTypeId, $"seed_slot[{slot.SlotIndex}]") : SlotCooldownFromDto(slot),
                SunBefore = sunBefore,
                SunAfter = sunAfter,
                PlantType = plantTypeId,
                PlantTypeName = ((PlantType)plantTypeId).ToString(),
                SeedSlotIndex = slot.SlotIndex,
                CardInstanceId = slot.CardInstanceId,
                Row = source.Row,
                Column = source.Column,
                Message = plantPlaced ? $"Fusion attempted from seed slot {slot.SlotIndex} and cost paid." : "SetPlant returned null.",
                IllegalReason = plantPlaced && costPaid ? null : plantPlaced ? "cost_payment_failed" : "setplant_returned_null"
            };
        }
        catch (Exception ex)
        {
            return PlacementResult.Fail(plantTypeId, source.Row, source.Column, "CreatePlant.SetPlant fusion failed: " + ex.Message, "setplant_failed", costInfo, sunBefore, board.theSun)
                .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId);
        }
    }

    private PlacementResult TryPlacePlant(int plantTypeId, int row, int column, bool checkLegal, ObservationDto? precheckedObservation = null)
    {
        var board = FindBoard();
        if (board == null)
        {
            return PlacementResult.Fail(plantTypeId, row, column, "Board not found.", "board_not_found");
        }

        var createPlant = FindCreatePlant();
        if (createPlant == null)
        {
            return PlacementResult.Fail(plantTypeId, row, column, "CreatePlant.Instance not found.", "create_plant_not_found");
        }

        var gateObservation = precheckedObservation ?? BuildObservation();
        if (gateObservation.SeedSelectionActive)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                "Seed selection UI is active; plant actions are disabled until Let's Rock has completed.",
                "seed_selection_active");
        }

        var plantType = (PlantType)plantTypeId;
        var costInfo = GetPlantCost(plantTypeId);
        var sunBefore = board.theSun;

        if (sunBefore < costInfo.Cost)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                $"Insufficient sun: need {costInfo.Cost}, have {sunBefore}.",
                "insufficient_sun",
                costInfo,
                sunBefore,
                sunBefore);
        }

        var cooldownInfo = GetCardCooldown(plantTypeId);
        if (!cooldownInfo.Ready)
        {
            var reason = cooldownInfo.Found ? "cooldown" : "card_not_found";
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                cooldownInfo.Found
                    ? $"Card is on cooldown: {cooldownInfo.CurrentCooldown:0.###}/{cooldownInfo.FullCooldown:0.###}."
                    : "No matching CardUI seed packet was found.",
                reason,
                costInfo,
                sunBefore,
                sunBefore,
                cooldownInfo);
        }

        if (checkLegal)
        {
            var occupancy = gateObservation;
            if (IsCellOccupied(occupancy, row, column))
            {
                return PlacementResult.Fail(
                    plantTypeId,
                    row,
                    column,
                    "Cell is occupied by a logical or visible plant object.",
                    "occupied_cell",
                    costInfo,
                    sunBefore,
                    sunBefore,
                    cooldownInfo);
            }

            try
            {
                if (!createPlant.CheckBox(column, row, plantType))
                {
                    return PlacementResult.Fail(
                        plantTypeId,
                        row,
                        column,
                        "CreatePlant.CheckBox returned false.",
                        "invalid_cell_or_terrain",
                        costInfo,
                        sunBefore,
                        sunBefore);
                }
            }
            catch (Exception ex)
            {
                return PlacementResult.Fail(
                    plantTypeId,
                    row,
                    column,
                    "CreatePlant.CheckBox failed: " + ex.Message,
                    "checkbox_failed",
                    costInfo,
                    sunBefore,
                    sunBefore);
            }
        }

        try
        {
            var obj = createPlant.SetPlant(column, row, plantType, null, Vector2.zero, true, true, null);
            var plantPlaced = obj != null;
            var costPaid = false;
            var paymentSource = "not_paid";
            var sunAfter = board.theSun;

            if (plantPlaced)
            {
                costPaid = TryPaySun(board, costInfo.Cost, sunBefore, out sunAfter, out paymentSource);
            }

            var cooldownStarted = false;
            var cooldownSource = "not_started";
            if (plantPlaced && costPaid)
            {
                cooldownStarted = TryStartCardCooldown(plantTypeId, out cooldownSource);
                InvalidateSeedRuntimeCache($"plant_type[{plantTypeId}] placed");
            }

            return new PlacementResult
            {
                Success = plantPlaced && costPaid,
                PlantPlaced = plantPlaced,
                CostPaid = costPaid,
                CooldownStarted = cooldownStarted,
                PlantCost = costInfo.Cost,
                CostSource = costInfo.Source,
                PaymentSource = paymentSource,
                CooldownSource = cooldownSource,
                CardCooldown = GetCardCooldown(plantTypeId),
                SunBefore = sunBefore,
                SunAfter = sunAfter,
                PlantType = plantTypeId,
                PlantTypeName = plantType.ToString(),
                Row = row,
                Column = column,
                Message = plantPlaced ? "Plant placed and cost paid." : "SetPlant returned null.",
                IllegalReason = plantPlaced && costPaid ? null : plantPlaced ? "cost_payment_failed" : "setplant_returned_null"
            };
        }
        catch (Exception ex)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                "CreatePlant.SetPlant failed: " + ex.Message,
                "setplant_failed",
                costInfo,
                sunBefore,
                board.theSun);
        }
    }

    private PlacementResult TryPlaceSeedSlot(int seedSlotIndex, int row, int column, bool checkLegal, ObservationDto? precheckedObservation = null)
    {
        if (!TryGetSeedSlotForPlacement(seedSlotIndex, out var slot, out var card))
        {
            return PlacementResult.Fail(
                -1,
                row,
                column,
                $"Seed slot {seedSlotIndex} is not available in the active gameplay card bank.",
                "seed_slot_not_found")
                .WithSeedSlot(seedSlotIndex);
        }

        var plantTypeId = slot.PlantType;
        var board = FindBoard();
        if (board == null)
        {
            return PlacementResult.Fail(plantTypeId, row, column, "Board not found.", "board_not_found")
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }

        var createPlant = FindCreatePlant();
        if (createPlant == null)
        {
            return PlacementResult.Fail(plantTypeId, row, column, "CreatePlant.Instance not found.", "create_plant_not_found")
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }

        var gateObservation = precheckedObservation ?? BuildObservation();
        if (gateObservation.SeedSelectionActive)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                "Seed selection UI is active; plant actions are disabled until Let's Rock has completed.",
                "seed_selection_active")
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }

        var plantType = (PlantType)plantTypeId;
        var costInfo = new PlantCostInfo
        {
            PlantType = plantTypeId,
            Cost = Math.Max(0, slot.SeedCost),
            Source = slot.Source ?? "seed_slot",
            Warning = slot.Warning
        };
        var sunBefore = board.theSun;

        if (!slot.Usable || slot.Disabled)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                $"Seed slot {seedSlotIndex} is not usable.",
                slot.Disabled ? "slot_disabled" : "slot_not_usable",
                costInfo,
                sunBefore,
                sunBefore,
                SlotCooldownFromDto(slot))
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }

        if (sunBefore < costInfo.Cost)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                $"Insufficient sun for seed slot {seedSlotIndex}: need {costInfo.Cost}, have {sunBefore}.",
                "insufficient_sun",
                costInfo,
                sunBefore,
                sunBefore,
                SlotCooldownFromDto(slot))
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }

        var cooldownInfo = card != null
            ? BuildCardCooldownDto(card, plantTypeId, $"seed_slot[{seedSlotIndex}]")
            : SlotCooldownFromDto(slot);
        if (!cooldownInfo.Ready)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                $"Seed slot {seedSlotIndex} is on cooldown: {cooldownInfo.CurrentCooldown:0.###}/{cooldownInfo.FullCooldown:0.###}.",
                "cooldown",
                costInfo,
                sunBefore,
                sunBefore,
                cooldownInfo)
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }

        if (checkLegal)
        {
            if (IsCellOccupied(gateObservation, row, column))
            {
                return PlacementResult.Fail(
                    plantTypeId,
                    row,
                    column,
                    "Cell is occupied by a logical or visible plant object.",
                    "occupied_cell",
                    costInfo,
                    sunBefore,
                    sunBefore,
                    cooldownInfo)
                    .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
            }

            try
            {
                if (!createPlant.CheckBox(column, row, plantType))
                {
                    return PlacementResult.Fail(
                        plantTypeId,
                        row,
                        column,
                        "CreatePlant.CheckBox returned false.",
                        "invalid_cell_or_terrain",
                        costInfo,
                        sunBefore,
                        sunBefore,
                        cooldownInfo)
                        .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
                }
            }
            catch (Exception ex)
            {
                return PlacementResult.Fail(
                    plantTypeId,
                    row,
                    column,
                    "CreatePlant.CheckBox failed: " + ex.Message,
                    "checkbox_failed",
                    costInfo,
                    sunBefore,
                    sunBefore,
                    cooldownInfo)
                    .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
            }
        }

        try
        {
            var obj = createPlant.SetPlant(column, row, plantType, null, Vector2.zero, true, true, null);
            var plantPlaced = obj != null;
            var costPaid = false;
            var paymentSource = "not_paid";
            var sunAfter = board.theSun;

            if (plantPlaced)
            {
                costPaid = TryPaySun(board, costInfo.Cost, sunBefore, out sunAfter, out paymentSource);
            }

            var cooldownStarted = false;
            var cooldownSource = "not_started";
            if (plantPlaced && costPaid && card != null)
            {
                cooldownStarted = TryStartSeedSlotCooldown(card, seedSlotIndex, out cooldownSource);
                InvalidateSeedRuntimeCache($"seed_slot[{seedSlotIndex}] placed");
            }

            return new PlacementResult
            {
                Success = plantPlaced && costPaid,
                PlantPlaced = plantPlaced,
                CostPaid = costPaid,
                CooldownStarted = cooldownStarted,
                PlantCost = costInfo.Cost,
                CostSource = costInfo.Source,
                CostWarning = costInfo.Warning,
                PaymentSource = paymentSource,
                CooldownSource = cooldownSource,
                CardCooldown = card != null ? BuildCardCooldownDto(card, plantTypeId, $"seed_slot[{seedSlotIndex}]") : SlotCooldownFromDto(slot),
                SunBefore = sunBefore,
                SunAfter = sunAfter,
                PlantType = plantTypeId,
                PlantTypeName = plantType.ToString(),
                SeedSlotIndex = seedSlotIndex,
                CardInstanceId = slot.CardInstanceId,
                Row = row,
                Column = column,
                Message = plantPlaced ? $"Plant placed from seed slot {seedSlotIndex} and cost paid." : "SetPlant returned null.",
                IllegalReason = plantPlaced && costPaid ? null : plantPlaced ? "cost_payment_failed" : "setplant_returned_null"
            };
        }
        catch (Exception ex)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                "CreatePlant.SetPlant failed: " + ex.Message,
                "setplant_failed",
                costInfo,
                sunBefore,
                board.theSun)
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }
    }

    private ObservationDto BuildObservation(bool includeDebugArrays = false, bool forceSeedProbe = false, bool forceRestartProbe = false)
    {
        var observeWatch = Stopwatch.StartNew();
        var seedProbeMs = 0.0;
        var uiScanMs = 0.0;
        var board = FindBoard();
        var restartInfo = DetectRestartScreenInfo(forceRestartProbe || board == null);
        var obs = new ObservationDto
        {
            BoardFound = board != null,
            CreatePlantFound = FindCreatePlant() != null,
            GameSpeed = TryReadGameSpeed(),
            RequestedGameSpeed = _config.GameSpeed,
            GameSpeedMode = _config.GameSpeedMode,
            UnityTimeScale = SafeReadTimeScale(),
            FixedDeltaTime = SafeReadFixedDeltaTime(),
            EffectiveGameSpeed = ResolveEffectiveGameSpeed()
        };
        ApplyRestartScreenInfo(obs, restartInfo);

        if (board == null)
        {
            var lossDetected = HasLossRestartEvidence(obs);
            obs.Done = lossDetected;
            obs.Over = obs.Done;
            obs.TerminalHint = obs.Done ? "game_over_or_loss" : "board_not_found";
            obs.LegalActionReason = obs.Done ? "game_over_restart_screen" : "board_not_found";
            obs.CanReadBoard = false;
            obs.OnSeedSelectionScreen = false;
            obs.NextStep = obs.OnLossScreen ? "click_restart" : "wait_for_board";
            obs.DebugMessage = obs.OnLossScreen ? "Loss screen is visible but Board was not found." : "Board was not found.";
            obs.LegalActions.Add(0);
            obs.LegalActionCount = obs.LegalActions.Count;
            obs.ActionCount = GetActionCount(_config.FallbackRows, _config.FallbackColumns);
            ApplyAdventureObservationFields(obs);
            AddPerformanceTimings(obs, observeWatch, seedProbeMs, uiScanMs);
            return obs;
        }

        try
        {
            obs.Sun = board.theSun;
            obs.Wave = board.theWave;
            obs.MaxWave = board.theMaxWave;
            obs.RowCount = SafePositive(board.rowNum, _config.FallbackRows);
            obs.ColumnCount = SafePositive(board.columnNum, _config.FallbackColumns);
            obs.KillCount = board.killZombieCount;
            obs.Over = board.over;
            obs.BoardCardSelectable = board.cardSelectable;
            obs.BoardStartMove = board.startMove;
            obs.Time = board.time;
            obs.FrameCount = Time.frameCount;
            obs.RealtimeSinceStartup = Time.realtimeSinceStartup;
            obs.MoreZombiesComing = board.moreZombiesComing;
            obs.CanReadBoard = true;
            obs.PlantCount = board.plantArray?.Count ?? 0;
            obs.ZombieCount = board.zombieArray?.Count ?? 0;
            obs.BulletCount = board.bulletArray?.Count ?? 0;
            obs.MinBoardX = board.boardMinX;
            obs.MaxBoardX = board.boardMaxX;
            obs.ZombieMinX = board.zombieMinX;
            obs.ZombieMaxX = board.zombieMaxX;
            AddSpeedDiagnostics(obs, board);

            AddPlantCosts(obs);
            AddCardCooldowns(obs);
            AddPlants(board, obs);
            if (includeDebugArrays)
            {
                AddVisiblePlants(board, obs);
            }
            AddVisibleMowers(board, obs);
            AddZombies(board, obs);
            obs.PlantCount = obs.Plants.Count;
            obs.ZombieCount = obs.Zombies.Count;
            AddLaneSummaries(obs);

            var possibleWin = obs.MaxWave > 0 && obs.Wave >= obs.MaxWave && obs.ZombieCount == 0 && !obs.MoreZombiesComing;
            obs.Done = obs.Over;
            obs.TerminalHint = possibleWin ? "possible_win" : obs.Over ? "game_over_or_loss" : "running";
            if (!HasLossRestartEvidence(obs) && (forceRestartProbe || obs.Over))
            {
                restartInfo = DetectRestartScreenInfo(broadScan: true);
                ApplyRestartScreenInfo(obs, restartInfo);
            }

            var lossEvidence = HasLossRestartEvidence(obs) && !HasPostWinEvidence(obs) && obs.TerminalHint != "possible_win";
            obs.OnLossScreen = lossEvidence;
            if (lossEvidence)
            {
                ApplyRestartTerminalOverride(obs);
                AddPerformanceTimings(obs, observeWatch, seedProbeMs, uiScanMs);
                return obs;
            }

            obs.TotalPlantHealth = obs.Plants.Sum(p => Math.Max(0, p.Health));
            var rawGameplayReady = obs.BoardFound &&
                                   obs.CreatePlantFound &&
                                   obs.BoardStartMove &&
                                   !obs.Done &&
                                   obs.CardCooldowns.All(c => c.Found);
            var seedState = ResolveSeedStateForObservation(board, rawGameplayReady, forceSeedProbe, out seedProbeMs, out uiScanMs);
            var activeGameplayCounts = new Dictionary<int, int>(seedState.ActiveGameplayTypeCounts);
            var requiredGameplayCounts = BuildTypeCounts(_config.PlantTypes);
            obs.SeedSelectionActive = seedState.SeedSelectionActive;
            obs.SeedSelectionPanelActive = seedState.SeedSelectionPanelActive;
            obs.StartButtonActive = seedState.StartButtonActive;
            obs.BlockingRewardUiActive = seedState.BlockingRewardUiActive;
            var liveBoardRunning = IsLiveBoardRunning(obs);
            if (liveBoardRunning && obs.BlockingRewardUiActive)
            {
                obs.BlockingRewardUiActive = false;
                LoggerInstance.Msg(
                    $"[safety] suppress reward UI during gameplay wave={obs.Wave}/{obs.MaxWave} zombies={obs.ZombieCount} plants={obs.PlantCount} terminal={obs.TerminalHint}"
                );
            }
            obs.OnSeedSelectionScreen = obs.SeedSelectionActive;
            obs.ActiveGameplayCardBankCount = seedState.ActiveGameplayCardBankCount;
            obs.SeedSlots.AddRange(seedState.SeedSlots);
            obs.SeedSlotCount = obs.SeedSlots.Count;
            obs.SlotPlantTypes = obs.SeedSlots.Select(slot => slot.PlantType).ToArray();
            RefreshCompatibilityFieldsFromSeedSlots(obs);
            obs.ActualGameplayReady = rawGameplayReady &&
                                      !obs.SeedSelectionActive &&
                                      !obs.BlockingRewardUiActive &&
                                      CountsCover(activeGameplayCounts, requiredGameplayCounts);
            obs.GameplayReady = obs.ActualGameplayReady;
            if (obs.SeedSelectionActive)
            {
                obs.LegalActionReason = "seed_selection_active";
            }
            else if (obs.BlockingRewardUiActive)
            {
                obs.LegalActionReason = "blocking_reward_ui_active";
            }
            else if (!obs.ActualGameplayReady)
            {
                obs.LegalActionReason = "gameplay_not_ready";
            }

            obs.NextStep = obs.OnLossScreen
                ? "click_restart"
                : obs.OnSeedSelectionScreen
                    ? "auto_select_seeds_then_lets_rock"
                    : obs.GameplayReady
                        ? "play"
                        : obs.BlockingRewardUiActive
                            ? "cleanup_reward_ui"
                            : "wait_for_gameplay_ready";
            ApplyAdventureObservationFields(obs);
            obs.DebugMessage = $"terminalHint={obs.TerminalHint}, seedSelectionActive={obs.SeedSelectionActive}, gameplayReady={obs.GameplayReady}, seedSlots={obs.SeedSlotCount}";

            AddLegalActions(obs);
            obs.LegalActionCount = obs.LegalActions.Count;
            obs.ActionCount = GetActionCount(obs.RowCount, obs.ColumnCount, obs.SeedSlots.Count);
        }
        catch (Exception ex)
        {
            obs.ReadError = ex.Message;
            obs.TerminalHint = "observation_error";
        }

        AddPerformanceTimings(obs, observeWatch, seedProbeMs, uiScanMs);
        return obs;
    }

    private static void ApplyRestartScreenInfo(ObservationDto obs, RestartScreenInfo info)
    {
        obs.OnGameOverScreen = info.OnGameOverScreen;
        obs.LossMenuActive = info.LossMenuActive;
        obs.GameOverTextVisible = info.GameOverTextVisible;
        obs.OnRestartScreen = info.OnRestartScreen;
        obs.RestartButtonActive = info.RestartButtonActive;
        obs.RestartButtonName = info.RestartButtonName;
        obs.RestartButtonPath = info.RestartButtonPath;
        obs.RestartDetectionReason = info.RestartDetectionReason;
        obs.RestartDetectionMode = info.RestartDetectionMode;
        obs.OnPauseMenu = info.OnPauseMenu;
        obs.PauseMenuActive = info.PauseMenuActive;
        obs.PauseRestartButtonActive = info.PauseRestartButtonActive;
        obs.screen_check_ms = info.screen_check_ms;
        obs.OnLossScreen = obs.OnLossScreen || HasLossRestartEvidence(info);
    }

    private void ApplyRestartTerminalOverride(ObservationDto obs)
    {
        obs.Done = true;
        obs.Over = true;
        obs.GameplayReady = false;
        obs.ActualGameplayReady = false;
        obs.SeedSelectionActive = false;
        obs.SeedSelectionPanelActive = false;
        obs.StartButtonActive = false;
        obs.TerminalHint = "game_over_or_loss";
        obs.LegalActionReason = "game_over_restart_screen";
        obs.NextStep = "click_restart";
        obs.DebugMessage = $"Game Over / Restart overlay detected: {obs.RestartDetectionReason}";
        obs.LegalActions.Clear();
        obs.LegalActions.Add(0);
        obs.LegalActionCount = 1;
        obs.ActionCount = GetActionCount(
            SafePositive(obs.RowCount, _config.FallbackRows),
            SafePositive(obs.ColumnCount, _config.FallbackColumns),
            Math.Max(obs.SeedSlotCount, _config.PlantTypes.Count));
        ApplyAdventureObservationFields(obs);
    }

    private static void ApplyAdventureObservationFields(ObservationDto obs)
    {
        if (IsLiveBoardRunning(obs) && obs.BlockingRewardUiActive)
        {
            obs.BlockingRewardUiActive = false;
        }
        obs.ScreenState = obs.BlockingRewardUiActive
            ? "reward_unlock"
            : obs.OnGameOverScreen || obs.OnLossScreen || obs.RestartButtonActive
                ? "game_over"
                : obs.SeedSelectionActive || obs.OnSeedSelectionScreen
                    ? "seed_selection"
                    : obs.GameplayReady
                        ? "gameplay"
                        : obs.BoardFound
                            ? "transition"
                            : "loading_or_menu";
        obs.CurrentMode = SafeReadGameBoardType();
        obs.IsMainMenu = obs.ScreenState == "loading_or_menu" && !obs.BoardFound;
        obs.IsAdventureButtonVisible = false;
        obs.StartupPopupVisible = false;
        obs.StartupOkButtonVisible = false;
        obs.MainMenuBlockedByPopup = false;
        obs.IsSeedSelectionScreen = obs.ScreenState == "seed_selection";
        obs.IsGameplayReady = obs.ScreenState == "gameplay";
        obs.IsLevelComplete = obs.ScreenState == "reward_unlock" || obs.ScreenState == "level_complete_trophy" || obs.ScreenState == "reward_screen";
        obs.IsRewardScreen = obs.ScreenState == "reward_unlock" || obs.ScreenState == "reward_screen" || obs.ScreenState == "level_complete_trophy";
        obs.IsNewPlantUnlockedScreen = false;
        obs.IsAlmanacOrSeedPacketScreen = false;
        obs.IsGameOverScreen = obs.ScreenState == "game_over";
        obs.CurrentAdventureLevel = SafeReadGameBoardLevel();
        obs.CurrentWorldOrStage = obs.CurrentAdventureLevel > 0 ? ((obs.CurrentAdventureLevel - 1) / 10) + 1 : 0;
        obs.CurrentDayLevel = obs.CurrentAdventureLevel > 0 ? ((obs.CurrentAdventureLevel - 1) % 10) + 1 : 0;
        obs.UnlockedSeedNames = SeedNames(obs.SeedSlots.Select(slot => slot.PlantType));
        obs.AvailableSeedNames = Array.Empty<string>();
        obs.SelectedSeedNames = SeedNames(obs.SeedSlots.Select(slot => slot.PlantType));
        obs.UnknownVisibleSeedNames = Array.Empty<string>();
    }

    private void AddPlantCosts(ObservationDto obs)
    {
        foreach (var plantTypeId in _config.PlantTypes)
        {
            var cost = GetPlantCost(plantTypeId);
            obs.PlantCosts.Add(new PlantCostDto
            {
                PlantType = plantTypeId,
                PlantTypeName = ((PlantType)plantTypeId).ToString(),
                Cost = cost.Cost,
                Source = cost.Source
            });
        }
    }

    private void AddCardCooldowns(ObservationDto obs)
    {
        foreach (var plantTypeId in _config.PlantTypes)
        {
            obs.CardCooldowns.Add(GetCardCooldown(plantTypeId));
        }
    }

    private void RefreshCompatibilityFieldsFromSeedSlots(ObservationDto obs)
    {
        if (obs.SeedSlots.Count == 0)
        {
            return;
        }

        obs.PlantCosts.Clear();
        obs.CardCooldowns.Clear();
        foreach (var slot in obs.SeedSlots.OrderBy(s => s.SlotIndex))
        {
            obs.PlantCosts.Add(new PlantCostDto
            {
                PlantType = slot.PlantType,
                PlantTypeName = slot.PlantTypeName,
                Cost = slot.SeedCost,
                Source = $"seed_slot[{slot.SlotIndex}]/{slot.Source}"
            });
            var cooldown = SlotCooldownFromDto(slot);
            cooldown.Source = $"seed_slot[{slot.SlotIndex}]/{slot.Source}";
            obs.CardCooldowns.Add(cooldown);
        }
    }

    private void AddPlants(Board board, ObservationDto obs)
    {
        var plants = board.plantArray;
        if (plants == null)
        {
            return;
        }

        for (var i = 0; i < plants.Count; i++)
        {
            try
            {
                var plant = plants[i];
                if (plant == null)
                {
                    continue;
                }

                var position = plant.transform.position;
                obs.Plants.Add(new PlantDto
                {
                    Index = i,
                    InstanceId = plant.GetInstanceID(),
                    Type = (int)plant.thePlantType,
                    TypeName = plant.thePlantType.ToString(),
                    Row = plant.thePlantRow,
                    Column = plant.thePlantColumn,
                    Health = plant.thePlantHealth,
                    MaxHealth = plant.thePlantMaxHealth,
                    Level = plant.theLevel,
                    AttackCooldown = plant.thePlantAttackCountDown,
                    ProduceCooldown = plant.thePlantProduceCountDown,
                    X = position.x,
                    Y = position.y
                });
            }
            catch
            {
                // Ignore destroyed IL2CPP wrappers during enumeration.
            }
        }
    }

    private void AddVisiblePlants(Board board, ObservationDto obs)
    {
        var primaryPlantIds = new HashSet<int>();
        AddComponentIds(primaryPlantIds, board.plantArray);
        foreach (var visiblePlant in ScanVisiblePlants(board, primaryPlantIds))
        {
            obs.VisiblePlants.Add(visiblePlant);
        }

        obs.VisiblePlantObjectCount = obs.VisiblePlants.Count(p => p.ActiveInHierarchy && p.InBoardBounds);
        obs.StaleVisiblePlantObjectCount = obs.VisiblePlants.Count(p => p.ActiveInHierarchy && p.InBoardBounds && !p.InPlantArray);
    }

    private void AddVisibleMowers(Board board, ObservationDto obs)
    {
        var mowerIds = GetMowerArrayIds(board, out var logicalMowerCount);
        obs.LogicalMowerCount = logicalMowerCount;
        foreach (var visibleMower in ScanVisibleMowers(board, mowerIds))
        {
            obs.VisibleMowers.Add(visibleMower);
        }

        obs.VisibleMowerObjectCount = obs.VisibleMowers.Count(m => m.ActiveInHierarchy && m.InBoardBounds);
        obs.StaleVisibleMowerObjectCount = obs.VisibleMowers.Count(m => m.ActiveInHierarchy && m.InBoardBounds && !m.InMowerArray);
        foreach (var group in obs.VisibleMowers.Where(m => m.ActiveInHierarchy && m.InBoardBounds).GroupBy(m => m.Row))
        {
            if (group.Count() > 1)
            {
                obs.DuplicateMowerRows.Add(group.Key);
            }
        }
        obs.DuplicateMowerRowCount = obs.DuplicateMowerRows.Count;
    }

    private void AddZombies(Board board, ObservationDto obs)
    {
        var zombies = board.zombieArray;
        if (zombies == null)
        {
            return;
        }

        for (var i = 0; i < zombies.Count; i++)
        {
            try
            {
                var zombie = zombies[i];
                if (zombie == null)
                {
                    continue;
                }

                var position = zombie.transform.position;
                obs.Zombies.Add(new ZombieDto
                {
                    Index = i,
                    Type = (int)zombie.theZombieType,
                    TypeName = zombie.theZombieType.ToString(),
                    Row = zombie.theZombieRow,
                    Column = SafeReadColumn(zombie),
                    Health = zombie.theHealth,
                    MaxHealth = zombie.theMaxHealth,
                    Status = (int)zombie.theStatus,
                    StatusName = zombie.theStatus.ToString(),
                    Speed = zombie.theSpeed,
                    Alive = SafeReadAlive(zombie),
                    X = position.x,
                    Y = position.y
                });
            }
            catch
            {
                // Ignore destroyed IL2CPP wrappers during enumeration.
            }
        }
    }

    private void AddLaneSummaries(ObservationDto obs)
    {
        for (var row = 0; row < obs.RowCount; row++)
        {
            var laneZombies = obs.Zombies.Where(z => z.Row == row && z.Alive).ToList();
            if (laneZombies.Count == 0)
            {
                obs.Lanes.Add(new LaneDto { Row = row, ZombieCount = 0 });
                continue;
            }

            var nearest = laneZombies.OrderBy(z => z.X).First();
            var coneheads = laneZombies.Count(IsConeheadZombie);
            var bucketheads = laneZombies.Count(IsBucketheadZombie);
            var toughZombies = laneZombies.Where(IsToughZombie).ToList();
            var nearestTough = toughZombies.OrderBy(z => z.X).FirstOrDefault();
            obs.Lanes.Add(new LaneDto
            {
                Row = row,
                ZombieCount = laneZombies.Count,
                NearestZombieX = nearest.X,
                NearestZombieHealth = nearest.Health,
                NearestZombieType = nearest.Type,
                ConeheadCount = coneheads,
                BucketheadCount = bucketheads,
                ToughZombieCount = toughZombies.Count,
                ToughZombieNearestX = nearestTough?.X,
                ToughZombiePressureScore = toughZombies.Sum(z => Math.Max(0f, 1f - z.X / 10f))
            });
        }
    }

    private static bool IsConeheadZombie(ZombieDto zombie)
    {
        var name = (zombie.TypeName ?? "").ToLowerInvariant();
        return zombie.Type is 2 or 12 || name.Contains("cone") || name.Contains("roadblock") || name.Contains("路障");
    }

    private static bool IsBucketheadZombie(ZombieDto zombie)
    {
        var name = (zombie.TypeName ?? "").ToLowerInvariant();
        return zombie.Type is 4 or 13 || name.Contains("bucket") || name.Contains("铁桶");
    }

    private static bool IsToughZombie(ZombieDto zombie) =>
        IsConeheadZombie(zombie) || IsBucketheadZombie(zombie) || zombie.Health >= 600 || zombie.MaxHealth >= 600;

    private void AddLegalActions(ObservationDto obs)
    {
        obs.LegalActions.Add(0);
        if (!obs.GameplayReady)
        {
            if (string.IsNullOrWhiteSpace(obs.LegalActionReason))
            {
                obs.LegalActionReason = "gameplay_not_ready";
            }

            return;
        }

        if (obs.SeedSlots.Count == 0)
        {
            obs.LegalActionReason = "seed_slots_not_found";
            return;
        }

        var createPlant = FindCreatePlant();
        if (createPlant == null)
        {
            return;
        }

        var size = obs.RowCount * obs.ColumnCount;
        foreach (var slot in obs.SeedSlots.OrderBy(slot => slot.SlotIndex))
        {
            if (!slot.Usable || obs.Sun < slot.SeedCost || !slot.Ready || slot.Disabled)
            {
                continue;
            }

            var plantType = (PlantType)slot.PlantType;

            for (var row = 0; row < obs.RowCount; row++)
            {
                for (var column = 0; column < obs.ColumnCount; column++)
                {
                    if (IsCellOccupied(obs, row, column))
                    {
                        continue;
                    }

                    var action = 1 + slot.SlotIndex * size + row * obs.ColumnCount + column;
                    try
                    {
                        if (createPlant.CheckBox(column, row, plantType))
                        {
                            obs.LegalActions.Add(action);
                        }
                    }
                    catch
                    {
                        return;
                    }
                }
            }
        }
    }

    private static bool IsCellOccupied(ObservationDto obs, int row, int column) =>
        obs.Plants.Any(p => p.Row == row && p.Column == column) ||
        obs.VisiblePlants.Any(p => p.ActiveInHierarchy && p.InBoardBounds && p.Row == row && p.Column == column);

    private PlantCostInfo GetPlantCost(int plantTypeId)
    {
        if (_seedRuntimeCache.Valid &&
            !_seedRuntimeCache.SeedSelectionActive &&
            _seedRuntimeCache.CachedPlantCosts.TryGetValue(plantTypeId, out var cachedCost) &&
            cachedCost > 0)
        {
            return new PlantCostInfo
            {
                PlantType = plantTypeId,
                Cost = cachedCost,
                Source = "cached_active_gameplay_card"
            };
        }

        var fromCard = TryReadPlantCostFromCards(plantTypeId);
        if (fromCard != null)
        {
            return fromCard;
        }

        return GetFallbackPlantCost(plantTypeId);
    }

    private static PlantCostInfo GetFallbackPlantCost(int plantTypeId)
    {
        var fallback = plantTypeId switch
        {
            (int)PlantType.SunFlower => 50,
            (int)PlantType.Peashooter => 100,
            _ => 0
        };

        return new PlantCostInfo
        {
            PlantType = plantTypeId,
            Cost = fallback,
            Source = fallback > 0
                ? "fallback_limited_registry"
                : "unknown_cost"
        };
    }

    private PlantCostInfo? TryReadPlantCostFromCards(int plantTypeId)
    {
        var candidates = new List<int>();
        try
        {
            var cards = Object.FindObjectsOfType<CardUI>();
            foreach (var card in cards)
            {
                try
                {
                    if (card == null || (int)card.thePlantType != plantTypeId)
                    {
                        continue;
                    }

                    var active = false;
                    try { active = card.gameObject != null && card.gameObject.activeInHierarchy; } catch { }
                    var selectedOrBanked = SafeCardSelectedOrBanked(card);
                    var cost = Math.Max(0, card.theSeedCost);
                    if (active && selectedOrBanked && cost > 0)
                    {
                        candidates.Add(cost);
                    }
                }
                catch
                {
                    // Ignore stale CardUI wrappers and continue searching.
                }
            }
        }
        catch
        {
            // Card UI is not available on every scene. The limited fallback below is intentional.
        }

        if (candidates.Count == 0)
        {
            return null;
        }

        return new PlantCostInfo
        {
            PlantType = plantTypeId,
            Cost = candidates.Min(),
            Source = "CardUI.theSeedCost_min_active_selected"
        };
    }

    private CardCooldownDto GetCardCooldown(int plantTypeId)
    {
        if (TryGetCachedGameplayCardCooldown(plantTypeId, out var cachedCooldown))
        {
            return cachedCooldown;
        }

        var cards = FindCardsForPlant(plantTypeId);
        if (cards.Count == 0)
        {
            return new CardCooldownDto
            {
                PlantType = plantTypeId,
                PlantTypeName = ((PlantType)plantTypeId).ToString(),
                Found = false,
                Ready = false,
                Source = "card_not_found"
            };
        }

        CardCooldownDto? best = null;
        foreach (var card in cards)
        {
            try
            {
                var dto = BuildCardCooldownDto(card, plantTypeId, "CardUI.CD/fullCD/isAvailable/disabled");

                if (best == null || IsBetterCardCooldown(dto, best))
                {
                    best = dto;
                }
            }
            catch
            {
                // Ignore stale CardUI wrappers.
            }
        }

        if (best == null)
        {
            return new CardCooldownDto
            {
                PlantType = plantTypeId,
                PlantTypeName = ((PlantType)plantTypeId).ToString(),
                Found = false,
                Ready = false,
                Source = "card_read_failed"
            };
        }

        best.MatchingCardCount = cards.Count;
        return best;
    }

    private bool TryGetCachedGameplayCardCooldown(int plantTypeId, out CardCooldownDto cooldown)
    {
        cooldown = new CardCooldownDto();
        if (!_seedRuntimeCache.Valid ||
            _seedRuntimeCache.SeedSelectionActive ||
            !_seedRuntimeCache.CachedGameplayCards.TryGetValue(plantTypeId, out var card))
        {
            return false;
        }

        try
        {
            if (card == null ||
                card.gameObject == null ||
                !card.gameObject.activeInHierarchy ||
                (int)card.thePlantType != plantTypeId)
            {
                _seedRuntimeCache.CachedGameplayCards.Remove(plantTypeId);
                return false;
            }

            cooldown = BuildCardCooldownDto(card, plantTypeId, "cached_active_gameplay_card");
            cooldown.MatchingCardCount = Math.Max(1, _seedRuntimeCache.ActiveGameplayTypeCounts.TryGetValue(plantTypeId, out var count) ? count : 1);
            return true;
        }
        catch
        {
            _seedRuntimeCache.CachedGameplayCards.Remove(plantTypeId);
            return false;
        }
    }

    private bool TryGetSeedSlotForPlacement(int seedSlotIndex, out SeedSlotDto slot, out CardUI? card)
    {
        slot = new SeedSlotDto { SlotIndex = seedSlotIndex, PlantType = -1, PlantTypeName = "unknown" };
        card = null;
        if (!_seedRuntimeCache.Valid || _seedRuntimeCache.SeedSelectionActive || seedSlotIndex < 0 ||
            seedSlotIndex >= _seedRuntimeCache.CachedSeedSlots.Count)
        {
            BuildSeedProbe();
        }

        if (seedSlotIndex < 0 || seedSlotIndex >= _seedRuntimeCache.CachedSeedSlots.Count)
        {
            return false;
        }

        var entry = _seedRuntimeCache.CachedSeedSlots[seedSlotIndex];
        try
        {
            if (entry.Card == null ||
                entry.Card.gameObject == null ||
                !entry.Card.gameObject.activeInHierarchy ||
                entry.Card.GetInstanceID() != entry.CardInstanceId)
            {
                BuildSeedProbe();
                if (seedSlotIndex < 0 || seedSlotIndex >= _seedRuntimeCache.CachedSeedSlots.Count)
                {
                    return false;
                }

                entry = _seedRuntimeCache.CachedSeedSlots[seedSlotIndex];
            }

            card = entry.Card;
            slot = BuildSeedSlotDto(card, seedSlotIndex, "cached_active_gameplay_card", includeHierarchyPath: false);
            return true;
        }
        catch
        {
            BuildSeedProbe();
            if (seedSlotIndex < 0 || seedSlotIndex >= _seedRuntimeCache.CachedSeedSlots.Count)
            {
                return false;
            }

            entry = _seedRuntimeCache.CachedSeedSlots[seedSlotIndex];
            card = entry.Card;
            slot = BuildSeedSlotDto(card, seedSlotIndex, "cached_active_gameplay_card", includeHierarchyPath: false);
            return true;
        }
    }

    private CardCooldownDto SlotCooldownFromDto(SeedSlotDto slot) => new()
    {
        PlantType = slot.PlantType,
        PlantTypeName = slot.PlantTypeName,
        Found = slot.CardInstanceId != 0,
        Ready = slot.Ready,
        RawCooldown = slot.RawCooldown,
        CurrentCooldown = slot.CurrentCooldown,
        FullCooldown = slot.FullCooldown,
        IsAvailable = slot.IsAvailable,
        Disabled = slot.Disabled,
        OnCardBank = true,
        IsSelected = true,
        SeedCost = slot.SeedCost,
        MatchingCardCount = 1,
        Source = slot.Source
    };

    private SeedSlotDto BuildSeedSlotDto(SeedCardDto card, int slotIndex, string source)
    {
        var seedCost = card.SeedCost;
        var warning = "";
        if (seedCost <= 0)
        {
            var fallback = GetFallbackPlantCost(card.PlantType);
            seedCost = fallback.Cost;
            warning = fallback.Cost > 0
                ? "seedCost was not readable from CardUI; using limited registry fallback."
                : "seedCost was not readable from CardUI and no fallback is known.";
            source = fallback.Source;
        }

        var fullCooldown = Math.Max(0f, card.FullCd);
        var rawCooldown = Math.Max(0f, card.Cd);
        var currentCooldown = Math.Max(0f, fullCooldown - rawCooldown);
        var ready = !card.Disabled && (card.IsAvailable || fullCooldown <= 0.05f || rawCooldown >= fullCooldown - 0.05f);
        return new SeedSlotDto
        {
            SlotIndex = slotIndex,
            CardInstanceId = card.InstanceId,
            PlantType = card.PlantType,
            PlantTypeName = card.PlantTypeName,
            SeedCost = Math.Max(0, seedCost),
            Ready = ready,
            Disabled = card.Disabled,
            IsAvailable = card.IsAvailable,
            RawCooldown = rawCooldown,
            FullCooldown = fullCooldown,
            CurrentCooldown = currentCooldown,
            Usable = card.Active && !card.Disabled && seedCost >= 0,
            Source = source,
            Warning = warning,
            HierarchyPath = card.HierarchyPath
        };
    }

    private SeedSlotDto BuildSeedSlotDto(CardUI card, int slotIndex, string source, bool includeHierarchyPath = true)
    {
        var plantType = (int)card.thePlantType;
        var seedCost = Math.Max(0, card.theSeedCost);
        var warning = "";
        if (seedCost <= 0)
        {
            var fallback = GetFallbackPlantCost(plantType);
            seedCost = fallback.Cost;
            warning = fallback.Cost > 0
                ? "seedCost was not readable from CardUI; using limited registry fallback."
                : "seedCost was not readable from CardUI and no fallback is known.";
            source = fallback.Source;
        }

        var cooldown = BuildCardCooldownDto(card, plantType, source);
        return new SeedSlotDto
        {
            SlotIndex = slotIndex,
            CardInstanceId = card.GetInstanceID(),
            PlantType = plantType,
            PlantTypeName = card.thePlantType.ToString(),
            SeedCost = Math.Max(0, seedCost),
            Ready = cooldown.Ready,
            Disabled = cooldown.Disabled,
            IsAvailable = cooldown.IsAvailable,
            RawCooldown = cooldown.RawCooldown,
            FullCooldown = cooldown.FullCooldown,
            CurrentCooldown = cooldown.CurrentCooldown,
            Usable = card.gameObject != null && card.gameObject.activeInHierarchy && !cooldown.Disabled && seedCost >= 0,
            Source = source,
            Warning = warning,
            HierarchyPath = includeHierarchyPath && card.transform != null ? BuildHierarchyPath(card.transform) : null
        };
    }

    private static List<SeedCardDto> SortSeedSlotCards(IEnumerable<SeedCardDto> cards)
    {
        return cards
            .OrderBy(card => ExtractSeedBankSlotIndex(card.HierarchyPath))
            .ThenBy(card => card.ScreenX)
            .ThenBy(card => card.LocalX)
            .ThenBy(card => card.InstanceId)
            .ToList();
    }

    private static int ExtractSeedBankSlotIndex(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return int.MaxValue;
        }

        var normalized = path.Replace('\\', '/');
        var marker = "/seed";
        var index = normalized.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
        while (index >= 0)
        {
            var start = index + marker.Length;
            var end = start;
            while (end < normalized.Length && char.IsDigit(normalized[end]))
            {
                end++;
            }

            if (end > start && int.TryParse(normalized.Substring(start, end - start), out var value))
            {
                return value;
            }

            index = normalized.IndexOf(marker, start, StringComparison.OrdinalIgnoreCase);
        }

        return int.MaxValue;
    }

    private static CardCooldownDto BuildCardCooldownDto(CardUI card, int plantTypeId, string source)
    {
        var dto = new CardCooldownDto
        {
            PlantType = plantTypeId,
            PlantTypeName = ((PlantType)plantTypeId).ToString(),
            Found = true,
            FullCooldown = Math.Max(0f, card.fullCD),
            IsAvailable = card.isAvailable,
            Disabled = card.disabled,
            OnCardBank = card.onCardBank,
            IsSelected = card.isSelected,
            SeedCost = card.theSeedCost,
            Source = source
        };
        dto.RawCooldown = Math.Max(0f, card.CD);
        dto.CurrentCooldown = Math.Max(0f, dto.FullCooldown - dto.RawCooldown);
        dto.Ready = !dto.Disabled && (dto.IsAvailable || dto.FullCooldown <= 0.05f || dto.RawCooldown >= dto.FullCooldown - 0.05f);
        return dto;
    }

    private static bool IsBetterCardCooldown(CardCooldownDto candidate, CardCooldownDto current)
    {
        if (candidate.OnCardBank != current.OnCardBank)
        {
            return candidate.OnCardBank;
        }

        if (candidate.Disabled != current.Disabled)
        {
            return !candidate.Disabled;
        }

        if (candidate.Ready != current.Ready)
        {
            return candidate.Ready;
        }

        return candidate.CurrentCooldown < current.CurrentCooldown;
    }

    private List<CardUI> FindCardsForPlant(int plantTypeId)
    {
        var result = new List<CardUI>();
        try
        {
            var cards = Object.FindObjectsOfType<CardUI>();
            foreach (var card in cards)
            {
                try
                {
                    if (card != null && (int)card.thePlantType == plantTypeId)
                    {
                        result.Add(card);
                    }
                }
                catch
                {
                    // Ignore stale CardUI wrappers.
                }
            }
        }
        catch
        {
            // Card UI is not available on every scene.
        }

        return result;
    }

    private bool TryStartCardCooldown(int plantTypeId, out string source)
    {
        var targets = SelectCooldownTargets(plantTypeId);
        var updated = 0;
        foreach (var card in targets)
        {
            try
            {
                var fullCd = Math.Max(0f, card.fullCD);
                if (fullCd <= 0f)
                {
                    continue;
                }

                card.CD = 0f;
                card.isAvailable = false;
                updated++;
            }
            catch
            {
                // Continue updating other matching cards.
            }
        }

        source = updated > 0 ? $"CardUI.CD=0 updated={updated}" : "no_card_cooldown_updated";
        return updated > 0;
    }

    private bool TryStartSeedSlotCooldown(CardUI card, int seedSlotIndex, out string source)
    {
        try
        {
            var fullCd = Math.Max(0f, card.fullCD);
            if (fullCd <= 0f)
            {
                source = $"seed_slot[{seedSlotIndex}].fullCD_zero";
                return false;
            }

            card.CD = 0f;
            card.isAvailable = false;
            source = $"seed_slot[{seedSlotIndex}].CardUI.CD=0";
            return true;
        }
        catch (Exception ex)
        {
            source = $"seed_slot[{seedSlotIndex}].cooldown_failed:{ex.Message}";
            return false;
        }
    }

    private void ResetCardCooldowns(List<string> actions)
    {
        var updated = 0;
        var seenCards = new HashSet<int>();
        foreach (var plantTypeId in _config.PlantTypes.Distinct())
        {
            foreach (var card in SelectCooldownTargets(plantTypeId))
            {
                try
                {
                    var id = card.GetInstanceID();
                    if (!seenCards.Add(id))
                    {
                        continue;
                    }

                    card.CD = Math.Max(0f, card.fullCD);
                    card.isAvailable = true;
                    updated++;
                }
                catch
                {
                    // Continue resetting other cards.
                }
            }
        }

        actions.Add($"ResetCardCooldowns(updated={updated})");
    }

    private List<CardUI> SelectCooldownTargets(int plantTypeId)
    {
        if (_seedRuntimeCache.Valid &&
            !_seedRuntimeCache.SeedSelectionActive &&
            _seedRuntimeCache.CachedGameplayCards.TryGetValue(plantTypeId, out var cachedCard))
        {
            try
            {
                if (cachedCard != null &&
                    cachedCard.gameObject != null &&
                    cachedCard.gameObject.activeInHierarchy &&
                    (int)cachedCard.thePlantType == plantTypeId)
                {
                    return new List<CardUI> { cachedCard };
                }
            }
            catch
            {
                _seedRuntimeCache.CachedGameplayCards.Remove(plantTypeId);
            }
        }

        var cards = FindCardsForPlant(plantTypeId);
        var bankCards = cards.Where(card =>
        {
            try { return card.onCardBank; }
            catch { return false; }
        }).ToList();

        if (bankCards.Count > 0)
        {
            return bankCards;
        }

        return cards.Where(card =>
        {
            try { return !card.disabled; }
            catch { return false; }
        }).ToList();
    }

    private bool TryPaySun(Board board, int cost, int sunBefore, out int sunAfter, out string paymentSource)
    {
        if (cost <= 0)
        {
            sunAfter = board.theSun;
            paymentSource = "zero_cost";
            return true;
        }

        try
        {
            board.UseSun(cost);
            sunAfter = board.theSun;
            if (sunBefore - sunAfter == cost)
            {
                paymentSource = "Board.UseSun";
                return true;
            }

            board.SetSun(Math.Max(0, sunBefore - cost));
            sunAfter = board.theSun;
            paymentSource = "Board.SetSun fallback after Board.UseSun";
            return sunBefore - sunAfter == cost;
        }
        catch
        {
            try
            {
                board.SetSun(Math.Max(0, sunBefore - cost));
                sunAfter = board.theSun;
                paymentSource = "Board.SetSun fallback after Board.UseSun exception";
                return sunBefore - sunAfter == cost;
            }
            catch
            {
                sunAfter = board.theSun;
                paymentSource = "payment_failed";
                return false;
            }
        }
    }

    private object BuildDoneInfo()
    {
        var obs = BuildObservation();
        return new { obs.Done, obs.TerminalHint, obs.Over, obs.Wave, obs.MaxWave, obs.ZombieCount };
    }

    private object BuildRewardHints()
    {
        var obs = BuildObservation();
        return new
        {
            obs.KillCount,
            obs.Wave,
            obs.TotalPlantHealth,
            obs.Done,
            obs.TerminalHint,
            nearestZombieXByLane = obs.Lanes.Select(l => new { l.Row, l.NearestZombieX }).ToArray()
        };
    }

    private object SeedProbe()
    {
        return BuildSeedProbe();
    }

    private object UiProbe(JsonElement root)
    {
        var includeAll = ReadBool(root, false, "include_all", "includeAll");
        var maxEntries = Math.Max(1, ReadInt(root, 350, "max_entries", "maxEntries"));
        var entries = BuildUiProbeEntries(includeAll, maxEntries);
        var chooserSignals = entries
            .Where(entry => IsSeedChooserSignal(entry.Name, entry.HierarchyPath, entry.Text, entry.ClassName))
            .ToList();

        return new
        {
            ok = true,
            screenWidth = Screen.width,
            screenHeight = Screen.height,
            includeAll,
            maxEntries,
            activeUiObjectCount = entries.Count,
            seedChooserSignalCount = chooserSignals.Count,
            seedChooserSignals = chooserSignals.ToArray(),
            objects = entries.ToArray(),
            note = "Use --ui-probe to inspect visible active UI names/text/hierarchy before adjusting seed selection classifiers."
        };
    }

    private object AdventureScreenState(JsonElement root)
    {
        var watch = Stopwatch.StartNew();
        var maxEntries = Math.Max(1, ReadInt(root, 500, "max_entries", "maxEntries"));
        var board = FindBoard();
        var restartInfo = DetectRestartScreenInfo(broadScan: true);
        var seedProbe = BuildSeedProbe();
        var entries = BuildUiProbeEntries(includeAll: false, maxEntries: maxEntries);
        var adventureSignals = entries.Where(IsAdventureButtonSignal).ToList();
        var mainMenuSignals = entries.Where(IsMainMenuSignal).ToList();
        var rewardSignals = entries.Where(entry => IsVisibleUiSignal(entry) && IsBlockingRewardUiSignal(entry)).ToList();
        var trophySignals = entries.Where(IsLevelCompleteTrophySignal).ToList();
        var newPlantSignals = entries.Where(IsNewPlantUnlockSignal).ToList();
        var almanacSignals = entries.Where(IsAlmanacOrSeedPacketSignal).ToList();
        var unlockSnapshot = BuildUnlockScreenSnapshot(seedProbe, rewardSignals, newPlantSignals, almanacSignals);
        var startupInfo = DetectStartupPopupInfo(entries, board != null, adventureSignals.Count > 0, mainMenuSignals.Count > 0);
        var lossDetected = HasLossRestartEvidence(restartInfo);
        var rawGameplayReady = false;
        try { rawGameplayReady = ComputeRawGameplayReady(board); } catch { }
        var trophyVisible = trophySignals.Count > 0;
        var rewardVisible = unlockSnapshot.RewardScreenVisible || rewardSignals.Count > 0;
        var unlockVisible = unlockSnapshot.UnlockScreenVisible || unlockSnapshot.NewPlantUnlockedVisible;
        var rewardActive = seedProbe.BlockingRewardUiActive || rewardVisible || unlockVisible || trophyVisible;

        var screenState = ClassifyAdventureScreenState(
            boardFound: board != null,
            startupPopupVisible: startupInfo.StartupPopupVisible,
            lossDetected: lossDetected,
            seedSelectionActive: seedProbe.SeedSelectionActive,
            gameplayReady: rawGameplayReady && !seedProbe.SeedSelectionActive && !seedProbe.BlockingRewardUiActive,
            rewardActive: rewardActive,
            adventureVisible: adventureSignals.Count > 0,
            mainMenuVisible: mainMenuSignals.Count > 0);
        if (trophyVisible)
        {
            screenState = "level_complete_trophy";
        }
        else if (rewardVisible || unlockVisible)
        {
            screenState = "reward_unlock";
        }
        var level = SafeReadGameBoardLevel();
        var firstTrophySignal = trophySignals.FirstOrDefault();

        watch.Stop();
        return new
        {
            ok = true,
            screenState,
            currentMode = SafeReadGameBoardType(),
            isMainMenu = screenState == "main_menu" || screenState == "startup_popup",
            isAdventureButtonVisible = adventureSignals.Count > 0,
            startupPopupVisible = startupInfo.StartupPopupVisible,
            startupOkButtonVisible = startupInfo.StartupOkButtonVisible,
            mainMenuBlockedByPopup = startupInfo.MainMenuBlockedByPopup,
            isSeedSelectionScreen = seedProbe.SeedSelectionActive,
            isGameplayReady = screenState == "gameplay",
            isLevelComplete = screenState == "reward_unlock" || screenState == "level_complete_trophy" || screenState == "reward_screen" || trophyVisible,
            isRewardScreen = screenState == "reward_unlock" || screenState == "level_complete_trophy" || screenState == "reward_screen",
            trophyVisible = trophyVisible,
            levelCompleteTrophyVisible = trophyVisible,
            rewardObjectVisible = rewardSignals.Count > 0 || trophyVisible,
            postWinClickRequired = trophyVisible,
            trophyObjectName = firstTrophySignal?.Name ?? "",
            trophyObjectPath = firstTrophySignal?.HierarchyPath ?? "",
            levelCompleteScreenVisible = screenState == "reward_unlock" || screenState == "level_complete_trophy" || screenState == "reward_screen" || trophyVisible,
            isNewPlantUnlockedScreen = newPlantSignals.Count > 0,
            rewardScreenVisible = unlockSnapshot.RewardScreenVisible,
            unlockScreenVisible = unlockSnapshot.UnlockScreenVisible,
            newPlantUnlockedVisible = unlockSnapshot.NewPlantUnlockedVisible,
            newPlantUnlockedName = unlockSnapshot.NewPlantUnlockedName,
            newPlantUnlockedPlantType = unlockSnapshot.NewPlantUnlockedPlantType,
            visibleRewardTexts = unlockSnapshot.VisibleRewardTexts,
            visibleSeedCardNames = unlockSnapshot.VisibleSeedCardNames,
            visibleSeedPlantTypes = unlockSnapshot.VisibleSeedPlantTypes,
            unknownUnlockObjects = unlockSnapshot.UnknownUnlockObjects,
            unknownVisibleSeedCards = unlockSnapshot.UnknownVisibleSeedCards,
            unlockSnapshot,
            isAlmanacOrSeedPacketScreen = almanacSignals.Count > 0,
            isGameOverScreen = lossDetected,
            currentAdventureLevel = level,
            currentWorldOrStage = level > 0 ? ((level - 1) / 10) + 1 : 0,
            currentDayLevel = level > 0 ? ((level - 1) % 10) + 1 : 0,
            unlockedSeedNames = SeedNames(seedProbe.ActiveGameplaySeedSlots.Select(slot => slot.PlantType)
                .Concat(seedProbe.SelectedSeedBankCards.Select(card => card.PlantType))
                .Concat(seedProbe.AvailableSeedCards.Select(card => card.PlantType))),
            availableSeedNames = SeedNames(seedProbe.AvailableSeedCards.Select(card => card.PlantType)),
            selectedSeedNames = SeedNames(seedProbe.SelectedSeedBankCards.Select(card => card.PlantType)
                .Concat(seedProbe.ActiveGameplaySeedSlots.Select(slot => slot.PlantType))),
            unknownVisibleSeedNames = seedProbe.AvailableSeedCards
                .Where(card => string.IsNullOrWhiteSpace(card.PlantTypeName))
                .Select(card => card.DisplayName ?? card.GameObjectName ?? card.HierarchyPath ?? card.PlantType.ToString())
                .Distinct()
                .ToArray(),
            boardFound = board != null,
            seedSelectionActive = seedProbe.SeedSelectionActive,
            seedSelectionPanelActive = seedProbe.SeedSelectionPanelActive,
            startButtonActive = seedProbe.StartButtonActive,
            blockingRewardUiActive = seedProbe.BlockingRewardUiActive || rewardSignals.Count > 0,
            restartButtonActive = restartInfo.RestartButtonActive,
            restartDetectionReason = restartInfo.RestartDetectionReason,
            adventureSignalCount = adventureSignals.Count,
            mainMenuSignalCount = mainMenuSignals.Count,
            startupOkSignalCount = startupInfo.StartupOkSignals.Count,
            startupPopupSignalCount = startupInfo.StartupPopupSignals.Count,
            rewardSignalCount = rewardSignals.Count,
            trophySignalCount = trophySignals.Count,
            newPlantSignalCount = newPlantSignals.Count,
            almanacSignalCount = almanacSignals.Count,
            adventureSignals = adventureSignals.Take(12).ToArray(),
            startupOkSignals = startupInfo.StartupOkSignals.Take(8).ToArray(),
            startupPopupSignals = startupInfo.StartupPopupSignals.Take(8).ToArray(),
            rewardSignals = rewardSignals.Take(12).ToArray(),
            trophySignals = trophySignals.Take(12).ToArray(),
            screen_state_ms = Math.Round(watch.Elapsed.TotalMilliseconds, 3)
        };
    }

    private object PressAdventureOnce(JsonElement root)
    {
        var before = AdventureScreenState(root);
        var actions = new List<string>();
        var entries = BuildUiProbeEntries(includeAll: false, maxEntries: 500);
        var startupInfo = DetectStartupPopupInfo(
            entries,
            FindBoard() != null,
            entries.Any(IsAdventureButtonSignal),
            entries.Any(IsMainMenuSignal));
        if (startupInfo.StartupPopupVisible)
        {
            actions.Add("Adventure click refused because startup popup is active.");
            return new
            {
                ok = false,
                clicked = false,
                methodUsed = "refused_startup_popup_active",
                targetName = "",
                targetPath = "",
                actions,
                before,
                after = AdventureScreenState(root),
                message = "Startup popup is active; dismiss it before clicking Adventure."
            };
        }
        var clicked = TryClickFirstVisibleUiSignal(
            "Adventure",
            IsAdventureButtonSignalText,
            actions,
            out var methodUsed,
            out var targetName,
            out var targetPath);
        if (clicked)
        {
            InvalidateSeedRuntimeCache("press_adventure_once");
            InvalidateRestartUiCache("press_adventure_once");
        }

        return new
        {
            ok = clicked,
            clicked,
            methodUsed,
            targetName,
            targetPath,
            actions,
            before,
            after = AdventureScreenState(root),
            message = clicked ? "Adventure button clicked." : "No visible Adventure button candidate could be clicked."
        };
    }

    private object ClickStartupOkOnce(JsonElement root)
    {
        var before = AdventureScreenState(root);
        var actions = new List<string>();
        var entries = BuildUiProbeEntries(includeAll: false, maxEntries: 500);
        var startupInfo = DetectStartupPopupInfo(
            entries,
            FindBoard() != null,
            entries.Any(IsAdventureButtonSignal),
            entries.Any(IsMainMenuSignal));
        var clicked = false;
        var methodUsed = "";
        var targetName = "";
        var targetPath = "";
        if (startupInfo.StartupOkButtonVisible)
        {
            clicked = TryClickFirstVisibleUiSignal(
                "StartupOK",
                IsStartupOkSignalText,
                actions,
                out methodUsed,
                out targetName,
                out targetPath);
        }

        if (!clicked && startupInfo.StartupPopupVisible)
        {
            clicked = TryNativeMouseClickNormalized(
                xNorm: 860f / 1760f,
                yNormFromTop: 625f / 900f,
                actions,
                out methodUsed);
            if (clicked)
            {
                targetName = "startup_ok_coordinate_fallback";
                targetPath = "";
            }
        }

        if (clicked)
        {
            InvalidateSeedRuntimeCache("click_startup_ok_once");
            InvalidateRestartUiCache("click_startup_ok_once");
        }

        return new
        {
            ok = clicked,
            clicked,
            methodUsed,
            targetName,
            targetPath,
            startupPopupVisible = startupInfo.StartupPopupVisible,
            startupOkButtonVisible = startupInfo.StartupOkButtonVisible,
            mainMenuBlockedByPopup = startupInfo.MainMenuBlockedByPopup,
            actions,
            before,
            after = AdventureScreenState(root),
            message = clicked ? "Startup OK button clicked." : "No startup OK candidate could be clicked."
        };
    }

    private object ClickTrophyOnce(JsonElement root)
    {
        var before = AdventureScreenState(root);
        var actions = new List<string>();
        var guardObservation = BuildObservation(forceSeedProbe: true, forceRestartProbe: true);
        if (ShouldBlockActiveGameplayMutation("trophy click-through", guardObservation, actions))
        {
            return new
            {
                ok = false,
                clicked = false,
                blockedBySafety = true,
                methodUsed = "blocked_active_gameplay",
                targetName = "",
                targetPath = "",
                trophyVisible = false,
                levelCompleteTrophyVisible = false,
                rewardObjectVisible = false,
                postWinClickRequired = false,
                trophyObjectName = "",
                trophyObjectPath = "",
                trophySignalCount = 0,
                trophySignals = Array.Empty<object>(),
                actions,
                before,
                after = AdventureScreenState(root),
                message = "Blocked trophy click-through while active gameplay is still running."
            };
        }

        var entries = BuildUiProbeEntries(includeAll: false, maxEntries: 500);
        var trophySignals = entries.Where(IsLevelCompleteTrophySignal).ToList();
        var firstTrophySignal = trophySignals.FirstOrDefault();
        var clicked = TryClickFirstVisibleUiSignal(
            "LevelCompleteTrophy",
            IsTrophySignalText,
            actions,
            out var methodUsed,
            out var targetName,
            out var targetPath);
        if (!clicked && trophySignals.Count > 0)
        {
            clicked = TryNativeMouseClickNormalized(
                xNorm: 0.5f,
                yNormFromTop: 0.46f,
                actions,
                out methodUsed);
            if (clicked)
            {
                targetName = firstTrophySignal?.Name ?? "level_complete_trophy_coordinate_fallback";
                targetPath = firstTrophySignal?.HierarchyPath ?? "";
            }
        }
        if (clicked)
        {
            InvalidateSeedRuntimeCache("click_trophy_once");
            InvalidateRestartUiCache("click_trophy_once");
        }

        return new
        {
            ok = clicked,
            clicked,
            methodUsed,
            targetName,
            targetPath,
            trophyVisible = trophySignals.Count > 0,
            levelCompleteTrophyVisible = trophySignals.Count > 0,
            rewardObjectVisible = trophySignals.Count > 0,
            postWinClickRequired = trophySignals.Count > 0,
            trophyObjectName = firstTrophySignal?.Name ?? "",
            trophyObjectPath = firstTrophySignal?.HierarchyPath ?? "",
            trophySignalCount = trophySignals.Count,
            trophySignals = trophySignals.Take(12).ToArray(),
            actions,
            before,
            after = AdventureScreenState(root),
            message = clicked ? "Level-complete trophy/reward object clicked." : "No visible level-complete trophy candidate could be clicked."
        };
    }

    private object ClickRewardContinueOnce(JsonElement root)
    {
        var before = AdventureScreenState(root);
        var actions = new List<string>();
        var guardObservation = BuildObservation(forceSeedProbe: true, forceRestartProbe: true);
        if (ShouldBlockActiveGameplayMutation("reward continue cleanup", guardObservation, actions))
        {
            return new
            {
                ok = false,
                clicked = false,
                blockedBySafety = true,
                methodUsed = "blocked_active_gameplay",
                targetName = "",
                targetPath = "",
                cleanupCount = 0,
                actions,
                before,
                after = AdventureScreenState(root),
                message = "Blocked reward/unlock cleanup while active gameplay is still running."
            };
        }

        var clicked = TryClickFirstVisibleUiSignal(
            "RewardContinue",
            IsRewardContinueSignalText,
            actions,
            out var methodUsed,
            out var targetName,
            out var targetPath);
        var cleanupCount = 0;
        if (!clicked)
        {
            var board = FindBoard();
            if (board != null)
            {
                cleanupCount = DestroyBlockingRewardObjects(board, actions);
                if (cleanupCount > 0)
                {
                    clicked = true;
                    methodUsed = "DestroyBlockingRewardObjects";
                    targetName = "blocking_reward_ui";
                    targetPath = "";
                }
            }
        }
        if (clicked)
        {
            InvalidateSeedRuntimeCache("click_reward_continue_once");
            InvalidateRestartUiCache("click_reward_continue_once");
        }

        return new
        {
            ok = clicked,
            clicked,
            methodUsed,
            targetName,
            targetPath,
            cleanupCount,
            actions,
            before,
            after = AdventureScreenState(root),
            message = clicked ? "Reward/unlock UI click or cleanup was invoked." : "No reward/unlock continue candidate could be clicked."
        };
    }

    private object ClickTryAgainOnce(JsonElement root)
    {
        var before = AdventureScreenState(root);
        var actions = new List<string>();
        var guardObservation = BuildObservation(forceSeedProbe: true, forceRestartProbe: true);
        if (ShouldBlockActiveGameplayMutation("try again click", guardObservation, actions))
        {
            return new
            {
                ok = false,
                clicked = false,
                blockedBySafety = true,
                methodUsed = "blocked_active_gameplay",
                targetName = "",
                targetPath = "",
                error = "",
                actions,
                before,
                after = AdventureScreenState(root),
                message = "Blocked Try Again/Restart click while active gameplay is still running."
            };
        }

        var restartInfo = DetectRestartScreenInfo(broadScan: true);
        var clicked = TryInvokeLoseMenuRestart(
            actions,
            restartInfo,
            out var methodUsed,
            out var targetName,
            out var targetPath,
            out var error);
        if (clicked)
        {
            InvalidateSeedRuntimeCache("click_try_again_once");
            InvalidateRestartUiCache("click_try_again_once");
        }

        return new
        {
            ok = clicked,
            clicked,
            methodUsed,
            targetName,
            targetPath,
            error,
            actions,
            before,
            after = AdventureScreenState(root),
            message = clicked ? "Try Again/Restart button clicked." : "No Try Again/Restart candidate could be clicked."
        };
    }

    private static string ClassifyAdventureScreenState(
        bool boardFound,
        bool startupPopupVisible,
        bool lossDetected,
        bool seedSelectionActive,
        bool gameplayReady,
        bool rewardActive,
        bool adventureVisible,
        bool mainMenuVisible)
    {
        if (startupPopupVisible)
        {
            return "startup_popup";
        }
        if (rewardActive)
        {
            return "reward_unlock";
        }
        if (seedSelectionActive)
        {
            return "seed_selection";
        }
        if (gameplayReady)
        {
            return "gameplay";
        }
        if (lossDetected)
        {
            return "game_over";
        }
        if (adventureVisible || (!boardFound && mainMenuVisible))
        {
            return "main_menu";
        }
        if (!boardFound)
        {
            return "loading_or_menu";
        }
        return "transition";
    }

    private static string[] SeedNames(IEnumerable<int> plantTypes) =>
        plantTypes
            .Where(plantType => plantType >= 0)
            .Select(plantType => ((PlantType)plantType).ToString())
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .Distinct()
            .OrderBy(name => name)
            .ToArray();

    private UnlockScreenSnapshotDto BuildUnlockScreenSnapshot(
        SeedProbeDto seedProbe,
        List<UiProbeEntryDto> rewardSignals,
        List<UiProbeEntryDto> newPlantSignals,
        List<UiProbeEntryDto> almanacSignals)
    {
        var visibleCards = seedProbe.AvailableSeedCards
            .Concat(seedProbe.SelectedSeedBankCards)
            .Concat(seedProbe.ActiveGameplayCardBankCards)
            .Concat(seedProbe.RuntimeCardWrappers)
            .Where(IsVisibleOnScreenCard)
            .GroupBy(card => card.InstanceId)
            .Select(group => group.First())
            .ToList();
        var signalEntries = rewardSignals
            .Concat(newPlantSignals)
            .Concat(almanacSignals)
            .Where(IsVisibleUiSignal)
            .GroupBy(entry => $"{entry.Name}|{entry.HierarchyPath}|{entry.Text}|{entry.ClassName}")
            .Select(group => group.First())
            .ToList();
        var inferredTypes = new Dictionary<int, string>();
        foreach (var card in visibleCards)
        {
            if (card.PlantType >= 0)
            {
                inferredTypes[card.PlantType] = PlantTypeName(card.PlantType);
            }
        }
        foreach (var entry in signalEntries)
        {
            var plantType = InferPlantTypeFromUiSignal(entry);
            if (plantType >= 0)
            {
                inferredTypes[plantType] = PlantTypeName(plantType);
            }
        }

        var newPlantType = -1;
        foreach (var entry in newPlantSignals.Concat(rewardSignals).Concat(almanacSignals))
        {
            newPlantType = InferPlantTypeFromUiSignal(entry);
            if (newPlantType >= 0)
            {
                break;
            }
        }
        if (newPlantType < 0 && visibleCards.Count == 1)
        {
            newPlantType = visibleCards[0].PlantType;
        }

        var unknownUnlockObjects = signalEntries
            .Where(entry => InferPlantTypeFromUiSignal(entry) < 0)
            .Take(24)
            .ToArray();
        var unknownVisibleSeedCards = visibleCards
            .Where(card => card.PlantType < 0 || string.IsNullOrWhiteSpace(card.PlantTypeName) || PlantTypeName(card.PlantType) == card.PlantType.ToString())
            .Take(24)
            .ToArray();

        return new UnlockScreenSnapshotDto
        {
            RewardScreenVisible = rewardSignals.Count > 0 || seedProbe.BlockingRewardUiActive,
            UnlockScreenVisible = newPlantSignals.Count > 0 || almanacSignals.Count > 0,
            NewPlantUnlockedVisible = newPlantSignals.Count > 0 || newPlantType >= 0,
            NewPlantUnlockedName = newPlantType >= 0 ? PlantTypeName(newPlantType) : "",
            NewPlantUnlockedPlantType = newPlantType,
            VisibleRewardTexts = signalEntries
                .Select(SummarizeUiSignal)
                .Where(text => !string.IsNullOrWhiteSpace(text))
                .Distinct()
                .Take(32)
                .ToArray(),
            VisibleSeedCardNames = inferredTypes
                .OrderBy(pair => pair.Key)
                .Select(pair => pair.Value)
                .Where(name => !string.IsNullOrWhiteSpace(name))
                .Distinct()
                .ToArray(),
            VisibleSeedPlantTypes = inferredTypes.Keys
                .Where(plantType => plantType >= 0)
                .Distinct()
                .OrderBy(plantType => plantType)
                .ToArray(),
            UnknownUnlockObjects = unknownUnlockObjects,
            UnknownVisibleSeedCards = unknownVisibleSeedCards
        };
    }

    private static string PlantTypeName(int plantType)
    {
        if (plantType < 0)
        {
            return "";
        }
        try
        {
            var name = ((PlantType)plantType).ToString();
            return string.IsNullOrWhiteSpace(name) ? plantType.ToString() : name;
        }
        catch
        {
            return plantType.ToString();
        }
    }

    private static string SummarizeUiSignal(UiProbeEntryDto entry)
    {
        var parts = new[] { entry.Text, entry.Name, entry.ClassName, entry.HierarchyPath }
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .Select(value => value!.Trim());
        return string.Join(" | ", parts);
    }

    private static int InferPlantTypeFromUiSignal(UiProbeEntryDto entry) =>
        InferPlantTypeFromText($"{entry.Text} {entry.Name} {entry.HierarchyPath} {entry.ClassName}");

    private static int InferPlantTypeFromText(string? value)
    {
        var normalized = NormalizeUiText(value);
        if (string.IsNullOrWhiteSpace(normalized))
        {
            return -1;
        }
        if (normalized.Contains("cherrybomb") ||
            normalized.Contains("cherry") ||
            normalized.Contains("bomb"))
        {
            return (int)PlantType.CherryBomb;
        }
        if (normalized.Contains("sunflower"))
        {
            return (int)PlantType.SunFlower;
        }
        if (normalized.Contains("peashooter") ||
            normalized.Contains("peashoter") ||
            normalized.Contains("peaseed") ||
            normalized.Contains("pea"))
        {
            return (int)PlantType.Peashooter;
        }
        if (normalized.Contains("wallnut") ||
            normalized.Contains("walnut") ||
            normalized.Contains("nut"))
        {
            return (int)PlantType.WallNut;
        }
        return -1;
    }

    private static StartupPopupInfo DetectStartupPopupInfo(
        List<UiProbeEntryDto> entries,
        bool boardFound,
        bool adventureVisible,
        bool mainMenuVisible)
    {
        var okSignals = entries.Where(IsStartupOkSignal).ToList();
        var popupSignals = entries.Where(IsStartupPopupSignal).ToList();
        var visible = (popupSignals.Count > 0 && !boardFound && (adventureVisible || mainMenuVisible)) ||
                      (okSignals.Count > 0 &&
                       (popupSignals.Count > 0 ||
                        (!boardFound && (adventureVisible || mainMenuVisible))));
        return new StartupPopupInfo
        {
            StartupPopupVisible = visible,
            StartupOkButtonVisible = okSignals.Count > 0,
            MainMenuBlockedByPopup = visible && (adventureVisible || mainMenuVisible),
            StartupOkSignals = okSignals,
            StartupPopupSignals = popupSignals
        };
    }

    private static bool IsStartupOkSignal(UiProbeEntryDto entry) =>
        IsVisibleUiSignal(entry) &&
        IsStartupOkSignalText(NormalizeUiText($"{entry.Name} {entry.HierarchyPath} {entry.Text} {entry.ClassName}"));

    private static bool IsStartupPopupSignal(UiProbeEntryDto entry)
    {
        if (!IsVisibleUiSignal(entry))
        {
            return false;
        }
        var normalized = NormalizeUiText($"{entry.Name} {entry.HierarchyPath} {entry.Text} {entry.ClassName}");
        return normalized.Contains("popup") ||
               normalized.Contains("noticepausemenu") ||
               normalized.Contains("modal") ||
               normalized.Contains("notice") ||
               normalized.Contains("announce") ||
               normalized.Contains("announcement") ||
               normalized.Contains("community") ||
               normalized.Contains("discord") ||
               normalized.Contains("qq") ||
               normalized.Contains("group") ||
               normalized.Contains("prompt") ||
               normalized.Contains("dialog") ||
               normalized.Contains("mask") ||
               normalized.Contains("window");
    }

    private static bool IsAdventureButtonSignal(UiProbeEntryDto entry) =>
        IsVisibleUiSignal(entry) &&
        IsAdventureButtonSignalText(NormalizeUiText($"{entry.Name} {entry.HierarchyPath} {entry.Text} {entry.ClassName}"));

    private static bool IsMainMenuSignal(UiProbeEntryDto entry)
    {
        if (!IsVisibleUiSignal(entry))
        {
            return false;
        }
        var normalized = NormalizeUiText($"{entry.Name} {entry.HierarchyPath} {entry.Text} {entry.ClassName}");
        return normalized.Contains("selectscreen") ||
               normalized.Contains("selectorscreen") ||
               normalized.Contains("mainmenu") ||
               normalized.Contains("woodsign") ||
               normalized.Contains("startadventure") ||
               normalized.Contains("newadv");
    }

    private static bool IsNewPlantUnlockSignal(UiProbeEntryDto entry)
    {
        if (!IsVisibleUiSignal(entry))
        {
            return false;
        }
        var normalized = NormalizeUiText($"{entry.Name} {entry.HierarchyPath} {entry.Text} {entry.ClassName}");
        return normalized.Contains("newplant") ||
               normalized.Contains("seedpacket") ||
               (normalized.Contains("presentopen") && !IsGameplayGiftDropSignal(normalized)) ||
               normalized.Contains("award");
    }

    private static bool IsAlmanacOrSeedPacketSignal(UiProbeEntryDto entry)
    {
        if (!IsVisibleUiSignal(entry))
        {
            return false;
        }
        var normalized = NormalizeUiText($"{entry.Name} {entry.HierarchyPath} {entry.Text} {entry.ClassName}");
        return normalized.Contains("almanac") ||
               normalized.Contains("seedpacket") ||
               normalized.Contains("plantcard");
    }

    private static bool IsLevelCompleteTrophySignal(UiProbeEntryDto entry)
    {
        if (!IsVisibleUiSignal(entry))
        {
            return false;
        }
        var normalized = NormalizeUiText($"{entry.Name} {entry.HierarchyPath} {entry.Text} {entry.ClassName}");
        return !IsGameplayGiftDropSignal(normalized) && IsTrophySignalText(normalized);
    }

    private static bool IsAdventureButtonSignalText(string normalized) =>
        normalized.Contains("startadventure") ||
        normalized.Contains("selectorScreenAdventure".ToLowerInvariant()) ||
        normalized.Contains("selectorscreenadventure") ||
        normalized.Contains("newadv") ||
        normalized.Contains("advanture") ||
        (normalized.Contains("adventure") &&
         !normalized.Contains("adventureeval") &&
         !normalized.Contains("adventuremodeprogression"));

    private static bool IsStartupOkSignalText(string normalized)
    {
        if (string.IsNullOrWhiteSpace(normalized) ||
            normalized.Contains("lookalmanac") ||
            normalized.Contains("cook") ||
            normalized.Contains("book") ||
            normalized.Contains("unlock"))
        {
            return false;
        }
        if (normalized.Contains("noticepausemenu") &&
            (normalized.Contains("pausemenubtn") ||
             normalized.EndsWith("image", StringComparison.Ordinal) ||
             normalized.Contains("noticepausemenuimage")))
        {
            return true;
        }
        return normalized == "ok" ||
               normalized == "okay" ||
               normalized == "confirm" ||
               normalized == "sure" ||
               normalized.EndsWith("ok", StringComparison.Ordinal) ||
               normalized.Contains("buttonok") ||
               normalized.Contains("okbutton") ||
               normalized.Contains("greenok");
    }

    private static bool IsRewardContinueSignalText(string normalized) =>
        normalized.Contains("continue") ||
        normalized.Contains("confirm") ||
        normalized.Contains("ok") ||
        normalized.Contains("close") ||
        normalized.Contains("next") ||
        normalized.Contains("collect") ||
        normalized.Contains("trophy") ||
        normalized.Contains("reward") ||
        normalized.Contains("prize") ||
        normalized.Contains("award") ||
        normalized.Contains("present") ||
               normalized.Contains("newplant") ||
               normalized.Contains("seedpacket") ||
               normalized.Contains("levelcomplete") ||
               normalized.Contains("levelcompleted");

    private static bool IsTrophySignalText(string normalized) =>
        normalized.Contains("trophy") ||
        normalized.Contains("levelcomplete") ||
        normalized.Contains("levelcompleted") ||
        normalized.Contains("levelcleared") ||
        ((normalized.Contains("award") ||
          normalized.Contains("reward") ||
          normalized.Contains("prize")) &&
         !normalized.Contains("newplant") &&
         !normalized.Contains("seedpacket") &&
         !normalized.Contains("plantcard") &&
         !normalized.Contains("almanac"));

    private bool TryClickFirstVisibleUiSignal(
        string label,
        Func<string, bool> normalizedPredicate,
        List<string> actions,
        out string methodUsed,
        out string targetName,
        out string targetPath)
    {
        methodUsed = "";
        targetName = "";
        targetPath = "";
        foreach (var gameObject in FindVisibleUiSignalObjects(normalizedPredicate))
        {
            try
            {
                targetName = SafeObjectName(gameObject) ?? label;
                targetPath = BuildHierarchyPath(gameObject.transform);
                if (TryInvokeVisibleButtonObject(gameObject, actions, out methodUsed) ||
                    TryNativeMouseClickGameObject(gameObject, actions, out methodUsed))
                {
                    actions.Add($"{label} clicked via {methodUsed}: {targetPath}");
                    return true;
                }
            }
            catch (Exception ex)
            {
                actions.Add($"{label} click candidate failed: {SafeObjectName(gameObject)} {ex.Message}");
            }
        }
        actions.Add($"{label} click candidate not found.");
        return false;
    }

    private List<GameObject> FindVisibleUiSignalObjects(Func<string, bool> normalizedPredicate)
    {
        var result = new List<GameObject>();
        try
        {
            foreach (var transform in Object.FindObjectsOfType<Transform>())
            {
                try
                {
                    if (transform == null || transform.gameObject == null || !transform.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    WorldToScreen(transform.position, out var inBounds);
                    if (!inBounds)
                    {
                        continue;
                    }

                    var gameObject = transform.gameObject;
                    var normalized = NormalizeUiText(
                        $"{SafeObjectName(gameObject)} {BuildHierarchyPath(transform)} {SafeReadGameObjectText(gameObject)} {string.Join(",", SafeComponentTypeNames(gameObject))}");
                    if (normalizedPredicate(normalized))
                    {
                        AddButtonCandidateWithAncestors(result, gameObject);
                    }
                }
                catch { }
            }
        }
        catch { }

        return result
            .Distinct()
            .OrderByDescending(gameObject => normalizedPredicate(NormalizeUiText(
                $"{SafeObjectName(gameObject)} {BuildHierarchyPath(gameObject.transform)} {SafeReadGameObjectText(gameObject)} {string.Join(",", SafeComponentTypeNames(gameObject))}")) ? 1 : 0)
            .ThenByDescending(gameObject => LooksLikeButtonType(string.Join(",", SafeComponentTypeNames(gameObject))) ? 1 : 0)
            .ThenBy(gameObject => BuildHierarchyPath(gameObject.transform))
            .ToList();
    }

    private SeedProbeDto BuildSeedProbe()
    {
        var seedProbeWatch = Stopwatch.StartNew();
        var board = FindBoard();
        var initBoard = FindInitBoard();
        var cards = ScanSeedCards();
        var gameplayReady = ComputeRawGameplayReady(board);

        var boardStartMove = false;
        try { boardStartMove = board?.startMove ?? false; } catch { }

        var gameplayBankIds = ReadInGameCardBankIds();
        var uiWatch = Stopwatch.StartNew();
        var seedChooserSignals = BuildSeedChooserSignals();
        uiWatch.Stop();
        var blockingRewardUiActive = seedChooserSignals.Any(signal => IsVisibleUiSignal(signal) && IsBlockingRewardUiSignal(signal));
        var exactChooserPanelActive = seedChooserSignals.Any(signal => IsVisibleUiSignal(signal) && IsChooserPanelPath(signal.HierarchyPath));
        var exactStartButtonActive = seedChooserSignals.Any(signal => IsVisibleUiSignal(signal) && IsStartButtonPath(signal.HierarchyPath)) || IsStartButtonActive();
        var exactSeedBankActive = seedChooserSignals.Any(signal => IsVisibleUiSignal(signal) && IsSeedBankSeedGroupPath(signal.HierarchyPath));
        var exactChooserPacketVisible = cards.Any(card => IsVisibleOnScreenCard(card) && IsChooserPacketPath(card.HierarchyPath));
        var exactChooserUiVisible = exactChooserPanelActive ||
                                    exactStartButtonActive ||
                                    exactChooserPacketVisible ||
                                    (exactSeedBankActive && (exactChooserPanelActive || exactStartButtonActive || exactChooserPacketVisible));
        var rawSeedSelectionPanelActive = exactChooserPanelActive || seedChooserSignals.Any(signal => IsVisibleUiSignal(signal) && IsPanelOrTitleSignal(signal));
        var rawChooseYourPlantsTextActive = exactChooserPanelActive || seedChooserSignals.Any(signal => IsVisibleUiSignal(signal) && IsChooseYourPlantsSignal(signal));
        var rawStartButtonActive = exactStartButtonActive || seedChooserSignals.Any(signal => IsVisibleUiSignal(signal) && IsLetsRockSignal(signal));
        var definiteGameplayActive = boardStartMove &&
                                     cards.Any(card => card.UiVisible &&
                                                       !IsChooserPacketPath(card.HierarchyPath) &&
                                                       !IsSelectedSeedBankPath(card.HierarchyPath) &&
                                                       (card.OnCardBank || gameplayBankIds.Contains(card.InstanceId) || NameSuggestsGameplayBank(card)) &&
                                                       !NameSuggestsAvailableChooser(card));
        var rawChooserVisualActive = !blockingRewardUiActive &&
                                     (exactChooserUiVisible ||
                                      (!definiteGameplayActive &&
                                       (rawSeedSelectionPanelActive ||
                                        rawChooseYourPlantsTextActive ||
                                        rawStartButtonActive ||
                                        IsSeedSelectionPanelActive(initBoard, board))));

        foreach (var card in cards)
        {
            var chooserPacketCard = IsVisibleOnScreenCard(card) && IsChooserPacketPath(card.HierarchyPath);
            var selectedBankPathCard = card.UiVisible && IsSelectedSeedBankPath(card.HierarchyPath);
            var visibleSelectedBankPathCard = IsVisibleOnScreenCard(card) && IsSelectedSeedBankPath(card.HierarchyPath);
            var gameplayCard = card.UiVisible && !rawChooserVisualActive && (gameplayReady || boardStartMove) &&
                               !chooserPacketCard &&
                               (selectedBankPathCard || card.OnCardBank || gameplayBankIds.Contains(card.InstanceId) || NameSuggestsGameplayBank(card));
            var selectedBankCard = rawChooserVisualActive && visibleSelectedBankPathCard;
            var availableCard = rawChooserVisualActive
                ? chooserPacketCard && !selectedBankCard && !card.Disabled
                : card.UiVisible && !selectedBankCard && !gameplayCard && !card.Disabled && !NameSuggestsGameplayBank(card);
            var staleCard = (card.PreSelected || card.IsSelected || card.OnCardBank) &&
                            !selectedBankCard && !gameplayCard;

            card.Classification = gameplayCard
                ? "activeGameplayCardBank"
                : selectedBankCard
                ? "selectedSeedBank"
                : availableCard
                ? "availableSeedCard"
                : staleCard
                ? "stalePreselectedCard"
                : "runtimeCardWrapper";
        }

        var selectedBankCards = cards.Where(card => card.Classification == "selectedSeedBank").ToList();
        var availableSeedCards = cards.Where(card => card.Classification == "availableSeedCard").ToList();
        var activeGameplayCardBankCards = cards.Where(card => card.Classification == "activeGameplayCardBank").ToList();
        var stalePreselectedCards = cards.Where(card => card.Classification == "stalePreselectedCard").ToList();
        var seedSelectionActive = !blockingRewardUiActive &&
                                  (rawChooserVisualActive ||
                                  ((availableSeedCards.Count > 0 || selectedBankCards.Count > 0) &&
                                   activeGameplayCardBankCards.Count == 0));
        var seedSelectionPanelActive = seedSelectionActive && rawSeedSelectionPanelActive;
        var chooseYourPlantsTextActive = seedSelectionActive && rawChooseYourPlantsTextActive;
        var startButtonActive = seedSelectionActive && rawStartButtonActive;
        var effectiveGameplayReady = (seedSelectionActive || blockingRewardUiActive) ? false : gameplayReady;

        var probe = new SeedProbeDto
        {
            BoardFound = board != null,
            InitBoardFound = initBoard != null,
            SeedSelectionActive = seedSelectionActive,
            SeedSelectionPanelActive = seedSelectionPanelActive,
            ChooseYourPlantsTextActive = chooseYourPlantsTextActive,
            StartButtonActive = startButtonActive,
            BlockingRewardUiActive = blockingRewardUiActive,
            BoardCardSelectable = board?.cardSelectable ?? false,
            BoardStartMove = boardStartMove,
            GameplayReady = effectiveGameplayReady,
            GameBoardType = SafeReadGameBoardType(),
            GameBoardLevel = SafeReadGameBoardLevel(),
            ConfiguredPlantTypes = _config.PlantTypes.ToArray(),
            CardCount = cards.Count,
            SelectedOrBankedCount = cards.Count(card => card.OnCardBank || card.IsSelected),
            AvailableSeedPacketCount = availableSeedCards.Count,
            SelectedBankVisibleCount = selectedBankCards.Count,
            SelectedBankPlantTypes = selectedBankCards.Select(card => card.PlantType).ToArray(),
            SelectedBankPlantTypeCounts = BuildPlantTypeCounts(selectedBankCards.Select(card => card.PlantType)).ToArray(),
            AvailableCardVisibleCount = availableSeedCards.Count,
            AvailableCardPlantTypes = availableSeedCards.Select(card => card.PlantType).Distinct().ToArray(),
            AvailableCardPlantTypeCounts = BuildPlantTypeCounts(availableSeedCards.Select(card => card.PlantType)).ToArray(),
            SeedChooserSignalCount = seedChooserSignals.Count,
            SeedChooserSignals = seedChooserSignals.ToArray(),
            StartReadyAvailable = initBoard != null && board != null,
            KnownHooks = new[]
            {
                "CardUI.OnMouseDown()",
                "UIButton/UIBtn/InputButton.OnMouseUpAsButton()",
                "InitBoard.ReadySetPlant()",
                "InitBoard.StartInit()"
            },
            Investigation = "Read-only probe. UI seed lists are split into available, selected-bank, active gameplay-bank, stale/preselected, and runtime-wrapper groups."
        };

        probe.Cards.AddRange(cards);
        probe.RuntimeCardWrappers.AddRange(cards);
        probe.AvailableSeedCards.AddRange(availableSeedCards);
        probe.SelectedSeedBankCards.AddRange(selectedBankCards);
        probe.ActiveGameplayCardBankCards.AddRange(activeGameplayCardBankCards);
        var activeGameplaySeedSlots = SortSeedSlotCards(activeGameplayCardBankCards);
        for (var i = 0; i < activeGameplaySeedSlots.Count; i++)
        {
            probe.ActiveGameplaySeedSlots.Add(BuildSeedSlotDto(activeGameplaySeedSlots[i], i, "active_gameplay_card_bank"));
        }
        probe.StalePreselectedCards.AddRange(stalePreselectedCards);
        probe.seed_probe_ms = Math.Round(seedProbeWatch.Elapsed.TotalMilliseconds, 3);
        probe.ui_scan_ms = Math.Round(uiWatch.Elapsed.TotalMilliseconds, 3);
        UpdateSeedRuntimeCache(probe);
        return probe;
    }

    private object AlmanacProbe(JsonElement root)
    {
        var includeAll = ReadBool(root, false, "include_all", "includeAll");
        var requested = ReadIntArray(root, "plant_types", "plantTypes");
        if (requested.Count == 0)
        {
            requested.AddRange(_config.PlantTypes);
        }

        var runtimeCards = ScanSeedCards();
        var enumEntries = new List<object>();
        var plantEntries = new List<object>();
        try
        {
            foreach (PlantType plantType in Enum.GetValues(typeof(PlantType)))
            {
                var id = (int)plantType;
                if (includeAll)
                {
                    enumEntries.Add(new
                    {
                        plantType = id,
                        plantTypeName = plantType.ToString()
                    });
                }

                if (!includeAll && !requested.Contains(id))
                {
                    continue;
                }

                var cost = GetPlantCost(id);
                var cooldown = GetCardCooldown(id);
                var cardMatches = runtimeCards.Where(card => card.PlantType == id).ToArray();
                plantEntries.Add(new
                {
                    plantType = id,
                    plantTypeName = plantType.ToString(),
                    displayName = plantType.ToString(),
                    cost = cost.Cost,
                    costSource = cost.Source,
                    cooldown = cooldown.CurrentCooldown,
                    fullCooldown = cooldown.FullCooldown,
                    cooldownSource = cooldown.Source,
                    cardFound = cooldown.Found,
                    runtimeCardCount = cardMatches.Length,
                    runtimeCards = cardMatches,
                    description = (string?)null,
                    role = (string?)null,
                    metadataSource = "runtime enum/CardUI; Python registry/almanac dump may enrich this result"
                });
            }
        }
        catch (Exception ex)
        {
            return new
            {
                ok = false,
                error = ex.Message,
                plants = plantEntries.ToArray(),
                enumEntries = enumEntries.ToArray()
            };
        }

        return new
        {
            ok = true,
            runtimeCardCount = runtimeCards.Count,
            plants = plantEntries.ToArray(),
            enumEntries = enumEntries.ToArray(),
            sources = new[]
            {
                "PlantType enum",
                "CardUI.theSeedCost",
                "CardUI.CD/fullCD/isAvailable/disabled",
                "registry/almanac enrichment in Python CLI"
            }
        };
    }

    private bool ComputeRawGameplayReady(Board? board)
    {
        if (board == null || FindCreatePlant() == null)
        {
            return false;
        }

        try
        {
            if (board.over)
            {
                return false;
            }

            var zombieCount = 0;
            try { zombieCount = board.zombieArray?.Count ?? 0; } catch { }
            var possibleWin = board.theMaxWave > 0 &&
                              board.theWave >= board.theMaxWave &&
                              zombieCount == 0 &&
                              !board.moreZombiesComing;
            if (possibleWin)
            {
                return false;
            }

            return board.startMove && _config.PlantTypes.All(plantType => GetCardCooldown(plantType).Found);
        }
        catch
        {
            return false;
        }
    }

    private SeedRuntimeSnapshot ResolveSeedStateForObservation(
        Board? board,
        bool rawGameplayReady,
        bool forceSeedProbe,
        out double seedProbeMs,
        out double uiScanMs)
    {
        seedProbeMs = 0.0;
        uiScanMs = 0.0;
        _observationsSinceSeedProbe++;

        var interval = Math.Max(0, _config.SeedScreenCheckInterval);
        var intervalDue = interval > 0 && _observationsSinceSeedProbe >= interval;
        var cacheNeedsRefresh = !_seedRuntimeCache.Valid ||
                                _seedRuntimeCache.SeedSelectionActive ||
                                _seedRuntimeCache.BlockingRewardUiActive ||
                                !_seedRuntimeCache.ActualGameplayReady;
        var shouldProbe = forceSeedProbe ||
                          !rawGameplayReady ||
                          cacheNeedsRefresh ||
                          intervalDue;

        if (shouldProbe)
        {
            var probe = BuildSeedProbe();
            seedProbeMs = probe.seed_probe_ms;
            uiScanMs = probe.ui_scan_ms;
            return SeedRuntimeSnapshot.FromProbe(probe);
        }

        RefreshCachedSeedSlotDtosFromCardRefs();
        if (!_seedRuntimeCache.Valid)
        {
            var probe = BuildSeedProbe();
            seedProbeMs = probe.seed_probe_ms;
            uiScanMs = probe.ui_scan_ms;
            return SeedRuntimeSnapshot.FromProbe(probe);
        }

        return _seedRuntimeCache.ToSnapshot(rawGameplayReady);
    }

    private void UpdateSeedRuntimeCache(SeedProbeDto probe)
    {
        var sortedSlotCards = SortSeedSlotCards(probe.ActiveGameplayCardBankCards);
        var activeGameplayCounts = BuildTypeCounts(sortedSlotCards.Select(card => card.PlantType));
        var requiredGameplayCounts = BuildTypeCounts(_config.PlantTypes);
        _seedRuntimeCache.Valid = true;
        _seedRuntimeCache.SeedSelectionActive = probe.SeedSelectionActive;
        _seedRuntimeCache.SeedSelectionPanelActive = probe.SeedSelectionPanelActive;
        _seedRuntimeCache.StartButtonActive = probe.StartButtonActive;
        _seedRuntimeCache.BlockingRewardUiActive = probe.BlockingRewardUiActive;
        _seedRuntimeCache.GameplayReady = probe.GameplayReady;
        _seedRuntimeCache.ActualGameplayReady = probe.GameplayReady &&
                                                !probe.SeedSelectionActive &&
                                                !probe.BlockingRewardUiActive &&
                                                CountsCover(activeGameplayCounts, requiredGameplayCounts);
        _seedRuntimeCache.ActiveGameplayCardBankCount = probe.ActiveGameplayCardBankCards.Count;
        _seedRuntimeCache.ActiveGameplayTypeCounts.Clear();
        foreach (var pair in activeGameplayCounts)
        {
            _seedRuntimeCache.ActiveGameplayTypeCounts[pair.Key] = pair.Value;
        }

        _seedRuntimeCache.CachedPlantCosts.Clear();
        foreach (var group in sortedSlotCards.Where(card => card.SeedCost > 0).GroupBy(card => card.PlantType))
        {
            _seedRuntimeCache.CachedPlantCosts[group.Key] = group.Min(card => card.SeedCost);
        }

        _seedRuntimeCache.CachedSeedSlotDtos.Clear();
        for (var i = 0; i < sortedSlotCards.Count; i++)
        {
            _seedRuntimeCache.CachedSeedSlotDtos.Add(BuildSeedSlotDto(sortedSlotCards[i], i, "active_gameplay_card_bank"));
        }

        RefreshCachedGameplayCardRefs(sortedSlotCards);
        _observationsSinceSeedProbe = 0;
    }

    private void InvalidateSeedRuntimeCache(string reason)
    {
        _seedRuntimeCache.Valid = false;
        _seedRuntimeCache.InvalidReason = reason;
        _seedRuntimeCache.ActiveGameplayTypeCounts.Clear();
        _seedRuntimeCache.CachedPlantCosts.Clear();
        _seedRuntimeCache.CachedGameplayCards.Clear();
        _seedRuntimeCache.CachedSeedSlots.Clear();
        _seedRuntimeCache.CachedSeedSlotDtos.Clear();
        _observationsSinceSeedProbe = Math.Max(_observationsSinceSeedProbe, _config.SeedScreenCheckInterval);
    }

    private void RefreshCachedGameplayCardRefs(List<SeedCardDto> sortedSlotCards)
    {
        _seedRuntimeCache.CachedGameplayCards.Clear();
        _seedRuntimeCache.CachedSeedSlots.Clear();
        var activeGameplayCardIds = new HashSet<int>(sortedSlotCards.Select(card => card.InstanceId));
        if (activeGameplayCardIds.Count == 0)
        {
            return;
        }

        var cardsById = new Dictionary<int, CardUI>();
        try
        {
            foreach (var card in Object.FindObjectsOfType<CardUI>())
            {
                try
                {
                    if (card == null || !activeGameplayCardIds.Contains(card.GetInstanceID()))
                    {
                        continue;
                    }

                    cardsById[card.GetInstanceID()] = card;
                    var plantType = (int)card.thePlantType;
                    if (!_seedRuntimeCache.CachedGameplayCards.ContainsKey(plantType))
                    {
                        _seedRuntimeCache.CachedGameplayCards[plantType] = card;
                    }
                }
                catch
                {
                    // Ignore stale CardUI wrappers while refreshing the cache.
                }
            }
        }
        catch
        {
            // Card UI is not available in every scene.
        }

        for (var i = 0; i < sortedSlotCards.Count; i++)
        {
            if (!cardsById.TryGetValue(sortedSlotCards[i].InstanceId, out var card))
            {
                continue;
            }

            _seedRuntimeCache.CachedSeedSlots.Add(new SeedSlotCacheEntry
            {
                SlotIndex = i,
                CardInstanceId = sortedSlotCards[i].InstanceId,
                Card = card
            });
        }
    }

    private void RefreshCachedSeedSlotDtosFromCardRefs()
    {
        if (!_seedRuntimeCache.Valid ||
            _seedRuntimeCache.SeedSelectionActive ||
            _seedRuntimeCache.CachedSeedSlots.Count == 0)
        {
            return;
        }

        var refreshed = new List<SeedSlotDto>();
        foreach (var entry in _seedRuntimeCache.CachedSeedSlots.OrderBy(slot => slot.SlotIndex))
        {
            try
            {
                var card = entry.Card;
                if (card == null ||
                    card.gameObject == null ||
                    !card.gameObject.activeInHierarchy ||
                    card.GetInstanceID() != entry.CardInstanceId)
                {
                    InvalidateSeedRuntimeCache("cached seed slot card stale");
                    return;
                }

                refreshed.Add(BuildSeedSlotDto(card, entry.SlotIndex, "cached_active_gameplay_card_live", includeHierarchyPath: false));
            }
            catch
            {
                InvalidateSeedRuntimeCache("cached seed slot refresh failed");
                return;
            }
        }

        if (refreshed.Count == 0)
        {
            return;
        }

        _seedRuntimeCache.CachedSeedSlotDtos.Clear();
        foreach (var slot in refreshed)
        {
            _seedRuntimeCache.CachedSeedSlotDtos.Add(slot);
        }
    }

    private void AddPerformanceTimings(ObservationDto obs, Stopwatch observeWatch, double seedProbeMs, double uiScanMs)
    {
        observeWatch.Stop();
        if (!_config.DebugPerformance)
        {
            return;
        }

        obs.observe_ms = Math.Round(observeWatch.Elapsed.TotalMilliseconds, 3);
        obs.bridge_observe_ms = obs.observe_ms;
        obs.seed_probe_ms = Math.Round(seedProbeMs, 3);
        obs.ui_scan_ms = Math.Round(uiScanMs, 3);
    }

    private void AddSpeedDiagnostics(ObservationDto obs, Board board)
    {
        obs.RequestedGameSpeed = _config.GameSpeed;
        obs.GameSpeedMode = _config.GameSpeedMode;
        obs.UnityTimeScale = SafeReadTimeScale();
        obs.FixedDeltaTime = SafeReadFixedDeltaTime();
        obs.EffectiveGameSpeed = ResolveEffectiveGameSpeed();
        obs.SpeedApplyCount = _speedApplyCount;
        obs.ValidSpeedModeApplyCount = _validSpeedModeApplyCount;
        obs.SunSpawnCompensationApplyCount = _sunSpawnCompensationApplyCount;
        obs.BridgeUpdateLoopCount = _bridgeUpdateLoopCount;
        obs.ResetCount = _resetCount;
        obs.LetsRockClickCount = _letsRockClickCount;
        try { obs.BoardInstanceId = board.GetInstanceID(); } catch { obs.BoardInstanceId = 0; }
        obs.NewZombieWaveCountDown = SafeReadFloat(() => board.newZombieWaveCountDown);
        obs.NextZombieWaveCountDown = SafeReadFloat(() => board.nextZombieWaveCountDown);
        obs.HugeWaveCountDown = SafeReadFloat(() => board.hugeWaveCountDown);

        if (TryReadFloatMember(board, out var skyInterval,
                "skySunSpawnInterval",
                "sunSpawnInterval",
                "sunDropInterval",
                "sunFallInterval"))
        {
            obs.SkySunSpawnInterval = skyInterval;
        }

        if (TryReadFloatMember(board, out var skyTimer,
                "skySunSpawnTimer",
                "sunSpawnTimer",
                "sunDropTimer",
                "sunCountdown",
                "nextSunSpawn"))
        {
            obs.SkySunSpawnTimer = skyTimer;
        }

        if (TryReadCollectionCount(board, out var sunCount,
                "sunArray",
                "sunList",
                "sunObjects",
                "suns",
                "sunObjectList",
                "sunArrayList",
                "fallingSunList",
                "fallingSuns"))
        {
            obs.ActiveFallingSunCount = sunCount;
            obs.ActiveSunObjectCount = sunCount;
        }

        if (TryReadFloatMember(board, out var sunPerMinute,
                "sunSpawnCountPerMinute",
                "sunSpawnRatePerMinute",
                "sunPerMinute"))
        {
            obs.SunSpawnCountPerMinute = sunPerMinute;
        }

        if (TryReadIntMember(board, out var collected,
                "sunCollectedCount",
                "collectedSunCount",
                "sunPickupCount",
                "sunPickedCount"))
        {
            obs.SunCollectedCount = collected;
        }

        if (_config.DebugSun)
        {
            RefreshSunDebugSnapshot(board);
            obs.ActiveBoardCount = _sunDebugSnapshot.ActiveBoardCount;
            obs.ActiveSkySunSpawnerCount = _sunDebugSnapshot.ActiveSkySunSpawnerCount;
            obs.ActiveSunObjectCount ??= _sunDebugSnapshot.ActiveSunObjectCount;
            obs.ActiveCoroutineCount = _sunDebugSnapshot.ActiveCoroutineCount;
        }
    }

    private void RefreshSunDebugSnapshot(Board board)
    {
        var frame = 0;
        try { frame = Time.frameCount; } catch { }
        if (_config.DebugSunSampleInterval > 0 && frame - _lastSunDebugFrame < _config.DebugSunSampleInterval)
        {
            return;
        }

        _lastSunDebugFrame = frame;
        var snapshot = new SunDebugSnapshot();
        snapshot.ActiveBoardCount = CountActiveBoards();
        snapshot.ActiveSkySunSpawnerCount = CountActiveSunSpawners(board);
        snapshot.ActiveSunObjectCount = ReadActiveSunObjectCount(board);
        snapshot.ActiveCoroutineCount = CountInvokingBehaviours();
        _sunDebugSnapshot = snapshot;
    }

    private static int? CountActiveBoards()
    {
        try { return Object.FindObjectsOfType<Board>().Length; }
        catch { return null; }
    }

    private static int? ReadActiveSunObjectCount(Board board)
    {
        if (TryReadCollectionCount(board, out var sunCount,
                "sunArray",
                "sunList",
                "sunObjects",
                "suns",
                "sunObjectList",
                "sunArrayList",
                "fallingSunList",
                "fallingSuns"))
        {
            return sunCount;
        }

        return CountLooseSunObjects();
    }

    private static int? CountActiveSunSpawners(Board board)
    {
        var candidates = new List<string>
        {
            "sunspawner",
            "sundropper",
            "sundrop",
            "skysun",
            "sunrain",
            "sunfall"
        };
        try
        {
            var count = 0;
            foreach (var behaviour in Object.FindObjectsOfType<MonoBehaviour>())
            {
                if (behaviour == null)
                {
                    continue;
                }

                var name = behaviour.GetType().Name?.ToLowerInvariant() ?? "";
                if (name.Contains("sunflower"))
                {
                    continue;
                }
                if (candidates.Any(fragment => name.Contains(fragment)))
                {
                    count++;
                }
            }
            if (count > 0)
            {
                return count;
            }

            return CountActiveBoards();
        }
        catch
        {
            return null;
        }
    }

    private static int? CountLooseSunObjects()
    {
        try
        {
            var ids = new HashSet<int>();
            foreach (var behaviour in Object.FindObjectsOfType<MonoBehaviour>())
            {
                if (behaviour == null)
                {
                    continue;
                }

                var gameObject = behaviour.gameObject;
                if (gameObject == null || !gameObject.activeInHierarchy)
                {
                    continue;
                }

                var typeName = behaviour.GetType().Name ?? "";
                var objectName = SafeObjectName(gameObject) ?? "";
                if (!LooksLikeLooseSunObject(typeName, objectName))
                {
                    continue;
                }

                ids.Add(gameObject.GetInstanceID());
            }

            return ids.Count;
        }
        catch
        {
            return null;
        }
    }

    private static bool LooksLikeLooseSunObject(string typeName, string objectName)
    {
        var normalized = (typeName + " " + objectName).ToLowerInvariant();
        if (!normalized.Contains("sun"))
        {
            return false;
        }

        return !(normalized.Contains("sunflower") ||
                 normalized.Contains("seed") ||
                 normalized.Contains("card") ||
                 normalized.Contains("bank") ||
                 normalized.Contains("button") ||
                 normalized.Contains("text") ||
                 normalized.Contains("cost"));
    }

    private static int? CountInvokingBehaviours()
    {
        try
        {
            var count = 0;
            foreach (var behaviour in Object.FindObjectsOfType<MonoBehaviour>())
            {
                if (behaviour == null)
                {
                    continue;
                }

                try
                {
                    if (behaviour.IsInvoking())
                    {
                        count++;
                    }
                }
                catch
                {
                    // Ignore invalid components.
                }
            }
            return count;
        }
        catch
        {
            return null;
        }
    }

    private static float SafeReadFloat(Func<float> reader)
    {
        try { return reader(); }
        catch { return 0f; }
    }

    private static bool TryReadFloatMember(object target, out float value, params string[] names)
    {
        if (TryReadNumberMember(target, out var number, names))
        {
            value = (float)number;
            return true;
        }

        value = 0f;
        return false;
    }

    private static bool TryReadIntMember(object target, out int value, params string[] names)
    {
        if (TryReadNumberMember(target, out var number, names))
        {
            value = (int)number;
            return true;
        }

        value = 0;
        return false;
    }

    private static bool TryReadNumberMember(object target, out double value, params string[] names)
    {
        foreach (var name in names)
        {
            if (!TryGetMemberValue(target, name, out var raw))
            {
                continue;
            }

            if (TryConvertNumber(raw, out value))
            {
                return true;
            }
        }

        value = 0.0;
        return false;
    }

    private static bool TryReadCollectionCount(object target, out int count, params string[] names)
    {
        foreach (var name in names)
        {
            if (!TryGetMemberValue(target, name, out var raw))
            {
                continue;
            }

            if (TryGetCollectionCount(raw, out count))
            {
                return true;
            }
        }

        count = 0;
        return false;
    }

    private static bool TryGetMemberValue(object target, string name, out object? value)
    {
        value = null;
        try
        {
            var flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;
            var type = target.GetType();
            var field = type.GetField(name, flags);
            if (field != null)
            {
                value = field.GetValue(target);
                return true;
            }

            var property = type.GetProperty(name, flags);
            if (property != null)
            {
                value = property.GetValue(target, null);
                return true;
            }
        }
        catch
        {
            return false;
        }

        return false;
    }

    private static bool TryConvertNumber(object? raw, out double value)
    {
        switch (raw)
        {
            case null:
                value = 0.0;
                return false;
            case int number:
                value = number;
                return true;
            case float number:
                value = number;
                return true;
            case double number:
                value = number;
                return true;
            case long number:
                value = number;
                return true;
            case short number:
                value = number;
                return true;
            case byte number:
                value = number;
                return true;
            case uint number:
                value = number;
                return true;
            case ulong number:
                value = number;
                return true;
        }

        try
        {
            value = Convert.ToDouble(raw);
            return true;
        }
        catch
        {
            value = 0.0;
            return false;
        }
    }

    private static bool TryGetCollectionCount(object? raw, out int count)
    {
        if (raw == null)
        {
            count = 0;
            return false;
        }

        switch (raw)
        {
            case Array array:
                count = array.Length;
                return true;
            case System.Collections.ICollection collection:
                count = collection.Count;
                return true;
        }

        try
        {
            var flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;
            var type = raw.GetType();
            var property = type.GetProperty("Count", flags) ?? type.GetProperty("Length", flags);
            if (property != null && TryConvertNumber(property.GetValue(raw, null), out var number))
            {
                count = (int)number;
                return true;
            }
        }
        catch
        {
            // Ignore reflection failures.
        }

        count = 0;
        return false;
    }

    private sealed class SunDebugSnapshot
    {
        public int? ActiveBoardCount { get; set; }
        public int? ActiveSkySunSpawnerCount { get; set; }
        public int? ActiveSunObjectCount { get; set; }
        public int? ActiveCoroutineCount { get; set; }
    }

    private object AutoSelectSeeds(JsonElement root)
    {
        var actions = new List<string>();
        var requested = ReadIntArray(root, "seed_types", "seedTypes", "plant_types", "plantTypes");
        if (requested.Count == 0)
        {
            requested.AddRange(_config.PlantTypes);
        }

        var board = FindBoard();
        var initBoard = FindInitBoard();
        var before = BuildSeedProbe();
        var success = true;
        var selected = new List<int>();
        var attempts = new List<SeedSelectionAttemptDto>();

        if (before.GameplayReady && CountsCover(
                BuildTypeCounts(before.ActiveGameplayCardBankCards.Select(card => card.PlantType)),
                BuildTypeCounts(requested)))
        {
            var alreadyVerified = VerifySeedMultiset(
                requested,
                before.ActiveGameplayCardBankCards.Select(card => card.PlantType),
                "activeGameplayCardBankCards");
            return new
            {
                ok = true,
                alreadyGameplayReady = true,
                requestedSeedTypes = requested.ToArray(),
                selectedSeedTypes = before.ActiveGameplayCardBankCards.Select(card => card.PlantType).ToArray(),
                missingSeedTypes = alreadyVerified.MissingSeedTypes,
                verification = alreadyVerified,
                gameplayReadyBeforeStart = true,
                startRequested = ReadBool(root, true, "start_level", "startLevel"),
                startInvoked = false,
                actions,
                selectionAttempts = attempts.ToArray(),
                before,
                afterSelectionBeforeStart = before,
                afterStart = before,
                after = before,
                message = "Requested seeds were already present in the active gameplay card bank."
            };
        }

        if (!IsSeedSelectionSafeForUiAction(before))
        {
            actions.Add("auto_select_seeds refused because seed-selection UI is not stable/safe for UI automation.");
            return new
            {
                ok = false,
                requestedSeedTypes = requested.ToArray(),
                selectedSeedTypes = Array.Empty<int>(),
                missingSeedTypes = requested.ToArray(),
                verification = VerifySeedMultiset(requested, Array.Empty<int>(), "strict_seed_screen_gate"),
                gameplayReadyBeforeStart = before.GameplayReady,
                startRequested = ReadBool(root, true, "start_level", "startLevel"),
                startInvoked = false,
                actions,
                selectionAttempts = attempts.ToArray(),
                before,
                afterSelectionBeforeStart = before,
                afterStart = before,
                after = before,
                message = "Seed automation is only allowed on a stable seed-selection screen, not during gameplay or transition states."
            };
        }

        var alreadySelectedCounts = BuildTypeCounts(before.SelectedSeedBankCards.Select(card => card.PlantType));
        var requestedToClick = new List<int>();
        foreach (var plantType in requested)
        {
            alreadySelectedCounts.TryGetValue(plantType, out var alreadySelectedCount);
            if (alreadySelectedCount > 0)
            {
                alreadySelectedCounts[plantType] = alreadySelectedCount - 1;
                selected.Add(plantType);
                actions.Add($"selected_bank_already_contains:{plantType}");
            }
            else
            {
                requestedToClick.Add(plantType);
            }
        }

        foreach (var plantType in requestedToClick)
        {
            var attempt = TrySelectVisibleSeedCard(plantType, attempts.Count, selected.Count(value => value == plantType), actions);
            attempts.Add(attempt);
            if (!attempt.Success)
            {
                success = false;
            }
            else
            {
                selected.Add(plantType);
            }
        }

        var afterSelectionBeforeStart = BuildSeedProbe();
        var verified = VerifySeedMultiset(
            requested,
            afterSelectionBeforeStart.SelectedSeedBankCards.Select(card => card.PlantType),
            "selectedSeedBankCards");
        if (!verified.Success)
        {
            success = false;
        }

        var gameplayReadyBeforeStart = afterSelectionBeforeStart.GameplayReady;
        var startInvoked = false;
        var startRequested = ReadBool(root, true, "start_level", "startLevel");
        if (success && startRequested && (afterSelectionBeforeStart.SeedSelectionActive || !gameplayReadyBeforeStart))
        {
            startInvoked = TryStartSelectedLevel(initBoard, actions);
            if (!startInvoked && BuildSeedProbe().SeedSelectionActive)
            {
                success = false;
                actions.Add("start_failed_while_seed_selection_active");
            }
        }

        var afterStart = BuildSeedProbe();
        return new
        {
            ok = success,
            requestedSeedTypes = requested.ToArray(),
            selectedSeedTypes = selected.ToArray(),
            missingSeedTypes = verified.MissingSeedTypes,
            verification = verified,
            gameplayReadyBeforeStart,
            startRequested,
            startInvoked,
            actions,
            selectionAttempts = attempts.ToArray(),
            before,
            afterSelectionBeforeStart,
            afterStart,
            after = afterStart,
            message = success
                ? "Requested seeds selected/verified. Python should wait for gameplayReady."
                : "Seed selection failed; do not start training."
        };
    }

    private object SelectSeedCardOnce(JsonElement root)
    {
        var actions = new List<string>();
        var plantType = ReadInt(root, _config.PlantTypes.Count > 0 ? _config.PlantTypes[0] : 1, "plant_type", "plantType", "seed_type", "seedType");
        var attemptIndex = ReadInt(root, 0, "attempt_index", "attemptIndex");
        var duplicateSelectionIndex = ReadInt(root, 0, "duplicate_selection_index", "duplicateSelectionIndex");
        var before = BuildSeedProbe();
        SeedSelectionAttemptDto attempt;
        if (!IsSeedSelectionSafeForUiAction(before))
        {
            attempt = new SeedSelectionAttemptDto
            {
                AttemptIndex = attemptIndex,
                PlantType = plantType,
                PlantTypeName = ((PlantType)plantType).ToString(),
                MethodUsed = "strict_seed_screen_gate",
                DuplicateSelectionIndex = duplicateSelectionIndex,
                SelectedBankCountBefore = before.SelectedBankVisibleCount,
                SelectedBankTypeCountBefore = before.SelectedSeedBankCards.Count(card => card.PlantType == plantType),
                Error = "Seed-selection UI is not stable/safe; refusing to click stale, gameplay, or transition UI."
            };
            actions.Add($"select_seed_card_once_refused_seed_ui_unstable:{plantType}");
        }
        else
        {
            attempt = TrySelectVisibleSeedCard(plantType, attemptIndex, duplicateSelectionIndex, actions);
        }

        var after = BuildSeedProbe();

        return new
        {
            ok = attempt.ClickInvoked,
            clickInvoked = attempt.ClickInvoked,
            plantType,
            plantTypeName = ((PlantType)plantType).ToString(),
            attempt,
            actions,
            before,
            after,
            message = attempt.ClickInvoked
                ? "Clicked one visible chooser seed card. Caller must delay and verify selected-bank state."
                : "No seed card click was invoked."
        };
    }

    private object PressLetsRockOnce(JsonElement root)
    {
        var actions = new List<string>();
        var before = BuildSeedProbe();
        GameObject? startObject = null;
        var methodUsed = "";
        var startClicked = false;

        if (!IsSeedSelectionSafeForUiAction(before))
        {
            actions.Add("Start press refused because seed-selection UI is not stable/safe for UI automation.");
        }
        else
        {
            startObject = FindExactStartButtonObject();
        }

        if (before.SeedSelectionActive && before.StartButtonActive && startObject == null)
        {
            actions.Add("Exact SeedLibrary/Start object not found.");
        }
        else if (startObject != null)
        {
            startClicked = TryPressExactStartButtonOnce(startObject, actions, out methodUsed);
        }

        if (startClicked)
        {
            _letsRockClickCount++;
        }

        var after = BuildSeedProbe();
        if (startClicked && !after.GameplayReady)
        {
            actions.Add("Start clicked; waiting for the game's normal seed-screen transition without InitBoard fallback.");
        }

        if (!startClicked && IsSeedSelectionSafeForUiAction(before))
        {
            var initBoard = FindInitBoard();
            startClicked = TryStartSelectedLevel(initBoard, actions);
            if (startClicked)
            {
                methodUsed = "TryStartSelectedLevel";
                _letsRockClickCount++;
                after = BuildSeedProbe();
            }
        }

        return new
        {
            ok = startClicked,
            startClicked,
            methodUsed,
            startObjectName = startObject != null ? SafeObjectName(startObject) : null,
            startHierarchyPath = startObject != null ? BuildHierarchyPath(startObject.transform) : null,
            actions,
            before,
            after,
            message = startClicked
                ? "Issued one Start/Let's Rock press. Caller must delay and verify gameplay state."
                : "Start/Let's Rock press was not invoked."
        };
    }

    private List<SeedCardDto> ScanSeedCards()
    {
        var cards = new List<SeedCardDto>();
        try
        {
            foreach (var card in Object.FindObjectsOfType<CardUI>())
            {
                try
                {
                    if (card == null)
                    {
                        continue;
                    }

                    cards.Add(BuildSeedCardDto(card));
                }
                catch
                {
                    // Ignore stale CardUI wrappers.
                }
            }
        }
        catch
        {
            // Card UI is not available in every scene.
        }

        return cards;
    }

    private SeedCardDto BuildSeedCardDto(CardUI card)
    {
        var transform = card.transform;
        var position = transform.position;
        var localPosition = transform.localPosition;
        var active = false;
        try { active = card.gameObject != null && card.gameObject.activeInHierarchy; } catch { }
        var rendererVisible = HasVisibleRenderer(card.gameObject);
        var screenX = 0f;
        var screenY = 0f;
        var screenZ = 0f;
        var inScreenBounds = false;
        var inScreenSpaceBounds = false;
        var cameraFound = false;
        try
        {
            var camera = Camera.main;
            if (camera != null)
            {
                cameraFound = true;
                var screen = camera.WorldToScreenPoint(position);
                screenX = screen.x;
                screenY = screen.y;
                screenZ = screen.z;
                inScreenBounds = screen.z > 0f &&
                                 screen.x >= -32f && screen.x <= Screen.width + 32f &&
                                 screen.y >= -32f && screen.y <= Screen.height + 32f;
            }
        }
        catch { }
        try
        {
            inScreenSpaceBounds = position.x >= -64f && position.x <= Screen.width + 64f &&
                                  position.y >= -64f && position.y <= Screen.height + 64f;
        }
        catch { }

        var hasVisibleUiComponent = HasVisibleUiComponent(card.gameObject);
        var hierarchyPath = BuildHierarchyPath(transform);
        var namePathSuggestsVisibleSeedUi = TextSuggestsSeedChooser($"{SafeObjectName(card.gameObject)}/{SafeObjectName(SafeReadCardParent(card))}/{hierarchyPath}");
        var uiVisible = active && (rendererVisible || hasVisibleUiComponent || inScreenBounds || inScreenSpaceBounds || !cameraFound || namePathSuggestsVisibleSeedUi);
        return new SeedCardDto
        {
            InstanceId = card.GetInstanceID(),
            Active = active,
            UiVisible = uiVisible,
            RendererVisible = rendererVisible,
            HasVisibleUiComponent = hasVisibleUiComponent,
            InScreenBounds = inScreenBounds,
            InScreenSpaceBounds = inScreenSpaceBounds,
            ScreenX = screenX,
            ScreenY = screenY,
            ScreenZ = screenZ,
            PlantType = (int)card.thePlantType,
            PlantTypeName = card.thePlantType.ToString(),
            DisplayName = card.thePlantType.ToString(),
            SeedCost = card.theSeedCost,
            OnCardBank = card.onCardBank,
            IsSelected = card.isSelected,
            PreSelected = card.preSelected,
            IsAvailable = card.isAvailable,
            Disabled = card.disabled,
            Selectable = active && !card.disabled,
            Cd = card.CD,
            FullCd = card.fullCD,
            X = position.x,
            Y = position.y,
            LocalX = localPosition.x,
            LocalY = localPosition.y,
            GameObjectName = SafeObjectName(card.gameObject),
            ParentName = transform.parent != null ? SafeObjectName(transform.parent.gameObject) : null,
            CardParentName = SafeObjectName(SafeReadCardParent(card)),
            RootName = transform.root != null ? SafeObjectName(transform.root.gameObject) : null,
            HierarchyPath = hierarchyPath,
            Text = SafeReadText(card.text),
            TextBg = SafeReadText(card.textBg),
            MatchedRegistryKey = card.thePlantType.ToString()
        };
    }

    private bool IsSeedSelectionLikelyActive(Board? board, List<SeedCardDto> cards)
    {
        if (cards.Count == 0)
        {
            return false;
        }

        try
        {
            var gameplayReady = ComputeRawGameplayReady(board);
            return IsSeedSelectionPanelActive(FindInitBoard(), board) ||
                   (!gameplayReady && !(board?.startMove ?? false) && cards.Any(card => card.UiVisible && !card.OnCardBank));
        }
        catch
        {
            return cards.Any(card => card.UiVisible && !card.OnCardBank);
        }
    }

    private List<UiProbeEntryDto> BuildSeedChooserSignals()
    {
        return BuildUiProbeEntries(includeAll: false, maxEntries: 500)
            .Where(entry => IsSeedChooserSignal(entry.Name, entry.HierarchyPath, entry.Text, entry.ClassName))
            .ToList();
    }

    private static bool IsVisibleUiSignal(UiProbeEntryDto entry) =>
        entry.UiVisible && entry.InScreenBounds;

    private static bool IsVisibleOnScreenCard(SeedCardDto card) =>
        card.UiVisible && card.InScreenBounds;

    private List<UiProbeEntryDto> BuildUiProbeEntries(bool includeAll, int maxEntries)
    {
        var entries = new List<UiProbeEntryDto>();
        try
        {
            foreach (var transform in Object.FindObjectsOfType<Transform>())
            {
                try
                {
                    if (transform == null || transform.gameObject == null)
                    {
                        continue;
                    }

                    var gameObject = transform.gameObject;
                    var activeSelf = false;
                    var activeInHierarchy = false;
                    try { activeSelf = gameObject.activeSelf; } catch { }
                    try { activeInHierarchy = gameObject.activeInHierarchy; } catch { }
                    if (!activeInHierarchy)
                    {
                        continue;
                    }

                    var path = BuildHierarchyPath(transform);
                    var componentTypes = SafeComponentTypeNames(gameObject);
                    var text = SafeReadGameObjectText(gameObject);
                    var position = transform.position;
                    var localPosition = transform.localPosition;
                    var screen = WorldToScreen(position, out var inWorldScreenBounds);
                    var inScreenSpaceBounds = false;
                    try
                    {
                        inScreenSpaceBounds = position.x >= -64f && position.x <= Screen.width + 64f &&
                                              position.y >= -64f && position.y <= Screen.height + 64f;
                    }
                    catch { }

                    var rendererVisible = HasVisibleRenderer(gameObject);
                    var uiVisible = HasVisibleUiComponent(gameObject);
                    var className = string.Join(",", componentTypes);
                    var isUiLike = includeAll ||
                                   uiVisible ||
                                   rendererVisible ||
                                   path.IndexOf("Canvas", StringComparison.OrdinalIgnoreCase) >= 0 ||
                                   componentTypes.Any(type => IsUiTypeName(type)) ||
                                   IsSeedChooserSignal(SafeObjectName(gameObject), path, text, className);
                    if (!isUiLike)
                    {
                        continue;
                    }

                    entries.Add(new UiProbeEntryDto
                    {
                        Name = SafeObjectName(gameObject),
                        ClassName = className,
                        Text = text,
                        ActiveSelf = activeSelf,
                        ActiveInHierarchy = activeInHierarchy,
                        RendererVisible = rendererVisible,
                        UiVisible = uiVisible || rendererVisible || inWorldScreenBounds || inScreenSpaceBounds,
                        ParentName = transform.parent != null ? SafeObjectName(transform.parent.gameObject) : null,
                        RootName = transform.root != null ? SafeObjectName(transform.root.gameObject) : null,
                        HierarchyPath = path,
                        X = position.x,
                        Y = position.y,
                        Z = position.z,
                        LocalX = localPosition.x,
                        LocalY = localPosition.y,
                        ScreenX = screen.x,
                        ScreenY = screen.y,
                        ScreenZ = screen.z,
                        InScreenBounds = inWorldScreenBounds
                    });

                }
                catch
                {
                    // Continue scanning other UI objects.
                }
            }
        }
        catch
        {
            // UI is not available in every scene.
        }

        return entries
            .OrderByDescending(entry => IsSeedChooserSignal(entry.Name, entry.HierarchyPath, entry.Text, entry.ClassName) ? 1 : 0)
            .ThenBy(entry => entry.HierarchyPath)
            .Take(maxEntries)
            .ToList();
    }

    private static bool IsUiTypeName(string typeName)
    {
        return typeName.IndexOf("UI", StringComparison.OrdinalIgnoreCase) >= 0 ||
               typeName.IndexOf("Button", StringComparison.OrdinalIgnoreCase) >= 0 ||
               typeName.IndexOf("Text", StringComparison.OrdinalIgnoreCase) >= 0 ||
               typeName.IndexOf("CardUI", StringComparison.OrdinalIgnoreCase) >= 0 ||
               typeName.IndexOf("SelectYourPlants", StringComparison.OrdinalIgnoreCase) >= 0 ||
               typeName.IndexOf("Canvas", StringComparison.OrdinalIgnoreCase) >= 0;
    }

    private static bool IsSeedChooserSignal(UiProbeEntryDto entry) =>
        IsSeedChooserSignal(entry.Name, entry.HierarchyPath, entry.Text, entry.ClassName);

    private static bool IsSeedChooserSignal(string? name, string? path, string? text, string? className)
    {
        if (IsChooserRootPath(path) ||
            IsChooserPacketPath(path) ||
            IsSelectedSeedBankPath(path) ||
            IsSeedBankSeedGroupPath(path) ||
            IsStartButtonPath(path))
        {
            return true;
        }

        var normalized = NormalizeUiText($"{name} {path} {text} {className}");
        if (string.IsNullOrEmpty(normalized))
        {
            return false;
        }

        var keywords = new[]
        {
            "chooseyourplants",
            "selectyourplants",
            "letsrock",
            "letrock",
            "viewlawn",
            "openalmanac",
            "selectpreviousloadout",
            "previousloadout",
            "uniqueplants",
            "commonplants",
            "seedbank",
            "cardui",
            "seedchooser",
            "almanac",
            "trophy",
            "reward",
            "prize",
            "award",
            "levelcomplete",
            "levelcompleted",
            "newplant"
        };

        if (keywords.Any(normalized.Contains))
        {
            return true;
        }

        return normalized.Contains("let") && normalized.Contains("rock");
    }

    private static bool IsBlockingRewardUiSignal(UiProbeEntryDto entry)
    {
        var normalized = NormalizeUiText($"{entry.Name} {entry.HierarchyPath} {entry.Text} {entry.ClassName}");
        if (IsGameplayGiftDropSignal(normalized))
        {
            return false;
        }
        return normalized.Contains("trophy") ||
               normalized.Contains("reward") ||
               normalized.Contains("prize") ||
               normalized.Contains("award") ||
               normalized.Contains("levelcomplete") ||
               normalized.Contains("levelcompleted") ||
               normalized.Contains("newplant");
    }

    private static bool IsGameplayGiftDropSignal(string normalized)
    {
        if (string.IsNullOrWhiteSpace(normalized))
        {
            return false;
        }

        var looksLikeGiftDrop =
            normalized.Contains("gift") ||
            normalized.Contains("present") ||
            normalized.Contains("giftbox") ||
            normalized.Contains("dropbox") ||
            normalized.Contains("rewardbox") ||
            normalized.Contains("prizebox");
        if (!looksLikeGiftDrop)
        {
            return false;
        }

        var looksLikePostGameUi =
            normalized.Contains("trophy") ||
            normalized.Contains("levelcomplete") ||
            normalized.Contains("levelcompleted") ||
            normalized.Contains("levelcleared") ||
            normalized.Contains("newplant") ||
            normalized.Contains("seedpacket") ||
            normalized.Contains("plantcard") ||
            normalized.Contains("almanac") ||
            normalized.Contains("continue") ||
            normalized.Contains("confirm");
        return !looksLikePostGameUi;
    }

    private static bool TextSuggestsSeedChooser(string? value) =>
        IsSeedChooserSignal(value, value, value, value);

    private static bool IsPanelOrTitleSignal(UiProbeEntryDto entry)
    {
        if (IsChooserPanelPath(entry.HierarchyPath) || IsChooserRootPath(entry.HierarchyPath))
        {
            return true;
        }

        var normalized = NormalizeUiText($"{entry.Name} {entry.HierarchyPath} {entry.Text} {entry.ClassName}");
        return normalized.Contains("chooseyourplants") ||
               normalized.Contains("selectyourplants") ||
               normalized.Contains("seedchooser") ||
               normalized.Contains("viewlawn") ||
               normalized.Contains("selectpreviousloadout") ||
               normalized.Contains("uniqueplants") ||
               normalized.Contains("commonplants");
    }

    private static bool IsChooseYourPlantsSignal(UiProbeEntryDto entry)
    {
        if (IsChooserPanelPath(entry.HierarchyPath))
        {
            return true;
        }

        var normalized = NormalizeUiText($"{entry.Name} {entry.HierarchyPath} {entry.Text} {entry.ClassName}");
        return normalized.Contains("chooseyourplants") || normalized.Contains("selectyourplants");
    }

    private static bool IsLetsRockSignal(UiProbeEntryDto entry)
    {
        if (IsStartButtonPath(entry.HierarchyPath))
        {
            return true;
        }

        var normalized = NormalizeUiText($"{entry.Name} {entry.HierarchyPath} {entry.Text} {entry.ClassName}");
        return normalized.Contains("letsrock") ||
               normalized.Contains("letrock") ||
               (normalized.Contains("let") && normalized.Contains("rock")) ||
               normalized.Contains("readysetplant");
    }

    private static string NormalizeUiText(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return "";
        }

        var builder = new StringBuilder(value.Length);
        foreach (var ch in value.ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(ch))
            {
                builder.Append(ch);
            }
        }

        return builder.ToString();
    }

    private static bool IsChooserRootPath(string? path)
    {
        var normalized = NormalizeUiText(path);
        return normalized.Contains("canvasupingameuiclonebottomseedlibraryselectyourplants") ||
               normalized.Contains("canvasupingameuiclonebottomseedlibrarygridpagespage") ||
               normalized.Contains("canvasupingameuiclonebottomseedlibrarylookalmanac") ||
               normalized.Contains("canvasupingameuiclonebottomseedlibraryshowlawn") ||
               normalized.Contains("canvasupingameuiclonebottomseedlibraryreselectcard") ||
               normalized.Contains("canvasupingameuiclonebottomseedlibraryshownormal") ||
               normalized.Contains("canvasupingameuiclonebottomseedlibrarysnowcolorful") ||
               IsStartButtonPath(path);
    }

    private static bool IsChooserPanelPath(string? path)
    {
        return NormalizeUiText(path).Contains("canvasupingameuiclonebottomseedlibraryselectyourplants");
    }

    private static bool IsChooserPacketPath(string? path)
    {
        var normalized = NormalizeUiText(path);
        return normalized.Contains("canvasupingameuiclonebottomseedlibrarygridpagespage") &&
               normalized.EndsWith("packet", StringComparison.Ordinal);
    }

    private static bool IsSelectedSeedBankPath(string? path)
    {
        var normalized = NormalizeUiText(path);
        return normalized.Contains("canvasupingameuicloneseedbankseedgroupseed") &&
               !normalized.Contains("bottomseedlibrary");
    }

    private static bool IsSeedBankSeedGroupPath(string? path)
    {
        var normalized = NormalizeUiText(path);
        return normalized.Contains("canvasupingameuicloneseedbankseedgroup") &&
               !normalized.Contains("bottomseedlibrary");
    }

    private static bool IsStartButtonPath(string? path)
    {
        var normalized = NormalizeUiText(path);
        return normalized.Contains("canvasupingameuiclonebottomseedlibrarystart") &&
               !normalized.Contains("restart");
    }

    private static bool IsExactStartButtonObject(GameObject? gameObject)
    {
        if (gameObject == null)
        {
            return false;
        }

        var normalized = NormalizeUiText(BuildHierarchyPath(gameObject.transform));
        return normalized.EndsWith("canvasupingameuiclonebottomseedlibrarystart", StringComparison.Ordinal);
    }

    private bool IsSeedSelectionPanelActive(InitBoard? initBoard, Board? board)
    {
        return IsChooseYourPlantsTextActive();
    }

    private bool IsChooseYourPlantsTextActive()
    {
        try
        {
            foreach (var chooser in Object.FindObjectsOfType<SelectYourPlants>())
            {
                try
                {
                    if (chooser == null || chooser.gameObject == null || !chooser.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    WorldToScreen(chooser.transform.position, out var inBounds);
                    if (inBounds)
                    {
                        return true;
                    }
                }
                catch { }
            }
        }
        catch { }

        return false;
    }

    private bool IsStartButtonActive()
    {
        try
        {
            foreach (var transform in Object.FindObjectsOfType<Transform>())
            {
                try
                {
                    if (transform == null || transform.gameObject == null || !transform.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    var path = BuildHierarchyPath(transform);
                    var name = (SafeObjectName(transform.gameObject) ?? "").ToLowerInvariant();
                    var normalizedPath = path.ToLowerInvariant();
                    WorldToScreen(transform.position, out var inBounds);
                    if (!inBounds)
                    {
                        continue;
                    }

                    if (IsStartButtonPath(path) ||
                        ((name.Contains("start") || name.Contains("ready") || name.Contains("rock") || name.Contains("lets")) &&
                         (normalizedPath.Contains("seed") || normalizedPath.Contains("select") || normalizedPath.Contains("choose") || normalizedPath.Contains("init"))))
                    {
                        return true;
                    }
                }
                catch { }
            }
        }
        catch { }

        return false;
    }

    private HashSet<int> ReadInGameCardBankIds()
    {
        var result = new HashSet<int>();
        try
        {
            foreach (var ui in Object.FindObjectsOfType<InGameUI>())
            {
                try
                {
                    if (ui == null || ui.cardOnBank == null)
                    {
                        continue;
                    }

                    foreach (var card in ui.cardOnBank)
                    {
                        try
                        {
                            if (card != null)
                            {
                                result.Add(card.GetInstanceID());
                            }
                        }
                        catch { }
                    }
                }
                catch { }
            }
        }
        catch { }

        return result;
    }

    private static bool NameSuggestsSelectedBank(SeedCardDto card)
    {
        if (IsSelectedSeedBankPath(card.HierarchyPath))
        {
            return true;
        }

        var text = NormalizeUiText($"{card.GameObjectName}/{card.ParentName}/{card.CardParentName}/{card.RootName}/{card.HierarchyPath}/{card.Text}/{card.TextBg}");
        return (text.Contains("selectedbank") ||
                text.Contains("selectedseed") ||
                text.Contains("cardbank") ||
                text.Contains("seedbank") ||
                text.Contains("bank")) &&
               !text.Contains("almanac") &&
               !NameSuggestsAvailableChooser(card);
    }

    private static bool NameSuggestsAvailableChooser(SeedCardDto card)
    {
        if (IsChooserPacketPath(card.HierarchyPath))
        {
            return true;
        }

        var text = NormalizeUiText($"{card.GameObjectName}/{card.ParentName}/{card.CardParentName}/{card.RootName}/{card.HierarchyPath}/{card.Text}/{card.TextBg}");
        return text.Contains("available") ||
               text.Contains("cardlist") ||
               text.Contains("cardpool") ||
               text.Contains("choosergrid") ||
               text.Contains("choosegrid") ||
               text.Contains("commonplants") ||
               text.Contains("uniqueplants") ||
               text.Contains("allplants") ||
               text.Contains("plantlist") ||
               text.Contains("almanac");
    }

    private static bool NameSuggestsGameplayBank(SeedCardDto card)
    {
        if (IsChooserPacketPath(card.HierarchyPath) || IsSelectedSeedBankPath(card.HierarchyPath))
        {
            return false;
        }

        var text = NormalizeUiText($"{card.GameObjectName}/{card.ParentName}/{card.CardParentName}/{card.RootName}/{card.HierarchyPath}");
        return text.Contains("ingame") || text.Contains("seedbank") || text.Contains("cardbank");
    }

    private SeedSelectionAttemptDto TrySelectVisibleSeedCard(
        int plantType,
        int attemptIndex,
        int sameTypeSelectionsBefore,
        List<string> actions)
    {
        var probeBefore = BuildSeedProbe();
        var beforeTypeCount = probeBefore.SelectedSeedBankCards.Count(card => card.PlantType == plantType);
        var beforeBankCount = probeBefore.SelectedBankVisibleCount;
        if (!IsSeedSelectionSafeForUiAction(probeBefore))
        {
            actions.Add($"seed_card_click_refused_seed_ui_unstable:{plantType}");
            return new SeedSelectionAttemptDto
            {
                AttemptIndex = attemptIndex,
                PlantType = plantType,
                PlantTypeName = ((PlantType)plantType).ToString(),
                MethodUsed = "strict_seed_screen_gate",
                SelectedBankCountBefore = beforeBankCount,
                SelectedBankTypeCountBefore = beforeTypeCount,
                DuplicateSelectionIndex = sameTypeSelectionsBefore,
                Error = "Seed-selection UI is not stable/safe immediately before seed card click.",
                Success = false
            };
        }

        var candidateDto = probeBefore.AvailableSeedCards
            .Where(card => card.PlantType == plantType)
            .OrderBy(card => card.Disabled ? 1 : 0)
            .ThenBy(card => IsChooserPacketPath(card.HierarchyPath) ? 0 : 1)
            .ThenBy(card => card.SeedCost)
            .ThenBy(card => card.HierarchyPath)
            .ThenBy(card => card.Y)
            .ThenBy(card => card.X)
            .FirstOrDefault() ??
            probeBefore.RuntimeCardWrappers
                .Where(card => probeBefore.SeedSelectionActive &&
                               IsVisibleOnScreenCard(card) &&
                               IsChooserPacketPath(card.HierarchyPath) &&
                               card.PlantType == plantType &&
                               !card.Disabled)
                .OrderBy(card => card.SeedCost)
                .ThenBy(card => card.HierarchyPath)
                .ThenBy(card => card.Y)
                .ThenBy(card => card.X)
                .FirstOrDefault();

        var attempt = new SeedSelectionAttemptDto
        {
            AttemptIndex = attemptIndex,
            PlantType = plantType,
            PlantTypeName = ((PlantType)plantType).ToString(),
            MethodUsed = "CardUI.OnMouseDown()",
            SelectedBankCountBefore = beforeBankCount,
            SelectedBankTypeCountBefore = beforeTypeCount,
            DuplicateSelectionIndex = sameTypeSelectionsBefore,
            VisibleCostsBefore = CostsForPlant(probeBefore.RuntimeCardWrappers, plantType).ToArray(),
            AvailableCostsBefore = CostsForPlant(probeBefore.AvailableSeedCards, plantType).ToArray(),
            SelectedBankCostsBefore = CostsForPlant(probeBefore.SelectedSeedBankCards, plantType).ToArray(),
            CandidateInstanceId = candidateDto?.InstanceId ?? 0,
            CandidateCostBefore = candidateDto?.SeedCost ?? 0,
            CandidateParentName = candidateDto?.ParentName,
            CandidateContainerName = candidateDto?.CardParentName ?? candidateDto?.ParentName,
            CandidateHierarchyPath = candidateDto?.HierarchyPath
        };

        if (candidateDto == null)
        {
            attempt.Error = "No visible chooser seed card matched the requested plant type.";
            actions.Add($"missing_visible_chooser_card:{plantType}");
            return attempt;
        }

        var card = FindCardByInstanceId(candidateDto.InstanceId);
        if (card == null)
        {
            attempt.Error = "Visible candidate disappeared before selection.";
            actions.Add($"visible_candidate_disappeared:{plantType}:{candidateDto.InstanceId}");
            return attempt;
        }

        try
        {
            card.OnMouseDown();
            attempt.ClickInvoked = true;
            actions.Add($"CardUI.OnMouseDown({candidateDto.InstanceId}, plantType={plantType})");
        }
        catch (Exception ex)
        {
            attempt.Error = "CardUI.OnMouseDown failed: " + ex.Message;
            actions.Add($"CardUI.OnMouseDown({candidateDto.InstanceId}) failed: {ex.Message}");
        }

        var probeAfter = BuildSeedProbe();
        attempt.SelectedBankCountAfter = probeAfter.SelectedBankVisibleCount;
        attempt.SelectedBankTypeCountAfter = probeAfter.SelectedSeedBankCards.Count(cardDto => cardDto.PlantType == plantType);
        attempt.VisibleCostsAfter = CostsForPlant(probeAfter.RuntimeCardWrappers, plantType).ToArray();
        attempt.AvailableCostsAfter = CostsForPlant(probeAfter.AvailableSeedCards, plantType).ToArray();
        attempt.SelectedBankCostsAfter = CostsForPlant(probeAfter.SelectedSeedBankCards, plantType).ToArray();
        var afterCandidate = probeAfter.RuntimeCardWrappers.FirstOrDefault(cardDto => cardDto.InstanceId == candidateDto.InstanceId);
        attempt.CandidateCostAfter = afterCandidate?.SeedCost ?? 0;
        attempt.SelectedBankCountIncreased = attempt.SelectedBankCountAfter == attempt.SelectedBankCountBefore + 1;
        attempt.SelectedBankTypeCountIncreased = attempt.SelectedBankTypeCountAfter == attempt.SelectedBankTypeCountBefore + 1;
        attempt.DuplicateCostIncreaseAccessible = sameTypeSelectionsBefore > 0 &&
                                                  attempt.VisibleCostsBefore.Length > 0 &&
                                                  attempt.VisibleCostsAfter.Length > 0;
        attempt.DuplicateCostIncreaseDetected = attempt.DuplicateCostIncreaseAccessible &&
                                                attempt.VisibleCostsAfter.Max() > attempt.VisibleCostsBefore.Max();
        attempt.Success = attempt.ClickInvoked &&
                          attempt.SelectedBankCountIncreased &&
                          attempt.SelectedBankTypeCountIncreased;
        if (!attempt.Success && string.IsNullOrEmpty(attempt.Error))
        {
            attempt.Error = "Selection did not add exactly one visible selected-bank card.";
        }

        return attempt;
    }

    private CardUI? FindCardByInstanceId(int instanceId)
    {
        try
        {
            foreach (var card in Object.FindObjectsOfType<CardUI>())
            {
                try
                {
                    if (card != null && card.GetInstanceID() == instanceId)
                    {
                        return card;
                    }
                }
                catch
                {
                    // Ignore stale CardUI wrappers.
                }
            }
        }
        catch
        {
            return null;
        }

        return null;
    }

    private static List<int> CostsForPlant(IEnumerable<SeedCardDto> cards, int plantType)
    {
        return cards
            .Where(card => card.PlantType == plantType && card.SeedCost > 0)
            .Select(card => card.SeedCost)
            .OrderBy(value => value)
            .ToList();
    }

    private static bool CountsCover(Dictionary<int, int> actual, Dictionary<int, int> expected)
    {
        foreach (var pair in expected)
        {
            actual.TryGetValue(pair.Key, out var actualCount);
            if (actualCount < pair.Value)
            {
                return false;
            }
        }

        return true;
    }

    private static Dictionary<int, int> BuildTypeCounts(IEnumerable<int> plantTypes)
    {
        var counts = new Dictionary<int, int>();
        foreach (var plantType in plantTypes)
        {
            counts.TryGetValue(plantType, out var current);
            counts[plantType] = current + 1;
        }

        return counts;
    }

    private static List<PlantTypeCountDto> BuildPlantTypeCounts(IEnumerable<int> plantTypes)
    {
        return BuildTypeCounts(plantTypes)
            .OrderBy(pair => pair.Key)
            .Select(pair => new PlantTypeCountDto
            {
                PlantType = pair.Key,
                PlantTypeName = ((PlantType)pair.Key).ToString(),
                Count = pair.Value
            })
            .ToList();
    }

    private static bool HasVisibleRenderer(GameObject? gameObject)
    {
        if (gameObject == null)
        {
            return false;
        }

        try
        {
            foreach (var renderer in gameObject.GetComponentsInChildren<Renderer>(true))
            {
                try
                {
                    if (renderer != null && renderer.enabled && renderer.gameObject.activeInHierarchy && renderer.isVisible)
                    {
                        return true;
                    }
                }
                catch { }
            }
        }
        catch { }

        return false;
    }

    private static bool HasVisibleUiComponent(GameObject? gameObject)
    {
        if (gameObject == null)
        {
            return false;
        }

        try
        {
            if (gameObject.GetComponent<RectTransform>() != null)
            {
                return true;
            }
        }
        catch { }

        try
        {
            if (gameObject.GetComponent<CanvasRenderer>() != null)
            {
                return true;
            }
        }
        catch { }

        try
        {
            return SafeComponentTypeNames(gameObject).Any(IsUiTypeName);
        }
        catch
        {
            return false;
        }
    }

    private static List<string> SafeComponentTypeNames(GameObject? gameObject)
    {
        var result = new List<string>();
        if (gameObject == null)
        {
            return result;
        }

        try
        {
            foreach (var component in gameObject.GetComponents<Component>())
            {
                try
                {
                    if (component == null)
                    {
                        continue;
                    }

                    var typeName = component.GetType().FullName ?? component.GetType().Name;
                    if (!string.IsNullOrWhiteSpace(typeName))
                    {
                        result.Add(typeName);
                    }
                }
                catch { }
            }
        }
        catch { }

        return result.Distinct().ToList();
    }

    private static string? SafeReadGameObjectText(GameObject? gameObject)
    {
        if (gameObject == null)
        {
            return null;
        }

        var texts = new List<string>();
        try
        {
            foreach (var component in gameObject.GetComponents<Component>())
            {
                try
                {
                    var text = SafeReadText(component);
                    if (!string.IsNullOrWhiteSpace(text))
                    {
                        texts.Add(text!);
                    }

                    foreach (var property in component.GetType().GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
                    {
                        try
                        {
                            if (!property.CanRead ||
                                property.GetIndexParameters().Length > 0 ||
                                property.Name.IndexOf("Text", StringComparison.OrdinalIgnoreCase) < 0)
                            {
                                continue;
                            }

                            var value = property.GetValue(component);
                            var nestedText = SafeReadText(value);
                            if (!string.IsNullOrWhiteSpace(nestedText))
                            {
                                texts.Add(nestedText!);
                            }
                        }
                        catch { }
                    }
                }
                catch { }
            }
        }
        catch { }

        if (texts.Count == 0)
        {
            return null;
        }

        return string.Join(" | ", texts.Distinct().Take(4));
    }

    private static string? SafeReadText(object? target)
    {
        if (target == null)
        {
            return null;
        }

        try
        {
            var property = target.GetType().GetProperty("text", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (property == null || !property.CanRead || property.GetIndexParameters().Length > 0)
            {
                property = target.GetType().GetProperty("Text", BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            }

            if (property != null && property.CanRead && property.GetIndexParameters().Length == 0)
            {
                var value = property.GetValue(target);
                if (value != null)
                {
                    var text = value.ToString();
                    return string.IsNullOrWhiteSpace(text) ? null : text;
                }
            }
        }
        catch { }

        return null;
    }

    private static Vector3 WorldToScreen(Vector3 position, out bool inBounds)
    {
        inBounds = false;
        try
        {
            var camera = Camera.main;
            if (camera == null)
            {
                return new Vector3(0f, 0f, 0f);
            }

            var screen = camera.WorldToScreenPoint(position);
            inBounds = screen.z > 0f &&
                       screen.x >= -64f && screen.x <= Screen.width + 64f &&
                       screen.y >= -64f && screen.y <= Screen.height + 64f;
            return screen;
        }
        catch
        {
            return new Vector3(0f, 0f, 0f);
        }
    }

    private static GameObject? SafeReadCardParent(CardUI card)
    {
        try { return card.parent; }
        catch { return null; }
    }

    private static string? SafeObjectName(GameObject? gameObject)
    {
        try { return gameObject != null ? gameObject.name : null; }
        catch { return null; }
    }

    private static string BuildHierarchyPath(Transform? transform)
    {
        if (transform == null)
        {
            return "";
        }

        var names = new List<string>();
        var current = transform;
        var guard = 0;
        while (current != null && guard++ < 64)
        {
            try { names.Add(current.name); }
            catch { break; }

            try { current = current.parent; }
            catch { break; }
        }

        names.Reverse();
        return string.Join("/", names);
    }

    private GameObject? FindExactStartButtonObject()
    {
        try
        {
            foreach (var transform in Object.FindObjectsOfType<Transform>())
            {
                try
                {
                    if (transform == null || transform.gameObject == null || !transform.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    WorldToScreen(transform.position, out var inBounds);
                    if (!inBounds)
                    {
                        continue;
                    }

                    if (IsExactStartButtonObject(transform.gameObject))
                    {
                        return transform.gameObject;
                    }
                }
                catch { }
            }
        }
        catch { }

        return null;
    }

    private bool TryPressExactStartButtonOnce(GameObject gameObject, List<string> actions, out string methodUsed)
    {
        methodUsed = "";
        foreach (var component in gameObject.GetComponents<Component>())
        {
            if (component == null)
            {
                continue;
            }

            var typeName = component.GetType().FullName ?? component.GetType().Name;
            if (TryInvokeNoArg(component, "OnMouseDown", out _))
            {
                actions.Add($"{typeName}.OnMouseDown({SafeObjectName(gameObject)})");
            }

            if (TryInvokeNoArg(component, "OnMouseUpAsButton", out var upError))
            {
                methodUsed = $"{typeName}.OnMouseUpAsButton";
                actions.Add($"{methodUsed}({SafeObjectName(gameObject)})");
                return true;
            }

            if (TryInvokeNoArg(component, "OnMouseUp", out _))
            {
                methodUsed = $"{typeName}.OnMouseUp";
                actions.Add($"{methodUsed}({SafeObjectName(gameObject)})");
                return true;
            }

            if (TryInvokeUnityEvent(component, "clickEvent", out methodUsed) ||
                TryInvokeUnityEvent(component, "onClick", out methodUsed))
            {
                actions.Add($"{methodUsed}({SafeObjectName(gameObject)})");
                return true;
            }

            if (!string.IsNullOrWhiteSpace(upError))
            {
                actions.Add($"{typeName}.OnMouseUpAsButton unavailable on {SafeObjectName(gameObject)}: {upError}");
            }
        }

        if (TryNativeMouseClickGameObject(gameObject, actions, out methodUsed))
        {
            return true;
        }

        return TrySendStartMessageRequireReceiver(gameObject, actions, out methodUsed);
    }

    private bool TrySendStartMessageRequireReceiver(GameObject gameObject, List<string> actions, out string methodUsed)
    {
        methodUsed = "";
        try
        {
            gameObject.SendMessage("OnMouseDown", SendMessageOptions.RequireReceiver);
            actions.Add($"GameObject.SendMessage.OnMouseDown({SafeObjectName(gameObject)})");
        }
        catch (Exception ex)
        {
            actions.Add($"GameObject.SendMessage.OnMouseDown unavailable on {SafeObjectName(gameObject)}: {ex.Message}");
        }

        foreach (var message in new[] { "OnMouseUpAsButton", "OnMouseUp" })
        {
            try
            {
                gameObject.SendMessage(message, SendMessageOptions.RequireReceiver);
                methodUsed = $"GameObject.SendMessage.{message}";
                actions.Add($"{methodUsed}({SafeObjectName(gameObject)})");
                return true;
            }
            catch (Exception ex)
            {
                actions.Add($"GameObject.SendMessage.{message} unavailable on {SafeObjectName(gameObject)}: {ex.Message}");
            }
        }

        foreach (var message in new[] { "ButtonInput", "EventTrigger" })
        {
            try
            {
                gameObject.SendMessage(message, SafeObjectName(gameObject) ?? "Start", SendMessageOptions.RequireReceiver);
                methodUsed = $"GameObject.SendMessage.{message}";
                actions.Add($"{methodUsed}({SafeObjectName(gameObject)})");
                return true;
            }
            catch (Exception ex)
            {
                actions.Add($"GameObject.SendMessage.{message} unavailable on {SafeObjectName(gameObject)}: {ex.Message}");
            }
        }

        return false;
    }

    private bool TryPressLetsRockButton(List<string> actions, out string methodUsed)
    {
        methodUsed = "";
        foreach (var gameObject in FindLetsRockButtonCandidates())
        {
            try
            {
                foreach (var component in gameObject.GetComponents<Component>())
                {
                    if (component == null)
                    {
                        continue;
                    }

                    var typeName = component.GetType().FullName ?? component.GetType().Name;
                    if (!LooksLikeButtonType(typeName) && !LooksLikeLetsRockButton(gameObject))
                    {
                        continue;
                    }

                    if (TryInvokeNoArg(component, "OnMouseDown", out _))
                    {
                        actions.Add($"{typeName}.OnMouseDown({SafeObjectName(gameObject)})");
                    }

                    if (TryInvokeNoArg(component, "OnMouseUpAsButton", out var upError))
                    {
                        methodUsed = $"{typeName}.OnMouseUpAsButton";
                        actions.Add($"{methodUsed}({SafeObjectName(gameObject)})");
                        return true;
                    }

                    if (TryInvokeNoArg(component, "OnMouseUp", out _))
                    {
                        methodUsed = $"{typeName}.OnMouseUp";
                        actions.Add($"{methodUsed}({SafeObjectName(gameObject)})");
                        return true;
                    }

                    if (TryInvokeUnityEvent(component, "clickEvent", out methodUsed) ||
                        TryInvokeUnityEvent(component, "onClick", out methodUsed))
                    {
                        actions.Add($"{methodUsed}({SafeObjectName(gameObject)})");
                        return true;
                    }

                    if (!string.IsNullOrWhiteSpace(upError))
                    {
                        actions.Add($"{typeName}.OnMouseUpAsButton unavailable on {SafeObjectName(gameObject)}: {upError}");
                    }
                }

                if (TryCoordinateClickLetsRockButton(gameObject, actions, out methodUsed))
                {
                    return true;
                }
            }
            catch (Exception ex)
            {
                actions.Add($"LetRock candidate failed: {SafeObjectName(gameObject)} {ex.Message}");
            }
        }

        actions.Add("LetRock button handler not found; falling back to InitBoard start method.");
        return false;
    }

    private List<GameObject> FindLetsRockButtonCandidates()
    {
        var result = new List<GameObject>();
        try
        {
            foreach (var transform in Object.FindObjectsOfType<Transform>())
            {
                try
                {
                    if (transform == null || transform.gameObject == null || !transform.gameObject.activeInHierarchy)
                    {
                        continue;
                    }

                    var gameObject = transform.gameObject;
                    WorldToScreen(transform.position, out var inBounds);
                    if (!inBounds)
                    {
                        continue;
                    }

                    if (LooksLikeLetsRockButton(gameObject))
                    {
                        AddButtonCandidateWithAncestors(result, gameObject);
                    }
                }
                catch { }
            }
        }
        catch { }

        return result
            .Distinct()
            .OrderByDescending(gameObject => IsExactStartButtonObject(gameObject) ? 1 : 0)
            .ThenByDescending(gameObject => IsStartButtonPath(BuildHierarchyPath(gameObject.transform)) ? 1 : 0)
            .ThenByDescending(gameObject => NormalizeUiText(SafeReadGameObjectText(gameObject)).Contains("letsrock") ? 1 : 0)
            .ThenBy(gameObject => BuildHierarchyPath(gameObject.transform))
            .ToList();
    }

    private static void AddButtonCandidateWithAncestors(List<GameObject> result, GameObject gameObject)
    {
        var current = gameObject.transform;
        var guard = 0;
        while (current != null && guard++ < 8)
        {
            try
            {
                if (current.gameObject != null && current.gameObject.activeInHierarchy)
                {
                    result.Add(current.gameObject);
                }
            }
            catch { }

            try { current = current.parent; }
            catch { break; }
        }
    }

    private static bool LooksLikeLetsRockButton(GameObject? gameObject)
    {
        if (gameObject == null)
        {
            return false;
        }

        var name = SafeObjectName(gameObject);
        var path = BuildHierarchyPath(gameObject.transform);
        var text = SafeReadGameObjectText(gameObject);
        var className = string.Join(",", SafeComponentTypeNames(gameObject));
        if (IsStartButtonPath(path))
        {
            return true;
        }

        var normalized = NormalizeUiText($"{name} {path} {text} {className}");
        if (normalized.Contains("letsrock") ||
            normalized.Contains("letrock") ||
            (normalized.Contains("let") && normalized.Contains("rock")) ||
            normalized.Contains("readysetplant"))
        {
            return true;
        }

        return (normalized.Contains("start") || normalized.Contains("battle") || normalized.Contains("ready")) &&
               (normalized.Contains("select") || normalized.Contains("seed") || normalized.Contains("choose") || normalized.Contains("init"));
    }

    private static bool LooksLikeButtonType(string typeName)
    {
        return typeName.IndexOf("UIButton", StringComparison.OrdinalIgnoreCase) >= 0 ||
               typeName.IndexOf("UIBtn", StringComparison.OrdinalIgnoreCase) >= 0 ||
               typeName.IndexOf("InputButton", StringComparison.OrdinalIgnoreCase) >= 0 ||
               typeName.EndsWith(".Button", StringComparison.OrdinalIgnoreCase) ||
               typeName.IndexOf("UnityEngine.UI.Button", StringComparison.OrdinalIgnoreCase) >= 0;
    }

    private static bool TryInvokeNoArg(object target, string methodName, out string error)
    {
        error = "";
        try
        {
            var method = target.GetType()
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(candidate => candidate.Name == methodName && candidate.GetParameters().Length == 0);
            if (method == null)
            {
                error = "method_not_found";
                return false;
            }

            method.Invoke(target, Array.Empty<object>());
            return true;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
    }

    private static bool TryInvokeUnityEvent(object target, string propertyName, out string methodUsed)
    {
        methodUsed = "";
        try
        {
            var property = target.GetType().GetProperty(propertyName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            if (property == null || !property.CanRead || property.GetIndexParameters().Length > 0)
            {
                return false;
            }

            var unityEvent = property.GetValue(target);
            if (unityEvent == null)
            {
                return false;
            }

            var invoke = unityEvent.GetType()
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(candidate => candidate.Name == "Invoke" && candidate.GetParameters().Length == 0);
            if (invoke == null)
            {
                return false;
            }

            invoke.Invoke(unityEvent, Array.Empty<object>());
            methodUsed = $"{target.GetType().FullName}.{propertyName}.Invoke";
            return true;
        }
        catch
        {
            return false;
        }
    }

    private bool TryCoordinateClickLetsRockButton(GameObject gameObject, List<string> actions, out string methodUsed)
    {
        methodUsed = "";
        var targets = new List<GameObject>();
        AddButtonCandidateWithAncestors(targets, gameObject);
        try
        {
            var screen = WorldToScreen(gameObject.transform.position, out var inBounds);
            if (!inBounds)
            {
                actions.Add($"LetRock coordinate fallback refused: {SafeObjectName(gameObject)} is not on-screen.");
                return false;
            }

            var camera = Camera.main;
            if (camera != null)
            {
                var world = camera.ScreenToWorldPoint(new Vector3(screen.x, screen.y, screen.z));
                foreach (var hit in Physics2D.OverlapPointAll(world))
                {
                    try
                    {
                        if (hit != null && hit.gameObject != null)
                        {
                            AddButtonCandidateWithAncestors(targets, hit.gameObject);
                        }
                    }
                    catch { }
                }

                actions.Add($"LetRock coordinate fallback at screen=({screen.x:0.0},{screen.y:0.0},{screen.z:0.0}) targets={targets.Count}");
            }
        }
        catch (Exception ex)
        {
            actions.Add("LetRock coordinate fallback hit-test failed: " + ex.Message);
        }

        foreach (var target in targets.Distinct())
        {
            try
            {
                foreach (var component in target.GetComponents<Component>())
                {
                    if (component == null)
                    {
                        continue;
                    }

                    var typeName = component.GetType().FullName ?? component.GetType().Name;
                    if (!LooksLikeButtonType(typeName) && !LooksLikeLetsRockButton(target))
                    {
                        continue;
                    }

                    try { TryInvokeNoArg(component, "OnMouseDown", out _); } catch { }
                    if (TryInvokeNoArg(component, "OnMouseUpAsButton", out _) ||
                        TryInvokeNoArg(component, "OnMouseUp", out _) ||
                        TryInvokeUnityEvent(component, "clickEvent", out _) ||
                        TryInvokeUnityEvent(component, "onClick", out _))
                    {
                        methodUsed = $"{typeName}.coordinateClick";
                        actions.Add($"{methodUsed}({SafeObjectName(target)})");
                        return true;
                    }
                }

                if (TrySendLetsRockMessages(target, gameObject, actions, out methodUsed))
                {
                    return true;
                }
            }
            catch { }
        }

        if (TryNativeMouseClickGameObject(gameObject, actions, out methodUsed))
        {
            return true;
        }

        return false;
    }

    private bool TrySendLetsRockMessages(
        GameObject target,
        GameObject startObject,
        List<string> actions,
        out string methodUsed)
    {
        methodUsed = "";
        try
        {
            var startName = SafeObjectName(startObject) ?? "Start";
            var startPath = BuildHierarchyPath(startObject.transform);
            target.SendMessage("OnMouseDown", SendMessageOptions.DontRequireReceiver);
            target.SendMessage("OnMouseUpAsButton", SendMessageOptions.DontRequireReceiver);
            target.SendMessage("OnMouseUp", SendMessageOptions.DontRequireReceiver);
            target.SendMessage("ButtonInput", startName, SendMessageOptions.DontRequireReceiver);
            target.SendMessage("ButtonInput", startPath, SendMessageOptions.DontRequireReceiver);
            target.SendMessage("EventTrigger", startName, SendMessageOptions.DontRequireReceiver);
            target.SendMessage("EventTrigger", startPath, SendMessageOptions.DontRequireReceiver);
            methodUsed = "GameObject.SendMessage";
            actions.Add($"{methodUsed}({SafeObjectName(target)} target={startName})");
            return true;
        }
        catch (Exception ex)
        {
            actions.Add($"GameObject.SendMessage failed on {SafeObjectName(target)}: {ex.Message}");
            return false;
        }
    }

    private bool TryNativeMouseClickGameObject(GameObject gameObject, List<string> actions, out string methodUsed)
    {
        methodUsed = "";
        try
        {
            var screen = WorldToScreen(gameObject.transform.position, out var inBounds);
            if (!inBounds)
            {
                actions.Add($"Win32 mouse click refused: {SafeObjectName(gameObject)} is not on-screen.");
                return false;
            }

            var window = Process.GetCurrentProcess().MainWindowHandle;
            if (window == IntPtr.Zero)
            {
                window = GetForegroundWindow();
            }

            if (window == IntPtr.Zero)
            {
                actions.Add("Win32 mouse click unavailable: game window handle was zero.");
                return false;
            }

            try { SetForegroundWindow(window); } catch { }

            var point = new NativePoint
            {
                X = (int)Math.Round(screen.x),
                Y = (int)Math.Round(Screen.height - screen.y)
            };
            if (!ClientToScreen(window, ref point))
            {
                actions.Add($"Win32 mouse click unavailable: ClientToScreen failed for {SafeObjectName(gameObject)}.");
                return false;
            }

            if (!SetCursorPos(point.X, point.Y))
            {
                actions.Add($"Win32 mouse click unavailable: SetCursorPos failed at ({point.X},{point.Y}).");
                return false;
            }

            mouse_event(MouseEventLeftDown, 0, 0, 0, UIntPtr.Zero);
            mouse_event(MouseEventLeftUp, 0, 0, 0, UIntPtr.Zero);
            methodUsed = "Win32.mouse_event";
            actions.Add(
                $"{methodUsed}({SafeObjectName(gameObject)} screen=({screen.x:0.0},{screen.y:0.0}) client=({(int)Math.Round(screen.x)},{(int)Math.Round(Screen.height - screen.y)}) desktop=({point.X},{point.Y}))");
            return true;
        }
        catch (Exception ex)
        {
            actions.Add($"Win32 mouse click failed for {SafeObjectName(gameObject)}: {ex.Message}");
            return false;
        }
    }

    private bool TryNativeMouseClickNormalized(
        float xNorm,
        float yNormFromTop,
        List<string> actions,
        out string methodUsed)
    {
        methodUsed = "";
        try
        {
            var window = Process.GetCurrentProcess().MainWindowHandle;
            if (window == IntPtr.Zero)
            {
                window = GetForegroundWindow();
            }

            if (window == IntPtr.Zero)
            {
                actions.Add("Win32 normalized click unavailable: game window handle was zero.");
                return false;
            }

            try { SetForegroundWindow(window); } catch { }

            var clientX = (int)Math.Round(Screen.width * Mathf.Clamp01(xNorm));
            var clientY = (int)Math.Round(Screen.height * Mathf.Clamp01(yNormFromTop));
            var point = new NativePoint { X = clientX, Y = clientY };
            if (!ClientToScreen(window, ref point))
            {
                actions.Add($"Win32 normalized click unavailable: ClientToScreen failed for client=({clientX},{clientY}).");
                return false;
            }

            if (!SetCursorPos(point.X, point.Y))
            {
                actions.Add($"Win32 normalized click unavailable: SetCursorPos failed at ({point.X},{point.Y}).");
                return false;
            }

            mouse_event(MouseEventLeftDown, 0, 0, 0, UIntPtr.Zero);
            mouse_event(MouseEventLeftUp, 0, 0, 0, UIntPtr.Zero);
            methodUsed = "Win32.normalized_mouse_event";
            actions.Add(
                $"{methodUsed}(norm=({xNorm:0.###},{yNormFromTop:0.###}) client=({clientX},{clientY}) desktop=({point.X},{point.Y}) screen=({Screen.width},{Screen.height}))");
            return true;
        }
        catch (Exception ex)
        {
            actions.Add("Win32 normalized click failed: " + ex.Message);
            return false;
        }
    }

    private static bool IsSeedSelectionSafeForUiAction(SeedProbeDto probe)
    {
        return probe.SeedSelectionActive &&
               probe.SeedSelectionPanelActive &&
               probe.StartButtonActive &&
               !probe.GameplayReady &&
               !probe.BlockingRewardUiActive;
    }

    private bool TryStartSelectedLevel(InitBoard? initBoard, List<string> actions)
    {
        var before = BuildSeedProbe();
        if (!IsSeedSelectionSafeForUiAction(before))
        {
            actions.Add("Start fallback refused because seed-selection UI is not stable/safe.");
            return false;
        }

        if (TryPressLetsRockButton(actions, out var buttonMethod))
        {
            actions.Add($"LetRock pressed via {buttonMethod}");
            return true;
        }

        actions.Add("No visible Start/Let's Rock button press invoked; InitBoard fallbacks are disabled during seed selection.");
        return false;
    }

    private SeedVerificationDto VerifySeedMultiset(IEnumerable<int> requested, IEnumerable<int> selected, string source)
    {
        var requestedList = requested.ToList();
        var selectedList = selected.ToList();
        var selectedCounts = BuildTypeCounts(selectedList);
        var missing = new List<int>();
        foreach (var plantType in requestedList)
        {
            selectedCounts.TryGetValue(plantType, out var count);
            if (count > 0)
            {
                selectedCounts[plantType] = count - 1;
            }
            else
            {
                missing.Add(plantType);
            }
        }

        return new SeedVerificationDto
        {
            Success = missing.Count == 0,
            Source = source,
            RequestedSeedTypes = requestedList.ToArray(),
            MissingSeedTypes = missing.ToArray(),
            SelectedSeedTypes = selectedList.ToArray(),
            RequestedPlantTypeCounts = BuildPlantTypeCounts(requestedList).ToArray(),
            SelectedPlantTypeCounts = BuildPlantTypeCounts(selectedList).ToArray()
        };
    }

    private bool TryInvokeMethod(object target, string methodName, object[] args, List<string> actions)
    {
        try
        {
            var method = target.GetType()
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(candidate => candidate.Name == methodName && candidate.GetParameters().Length == args.Length);
            if (method == null)
            {
                actions.Add($"{target.GetType().Name}.{methodName} not found.");
                return false;
            }

            method.Invoke(target, args);
            return true;
        }
        catch (Exception ex)
        {
            actions.Add($"{target.GetType().Name}.{methodName} failed: {ex.Message}");
            return false;
        }
    }

    private bool SafeCardSelectedOrBanked(CardUI card)
    {
        try { return card.onCardBank || card.isSelected; }
        catch { return false; }
    }

    private bool SafeCardDisabled(CardUI card)
    {
        try { return card.disabled; }
        catch { return true; }
    }

    private object ScreenStateFast()
    {
        var info = DetectRestartScreenFastCached();
        var board = FindBoard();
        var over = false;
        var wave = 0;
        var maxWave = 0;
        var zombieCount = 0;
        var moreZombiesComing = false;
        var boardStartMove = false;
        try
        {
            if (board != null)
            {
                over = board.over;
                wave = board.theWave;
                maxWave = board.theMaxWave;
                zombieCount = board.zombieArray?.Count ?? 0;
                moreZombiesComing = board.moreZombiesComing;
                boardStartMove = board.startMove;
            }
        }
        catch { }

        var possibleWin = maxWave > 0 && wave >= maxWave && zombieCount == 0 && !moreZombiesComing;
        var terminalHint = possibleWin ? "possible_win" : over ? "game_over_or_loss" : board == null ? "board_not_found" : "running";
        var gameplayReady = board != null &&
                            FindCreatePlant() != null &&
                            boardStartMove &&
                            !over &&
                            !HasLossRestartEvidence(info);
        return new
        {
            boardFound = board != null,
            gameplayReady,
            over,
            terminalHint,
            onGameOverScreen = info.OnGameOverScreen,
            lossMenuActive = info.LossMenuActive,
            gameOverTextVisible = info.GameOverTextVisible,
            onRestartScreen = info.OnRestartScreen,
            restartButtonActive = info.RestartButtonActive,
            onPauseMenu = info.OnPauseMenu,
            pauseMenuActive = info.PauseMenuActive,
            pauseRestartButtonActive = info.PauseRestartButtonActive,
            restartDetectionMode = info.RestartDetectionMode,
            screen_check_ms = _config.DebugPerformance ? info.screen_check_ms : 0.0
        };
    }

    private object LegalActionsCommand(JsonElement root)
    {
        var watch = Stopwatch.StartNew();
        var obs = BuildObservation(
            includeDebugArrays: ReadBool(root, _config.DebugObservation, "debug_observation", "debugObservation", "include_debug_arrays", "includeDebugArrays"),
            forceSeedProbe: ReadBool(root, false, "force_seed_probe", "forceSeedProbe"),
            forceRestartProbe: ReadBool(root, false, "force_restart_probe", "forceRestartProbe"));
        watch.Stop();
        return new
        {
            legalActions = obs.LegalActions.ToArray(),
            legalActionCount = obs.LegalActionCount,
            actionCount = obs.ActionCount > 0 ? obs.ActionCount : GetActionCount(obs.RowCount, obs.ColumnCount, obs.SeedSlots.Count),
            rowCount = obs.RowCount,
            columnCount = obs.ColumnCount,
            plantTypes = _config.PlantTypes.ToArray(),
            seedSlots = obs.SeedSlots.ToArray(),
            seedSlotCount = obs.SeedSlotCount,
            slotPlantTypes = obs.SlotPlantTypes,
            plantCosts = obs.PlantCosts.ToArray(),
            cardCooldowns = obs.CardCooldowns.ToArray(),
            gameplayReady = obs.GameplayReady,
            actualGameplayReady = obs.ActualGameplayReady,
            seedSelectionActive = obs.SeedSelectionActive,
            onGameOverScreen = obs.OnGameOverScreen,
            lossMenuActive = obs.LossMenuActive,
            gameOverTextVisible = obs.GameOverTextVisible,
            onLossScreen = obs.OnLossScreen,
            onRestartScreen = obs.OnRestartScreen,
            restartButtonActive = obs.RestartButtonActive,
            restartButtonName = obs.RestartButtonName,
            restartButtonPath = obs.RestartButtonPath,
            restartDetectionReason = obs.RestartDetectionReason,
            restartDetectionMode = obs.RestartDetectionMode,
            onPauseMenu = obs.OnPauseMenu,
            pauseMenuActive = obs.PauseMenuActive,
            pauseRestartButtonActive = obs.PauseRestartButtonActive,
            onSeedSelectionScreen = obs.OnSeedSelectionScreen,
            canReadBoard = obs.CanReadBoard,
            nextStep = obs.NextStep,
            debugMessage = obs.DebugMessage,
            seedSelectionPanelActive = obs.SeedSelectionPanelActive,
            startButtonActive = obs.StartButtonActive,
            blockingRewardUiActive = obs.BlockingRewardUiActive,
            reason = obs.LegalActionReason,
            sun = obs.Sun,
            legal_actions_ms = _config.DebugPerformance ? Math.Round(watch.Elapsed.TotalMilliseconds, 3) : 0.0,
            observe_ms = _config.DebugPerformance ? obs.observe_ms : 0.0,
            bridge_observe_ms = _config.DebugPerformance ? obs.bridge_observe_ms : 0.0,
            screen_check_ms = _config.DebugPerformance ? obs.screen_check_ms : 0.0,
            seed_probe_ms = _config.DebugPerformance ? obs.seed_probe_ms : 0.0,
            ui_scan_ms = _config.DebugPerformance ? obs.ui_scan_ms : 0.0
        };
    }

    private object TeacherActionCommand(JsonElement root)
    {
        var obs = BuildObservation(
            includeDebugArrays: ReadBool(root, _config.DebugObservation, "debug_observation", "debugObservation", "include_debug_arrays", "includeDebugArrays"),
            forceSeedProbe: ReadBool(root, false, "force_seed_probe", "forceSeedProbe"),
            forceRestartProbe: ReadBool(root, false, "force_restart_probe", "forceRestartProbe"));
        var action = RuleBasedTeacherAction(obs);
        return new
        {
            action,
            decoded = DecodeAction(action, obs),
            legalActions = obs.LegalActions.ToArray(),
            gameplayReady = obs.GameplayReady,
            actualGameplayReady = obs.ActualGameplayReady,
            seedSelectionActive = obs.SeedSelectionActive,
            onGameOverScreen = obs.OnGameOverScreen,
            lossMenuActive = obs.LossMenuActive,
            gameOverTextVisible = obs.GameOverTextVisible,
            onRestartScreen = obs.OnRestartScreen,
            restartButtonActive = obs.RestartButtonActive,
            restartDetectionMode = obs.RestartDetectionMode,
            onPauseMenu = obs.OnPauseMenu,
            pauseMenuActive = obs.PauseMenuActive,
            pauseRestartButtonActive = obs.PauseRestartButtonActive,
            reason = obs.LegalActionReason
        };
    }

    private int RuleBasedTeacherAction(ObservationDto obs)
    {
        var legal = new HashSet<int>(obs.LegalActions);
        if (legal.Count == 0)
        {
            return 0;
        }

        var rows = SafePositive(obs.RowCount, _config.FallbackRows);
        var columns = SafePositive(obs.ColumnCount, _config.FallbackColumns);

        const int sunflower = (int)PlantType.SunFlower;
        var sunflowerSlots = obs.SeedSlots
            .Where(slot => slot.PlantType == sunflower && slot.Usable && slot.Ready && obs.Sun >= slot.SeedCost)
            .OrderBy(slot => slot.SeedCost)
            .ThenBy(slot => slot.SlotIndex)
            .ToList();
        if (sunflowerSlots.Count > 0 && obs.Plants.Count < Math.Max(2, rows))
        {
            var action = EncodeAction(sunflowerSlots[0].SlotIndex, obs.Plants.Count % rows, 0, rows, columns);
            if (legal.Contains(action))
            {
                return action;
            }
        }

        var threatenedLanes = obs.Lanes
            .Where(l => l.ZombieCount > 0)
            .OrderBy(l => l.NearestZombieX ?? float.MaxValue)
            .ToList();

        foreach (var lane in threatenedLanes)
        {
            var row = Math.Max(0, Math.Min(rows - 1, lane.Row));
            var plantType = lane.NearestZombieX.HasValue && lane.NearestZombieX.Value <= 2f
                ? (int)PlantType.WallNut
                : (int)PlantType.Peashooter;

            var slots = obs.SeedSlots
                .Where(slot => slot.PlantType == plantType && slot.Usable && slot.Ready && obs.Sun >= slot.SeedCost)
                .OrderBy(slot => slot.SeedCost)
                .ThenBy(slot => slot.SlotIndex)
                .ToList();
            if (slots.Count == 0)
            {
                continue;
            }

            for (var column = 0; column < columns; column++)
            {
                var action = EncodeAction(slots[0].SlotIndex, row, column, rows, columns);
                if (legal.Contains(action))
                {
                    return action;
                }
            }
        }

        return legal.Contains(0) ? 0 : legal.First();
    }

    private static int EncodeAction(int seedSlotIndex, int row, int column, int rows, int columns) =>
        1 + seedSlotIndex * rows * columns + row * columns + column;

    private int GetActionCount(int rows, int columns, int seedSlotCount = 0) =>
        1 + Math.Max(seedSlotCount, _config.PlantTypes.Count) * SafePositive(rows, _config.FallbackRows) * SafePositive(columns, _config.FallbackColumns);

    private DecodedAction DecodeAction(int action, ObservationDto? observation = null)
    {
        var board = FindBoard();
        var rows = observation != null && observation.RowCount > 0
            ? observation.RowCount
            : board != null ? SafePositive(board.rowNum, _config.FallbackRows) : _config.FallbackRows;
        var columns = observation != null && observation.ColumnCount > 0
            ? observation.ColumnCount
            : board != null ? SafePositive(board.columnNum, _config.FallbackColumns) : _config.FallbackColumns;

        if (action <= 0)
        {
            return DecodedAction.Wait();
        }

        var cells = rows * columns;
        var encoded = action - 1;
        var seedSlotIndex = encoded / cells;
        var cell = encoded % cells;
        var slots = observation?.SeedSlots ?? new List<SeedSlotDto>();
        if (seedSlotIndex < 0 || seedSlotIndex >= slots.Count)
        {
            return DecodedAction.Invalid(action, $"seed slot index {seedSlotIndex} out of range; active seedSlotCount={slots.Count}");
        }

        var slot = slots[seedSlotIndex];
        return new DecodedAction
        {
            Kind = "plant",
            SeedSlotIndex = seedSlotIndex,
            CardInstanceId = slot.CardInstanceId,
            PlantType = slot.PlantType,
            PlantTypeName = slot.PlantTypeName,
            SeedCost = slot.SeedCost,
            Row = cell / columns,
            Column = cell % columns
        };
    }

    private void ApplyConfiguredGameSpeed()
    {
        try
        {
            EnsureOriginalSpeed();
            var mode = NormalizeGameSpeedMode(_config.GameSpeedMode);
            var targetGameSpeed = _originalGameSpeed;
            var targetTimeScale = _originalTimeScale;
            var targetFixedDeltaTime = _originalFixedDeltaTime;

            if (mode == "safe")
            {
                // Safe/valid mode keeps normal game timers intact; requested speed is diagnostic only.
            }
            else if (mode == "time_scale")
            {
                var scale = Math.Max(0.01f, _config.GameSpeed);
                targetTimeScale = scale;
                targetFixedDeltaTime = _originalFixedDeltaTime * scale;
            }
            else if (_config.GameSpeed > 0f)
            {
                targetGameSpeed = _config.GameSpeed;
            }

            var configChanged = _speedConfigDirty ||
                                _lastAppliedSpeedMode != mode ||
                                Math.Abs(_lastRequestedGameSpeed - _config.GameSpeed) > 0.0001f;
            var currentGameSpeed = TryReadGameSpeed();
            var currentTimeScale = SafeReadTimeScale();
            var currentFixedDeltaTime = SafeReadFixedDeltaTime();
            var speedDrifted = Math.Abs(currentGameSpeed - targetGameSpeed) > 0.0001f;
            var timeDrifted = Math.Abs(currentTimeScale - targetTimeScale) > 0.0001f;
            var fixedDrifted = Math.Abs(currentFixedDeltaTime - targetFixedDeltaTime) > 0.0001f;
            if (!configChanged && !speedDrifted && !timeDrifted && !fixedDrifted)
            {
                return;
            }

            var wrote = false;
            if (speedDrifted)
            {
                GameAPP.gameSpeed = targetGameSpeed;
                wrote = true;
            }
            if (timeDrifted)
            {
                Time.timeScale = targetTimeScale;
                wrote = true;
            }
            if (fixedDrifted)
            {
                Time.fixedDeltaTime = targetFixedDeltaTime;
                wrote = true;
            }

            if (wrote)
            {
                _speedApplyCount++;
                if (mode == "safe")
                {
                    _validSpeedModeApplyCount++;
                }
            }

            _speedConfigDirty = false;
            _lastAppliedSpeedMode = mode;
            _lastRequestedGameSpeed = _config.GameSpeed;
            _lastAppliedGameSpeed = targetGameSpeed;
            _lastAppliedTimeScale = targetTimeScale;
            _lastAppliedFixedDeltaTime = targetFixedDeltaTime;
        }
        catch
        {
            // GameAPP is not ready yet.
        }
    }

    private void MarkSpeedConfigDirty()
    {
        _speedConfigDirty = true;
    }

    private object RestoreGameSpeed()
    {
        return RestoreGameSpeedInternal();
    }

    private object RestoreGameSpeedInternal()
    {
        EnsureOriginalSpeed();
        var actions = new List<string>();
        try
        {
            GameAPP.gameSpeed = _originalGameSpeed;
            actions.Add($"GameAPP.gameSpeed={_originalGameSpeed}");
        }
        catch (Exception ex)
        {
            actions.Add("GameAPP.gameSpeed restore failed: " + ex.Message);
        }

        try
        {
            Time.timeScale = _originalTimeScale;
            actions.Add($"Time.timeScale={_originalTimeScale}");
        }
        catch (Exception ex)
        {
            actions.Add("Time.timeScale restore failed: " + ex.Message);
        }

        try
        {
            Time.fixedDeltaTime = _originalFixedDeltaTime;
            actions.Add($"Time.fixedDeltaTime={_originalFixedDeltaTime}");
        }
        catch (Exception ex)
        {
            actions.Add("Time.fixedDeltaTime restore failed: " + ex.Message);
        }

        _config.GameSpeed = _originalGameSpeed;
        _config.GameSpeedMode = "game_speed";
        _speedConfigDirty = false;
        _lastAppliedSpeedMode = "game_speed";
        _lastRequestedGameSpeed = _config.GameSpeed;
        _lastAppliedGameSpeed = _originalGameSpeed;
        _lastAppliedTimeScale = _originalTimeScale;
        _lastAppliedFixedDeltaTime = _originalFixedDeltaTime;

        return new
        {
            ok = true,
            restoredGameSpeed = _originalGameSpeed,
            restoredTimeScale = _originalTimeScale,
            restoredFixedDeltaTime = _originalFixedDeltaTime,
            actions
        };
    }

    private float TryReadGameSpeed()
    {
        try { return GameAPP.gameSpeed; }
        catch { return 0f; }
    }

    private void EnsureOriginalSpeed()
    {
        if (_hasOriginalSpeed)
        {
            return;
        }

        _hasOriginalSpeed = true;
        var gameSpeed = TryReadGameSpeed();
        _originalGameSpeed = gameSpeed > 0f ? gameSpeed : 1f;
        _originalTimeScale = SafeReadTimeScale();
        _originalFixedDeltaTime = SafeReadFixedDeltaTime();
    }

    private float SafeReadTimeScale()
    {
        try { return Time.timeScale; }
        catch { return _originalTimeScale; }
    }

    private float SafeReadFixedDeltaTime()
    {
        try { return Time.fixedDeltaTime; }
        catch { return _originalFixedDeltaTime; }
    }

    private float ResolveEffectiveGameSpeed()
    {
        var mode = NormalizeGameSpeedMode(_config.GameSpeedMode);
        if (mode == "safe")
        {
            return _originalGameSpeed;
        }
        if (mode == "time_scale")
        {
            return SafeReadTimeScale();
        }
        var speed = TryReadGameSpeed();
        return speed > 0f ? speed : _config.GameSpeed;
    }

    private static string NormalizeGameSpeedMode(string? value)
    {
        var raw = value?.Trim().ToLowerInvariant() ?? "game_speed";
        return raw switch
        {
            "time_scale" or "timescale" or "unity" or "unity_time_scale" => "time_scale",
            "safe" or "valid" or "valid_speed" => "safe",
            "game_speed" or "gamespeed" or "game" => "game_speed",
            _ => "game_speed"
        };
    }

    private static string SafeReadGameBoardType()
    {
        try { return GameAPP.theBoardType.ToString(); }
        catch { return "unknown"; }
    }

    private static int SafeReadGameBoardLevel()
    {
        try { return GameAPP.theBoardLevel; }
        catch { return -1; }
    }

    private void LogBoardReadyOnce()
    {
        if (_loggedBoardReady != 0)
        {
            return;
        }

        var board = FindBoard();
        if (board == null)
        {
            return;
        }

        _loggedBoardReady = 1;
        LoggerInstance.Msg($"Board detected. Sun={board.theSun}, wave={board.theWave}/{board.theMaxWave}");
    }

    private Board? FindBoard()
    {
        try
        {
            if (Board.Instance != null)
            {
                return Board.Instance;
            }
        }
        catch { }

        try { return Object.FindObjectOfType<Board>(); }
        catch { return null; }
    }

    private CreatePlant? FindCreatePlant()
    {
        try
        {
            if (CreatePlant.Instance != null)
            {
                return CreatePlant.Instance;
            }
        }
        catch { }

        try { return Object.FindObjectOfType<CreatePlant>(); }
        catch { return null; }
    }

    private InitBoard? FindInitBoard()
    {
        try
        {
            if (InitBoard.Instance != null)
            {
                return InitBoard.Instance;
            }
        }
        catch { }

        try { return Object.FindObjectOfType<InitBoard>(); }
        catch { return null; }
    }

    private static int SafePositive(int value, int fallback) => value > 0 ? value : fallback;

    private static int SafeReadColumn(Zombie zombie)
    {
        try { return zombie.Column; }
        catch { return -1; }
    }

    private static bool SafeReadAlive(Zombie zombie)
    {
        try { return zombie.Alive; }
        catch { return zombie.theHealth > 0; }
    }

    private static string? ReadString(JsonElement root, params string[] names)
    {
        foreach (var name in names)
        {
            if (root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String)
            {
                return value.GetString();
            }
        }

        return null;
    }

    private static int ReadInt(JsonElement root, int fallback, params string[] names)
    {
        return TryReadInt(root, out var value, names) ? value : fallback;
    }

    private static int? ReadNullableInt(JsonElement root, params string[] names)
    {
        return TryReadInt(root, out var value, names) ? value : null;
    }

    private static bool TryReadInt(JsonElement root, out int value, params string[] names)
    {
        foreach (var name in names)
        {
            if (!root.TryGetProperty(name, out var element))
            {
                continue;
            }

            if (element.ValueKind == JsonValueKind.Number && element.TryGetInt32(out value))
            {
                return true;
            }

            if (element.ValueKind == JsonValueKind.String && int.TryParse(element.GetString(), out value))
            {
                return true;
            }
        }

        value = 0;
        return false;
    }

    private static bool TryReadFloat(JsonElement root, out float value, params string[] names)
    {
        foreach (var name in names)
        {
            if (!root.TryGetProperty(name, out var element))
            {
                continue;
            }

            if (element.ValueKind == JsonValueKind.Number && element.TryGetSingle(out value))
            {
                return true;
            }

            if (element.ValueKind == JsonValueKind.String && float.TryParse(element.GetString(), out value))
            {
                return true;
            }
        }

        value = 0;
        return false;
    }

    private static bool ReadBool(JsonElement root, bool fallback, params string[] names)
    {
        foreach (var name in names)
        {
            if (!root.TryGetProperty(name, out var element))
            {
                continue;
            }

            if (element.ValueKind == JsonValueKind.True)
            {
                return true;
            }

            if (element.ValueKind == JsonValueKind.False)
            {
                return false;
            }

            if (element.ValueKind == JsonValueKind.String && bool.TryParse(element.GetString(), out var parsed))
            {
                return parsed;
            }
        }

        return fallback;
    }

    private static List<int> ReadIntArray(JsonElement root, params string[] names)
    {
        foreach (var name in names)
        {
            if (!root.TryGetProperty(name, out var element) || element.ValueKind != JsonValueKind.Array)
            {
                continue;
            }

            var values = new List<int>();
            foreach (var item in element.EnumerateArray())
            {
                if (item.ValueKind == JsonValueKind.Number && item.TryGetInt32(out var value))
                {
                    values.Add(value);
                }
            }

            return values;
        }

        return new List<int>();
    }
}

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

internal sealed class PendingRequest
{
    public PendingRequest(string json)
    {
        Json = json;
    }

    public string Json { get; }
    public TaskCompletionSource<BridgeResponse> Completion { get; } =
        new(TaskCreationOptions.RunContinuationsAsynchronously);
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

internal sealed class ProofReport
{
    public bool BoardFound { get; set; }
    public bool CreatePlantFound { get; set; }
    public bool CanReadBoard { get; set; }
    public bool CanReadPlants { get; set; }
    public bool CanReadZombies { get; set; }
    public bool GoNoGo { get; set; }
    public string? NextStep { get; set; }
    public int Sun { get; set; }
    public int Wave { get; set; }
    public int MaxWave { get; set; }
    public int PlantCount { get; set; }
    public int ZombieCount { get; set; }
    public string? ReadError { get; set; }
    public PlacementResult? PlacementAttempt { get; set; }
}

internal sealed class PlacementResult
{
    public bool Success { get; set; }
    public bool PlantPlaced { get; set; }
    public bool CostPaid { get; set; }
    public bool CooldownStarted { get; set; }
    public int PlantCost { get; set; }
    public string? CostSource { get; set; }
    public string? CostWarning { get; set; }
    public string? PaymentSource { get; set; }
    public string? CooldownSource { get; set; }
    public CardCooldownDto? CardCooldown { get; set; }
    public int SunBefore { get; set; }
    public int SunAfter { get; set; }
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public int SeedSlotIndex { get; set; } = -1;
    public int CardInstanceId { get; set; }
    public int Row { get; set; }
    public int Column { get; set; }
    public string? Message { get; set; }
    public string? IllegalReason { get; set; }

    public static PlacementResult Fail(
        int plantType,
        int row,
        int column,
        string message,
        string illegalReason,
        PlantCostInfo? costInfo = null,
        int sunBefore = 0,
        int sunAfter = 0,
        CardCooldownDto? cooldown = null) =>
        new()
        {
            Success = false,
            PlantPlaced = false,
            CostPaid = false,
            CooldownStarted = false,
            PlantCost = costInfo?.Cost ?? 0,
            CostSource = costInfo?.Source,
            CostWarning = costInfo?.Warning,
            PaymentSource = "not_paid",
            CooldownSource = "not_started",
            CardCooldown = cooldown,
            SunBefore = sunBefore,
            SunAfter = sunAfter,
            PlantType = plantType,
            PlantTypeName = ((PlantType)plantType).ToString(),
            Row = row,
            Column = column,
            Message = message,
            IllegalReason = illegalReason
        };

    public PlacementResult WithSeedSlot(int seedSlotIndex, int cardInstanceId = 0)
    {
        SeedSlotIndex = seedSlotIndex;
        CardInstanceId = cardInstanceId;
        return this;
    }
}

internal sealed class PlantCostInfo
{
    public int PlantType { get; set; }
    public int Cost { get; set; }
    public string Source { get; set; } = "unknown";
    public string? Warning { get; set; }
}

internal sealed class DecodedAction
{
    public string Kind { get; set; } = "wait";
    public int PlantType { get; set; } = -1;
    public string? PlantTypeName { get; set; }
    public int SeedSlotIndex { get; set; } = -1;
    public int CardInstanceId { get; set; }
    public int SeedCost { get; set; }
    public int Row { get; set; } = -1;
    public int Column { get; set; } = -1;
    public string? Error { get; set; }

    public static DecodedAction Wait() => new() { Kind = "wait" };

    public static DecodedAction Invalid(int action, string error) =>
        new() { Kind = "invalid", Error = $"Action {action}: {error}" };
}

internal sealed class ObservationDto
{
    public bool BoardFound { get; set; }
    public bool CreatePlantFound { get; set; }
    public bool GameplayReady { get; set; }
    public bool ActualGameplayReady { get; set; }
    public bool SeedSelectionActive { get; set; }
    public bool SeedSelectionPanelActive { get; set; }
    public bool StartButtonActive { get; set; }
    public bool BlockingRewardUiActive { get; set; }
    public int ActiveGameplayCardBankCount { get; set; }
    public string? LegalActionReason { get; set; }
    public int Sun { get; set; }
    public int Wave { get; set; }
    public int MaxWave { get; set; }
    public int RowCount { get; set; }
    public int ColumnCount { get; set; }
    public int KillCount { get; set; }
    public bool Over { get; set; }
    public bool Done { get; set; }
    public bool BoardCardSelectable { get; set; }
    public bool BoardStartMove { get; set; }
    public bool MoreZombiesComing { get; set; }
    public float Time { get; set; }
    public int FrameCount { get; set; }
    public float RealtimeSinceStartup { get; set; }
    public float GameSpeed { get; set; }
    public float RequestedGameSpeed { get; set; }
    public string? GameSpeedMode { get; set; }
    public float UnityTimeScale { get; set; }
    public float FixedDeltaTime { get; set; }
    public float EffectiveGameSpeed { get; set; }
    public float? SkySunSpawnInterval { get; set; }
    public float? SkySunSpawnTimer { get; set; }
    public int? ActiveFallingSunCount { get; set; }
    public float? SunSpawnCountPerMinute { get; set; }
    public int? SunCollectedCount { get; set; }
    public long SpeedApplyCount { get; set; }
    public long ValidSpeedModeApplyCount { get; set; }
    public long SunSpawnCompensationApplyCount { get; set; }
    public long BridgeUpdateLoopCount { get; set; }
    public long ResetCount { get; set; }
    public long LetsRockClickCount { get; set; }
    public int BoardInstanceId { get; set; }
    public int? ActiveBoardCount { get; set; }
    public int? ActiveSkySunSpawnerCount { get; set; }
    public int? ActiveSunObjectCount { get; set; }
    public int? ActiveCoroutineCount { get; set; }
    public float NewZombieWaveCountDown { get; set; }
    public float NextZombieWaveCountDown { get; set; }
    public float HugeWaveCountDown { get; set; }
    public int PlantCount { get; set; }
    public int VisiblePlantObjectCount { get; set; }
    public int StaleVisiblePlantObjectCount { get; set; }
    public int LogicalMowerCount { get; set; }
    public int VisibleMowerObjectCount { get; set; }
    public int StaleVisibleMowerObjectCount { get; set; }
    public int DuplicateMowerRowCount { get; set; }
    public int ZombieCount { get; set; }
    public int BulletCount { get; set; }
    public int TotalPlantHealth { get; set; }
    public float MinBoardX { get; set; }
    public float MaxBoardX { get; set; }
    public float ZombieMinX { get; set; }
    public float ZombieMaxX { get; set; }
    public string? TerminalHint { get; set; }
    public string? ScreenState { get; set; }
    public string? CurrentMode { get; set; }
    public bool IsMainMenu { get; set; }
    public bool IsAdventureButtonVisible { get; set; }
    public bool StartupPopupVisible { get; set; }
    public bool StartupOkButtonVisible { get; set; }
    public bool MainMenuBlockedByPopup { get; set; }
    public bool IsSeedSelectionScreen { get; set; }
    public bool IsGameplayReady { get; set; }
    public bool IsLevelComplete { get; set; }
    public bool IsRewardScreen { get; set; }
    public bool IsNewPlantUnlockedScreen { get; set; }
    public bool IsAlmanacOrSeedPacketScreen { get; set; }
    public bool IsGameOverScreen { get; set; }
    public int CurrentAdventureLevel { get; set; }
    public int CurrentWorldOrStage { get; set; }
    public int CurrentDayLevel { get; set; }
    public string[] UnlockedSeedNames { get; set; } = Array.Empty<string>();
    public string[] AvailableSeedNames { get; set; } = Array.Empty<string>();
    public string[] SelectedSeedNames { get; set; } = Array.Empty<string>();
    public string[] UnknownVisibleSeedNames { get; set; } = Array.Empty<string>();
    public string? ReadError { get; set; }
    public int LegalActionCount { get; set; }
    public int ActionCount { get; set; }
    public int SeedSlotCount { get; set; }
    public int[] SlotPlantTypes { get; set; } = Array.Empty<int>();
    public bool CanReadBoard { get; set; }
    public bool OnGameOverScreen { get; set; }
    public bool LossMenuActive { get; set; }
    public bool GameOverTextVisible { get; set; }
    public bool OnLossScreen { get; set; }
    public bool OnRestartScreen { get; set; }
    public bool RestartButtonActive { get; set; }
    public string? RestartButtonName { get; set; }
    public string? RestartButtonPath { get; set; }
    public string? RestartDetectionReason { get; set; }
    public string? RestartDetectionMode { get; set; }
    public bool OnPauseMenu { get; set; }
    public bool PauseMenuActive { get; set; }
    public bool PauseRestartButtonActive { get; set; }
    public bool OnSeedSelectionScreen { get; set; }
    public string? NextStep { get; set; }
    public string? DebugMessage { get; set; }
    public double observe_ms { get; set; }
    public double bridge_observe_ms { get; set; }
    public double screen_check_ms { get; set; }
    public double seed_probe_ms { get; set; }
    public double ui_scan_ms { get; set; }
    public List<PlantDto> Plants { get; } = new();
    public List<VisiblePlantDto> VisiblePlants { get; } = new();
    public List<VisibleMowerDto> VisibleMowers { get; } = new();
    public List<int> DuplicateMowerRows { get; } = new();
    public List<ZombieDto> Zombies { get; } = new();
    public List<LaneDto> Lanes { get; } = new();
    public List<PlantCostDto> PlantCosts { get; } = new();
    public List<CardCooldownDto> CardCooldowns { get; } = new();
    public List<SeedSlotDto> SeedSlots { get; } = new();
    public List<int> LegalActions { get; } = new();
}

internal sealed class RestartScreenInfo
{
    public bool OnGameOverScreen { get; set; }
    public bool LossMenuActive { get; set; }
    public bool GameOverTextVisible { get; set; }
    public bool OnRestartScreen { get; set; }
    public bool RestartButtonActive { get; set; }
    public string? RestartButtonName { get; set; }
    public string? RestartButtonPath { get; set; }
    public string? RestartDetectionReason { get; set; }
    public string? RestartDetectionMode { get; set; }
    public bool OnPauseMenu { get; set; }
    public bool PauseMenuActive { get; set; }
    public bool PauseRestartButtonActive { get; set; }
    public double screen_check_ms { get; set; }
    public GameObject? RestartButtonObject;
}

internal sealed class RestartUiCache
{
    public bool Valid { get; set; }
    public string? SceneKey { get; set; }
    public string? InvalidReason { get; set; }
    public LoseMenuBtn? LossRestartButton { get; set; }
    public GameObject? LossMenuRoot { get; set; }
    public GameObject? GameOverTextObject { get; set; }
    public PauseMenu_Btn? PauseRestartButton { get; set; }
    public GameObject? PauseMenuRoot { get; set; }
}

internal sealed class SeedSlotDto
{
    public int SlotIndex { get; set; }
    public int CardInstanceId { get; set; }
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public int SeedCost { get; set; }
    public bool Ready { get; set; }
    public bool Disabled { get; set; }
    public bool IsAvailable { get; set; }
    public float RawCooldown { get; set; }
    public float FullCooldown { get; set; }
    public float CurrentCooldown { get; set; }
    public bool Usable { get; set; }
    public string? Source { get; set; }
    public string? Warning { get; set; }
    public string? HierarchyPath { get; set; }
}

internal sealed class ResetCleanupReport
{
    public bool BoardFound { get; set; }
    public int LogicalPlantCount { get; set; }
    public int VisiblePlantObjectCount { get; set; }
    public int StaleVisiblePlantObjectCount { get; set; }
    public int LogicalMowerCount { get; set; }
    public int VisibleMowerObjectCount { get; set; }
    public int StaleVisibleMowerObjectCount { get; set; }
    public int DuplicateMowerRowCount { get; set; }
    public int LogicalZombieCount { get; set; }
    public int SceneZombieObjectCount { get; set; }
    public int LogicalBulletCount { get; set; }
    public int SceneBulletObjectCount { get; set; }
    public int LogicalGridItemCount { get; set; }
    public int SceneGridItemObjectCount { get; set; }
    public int SceneMowerObjectCount { get; set; }
    public List<VisiblePlantDto> VisiblePlants { get; } = new();
    public List<VisibleMowerDto> VisibleMowers { get; } = new();
    public List<int> DuplicateMowerRows { get; } = new();
}

internal sealed class PlantCostDto
{
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public int Cost { get; set; }
    public string? Source { get; set; }
}

internal sealed class CardCooldownDto
{
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public bool Found { get; set; }
    public bool Ready { get; set; }
    public float RawCooldown { get; set; }
    public float CurrentCooldown { get; set; }
    public float FullCooldown { get; set; }
    public bool IsAvailable { get; set; }
    public bool Disabled { get; set; }
    public bool OnCardBank { get; set; }
    public bool IsSelected { get; set; }
    public int SeedCost { get; set; }
    public int MatchingCardCount { get; set; }
    public string? Source { get; set; }
}

internal sealed class SeedCardDto
{
    public int InstanceId { get; set; }
    public bool Active { get; set; }
    public bool UiVisible { get; set; }
    public bool RendererVisible { get; set; }
    public bool HasVisibleUiComponent { get; set; }
    public bool InScreenBounds { get; set; }
    public bool InScreenSpaceBounds { get; set; }
    public float ScreenX { get; set; }
    public float ScreenY { get; set; }
    public float ScreenZ { get; set; }
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public string? DisplayName { get; set; }
    public int SeedCost { get; set; }
    public bool OnCardBank { get; set; }
    public bool IsSelected { get; set; }
    public bool PreSelected { get; set; }
    public bool IsAvailable { get; set; }
    public bool Disabled { get; set; }
    public bool Selectable { get; set; }
    public float Cd { get; set; }
    public float FullCd { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public float LocalX { get; set; }
    public float LocalY { get; set; }
    public string? GameObjectName { get; set; }
    public string? ParentName { get; set; }
    public string? CardParentName { get; set; }
    public string? RootName { get; set; }
    public string? HierarchyPath { get; set; }
    public string? Text { get; set; }
    public string? TextBg { get; set; }
    public string Classification { get; set; } = "runtimeCardWrapper";
    public string? MatchedRegistryKey { get; set; }
}

internal sealed class UiProbeEntryDto
{
    public string? Name { get; set; }
    public string? ClassName { get; set; }
    public string? Text { get; set; }
    public bool ActiveSelf { get; set; }
    public bool ActiveInHierarchy { get; set; }
    public bool RendererVisible { get; set; }
    public bool UiVisible { get; set; }
    public string? ParentName { get; set; }
    public string? RootName { get; set; }
    public string? HierarchyPath { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public float Z { get; set; }
    public float LocalX { get; set; }
    public float LocalY { get; set; }
    public float ScreenX { get; set; }
    public float ScreenY { get; set; }
    public float ScreenZ { get; set; }
    public bool InScreenBounds { get; set; }
}

internal sealed class UnlockScreenSnapshotDto
{
    public bool RewardScreenVisible { get; set; }
    public bool UnlockScreenVisible { get; set; }
    public bool NewPlantUnlockedVisible { get; set; }
    public string NewPlantUnlockedName { get; set; } = "";
    public int NewPlantUnlockedPlantType { get; set; } = -1;
    public string[] VisibleRewardTexts { get; set; } = Array.Empty<string>();
    public string[] VisibleSeedCardNames { get; set; } = Array.Empty<string>();
    public int[] VisibleSeedPlantTypes { get; set; } = Array.Empty<int>();
    public UiProbeEntryDto[] UnknownUnlockObjects { get; set; } = Array.Empty<UiProbeEntryDto>();
    public SeedCardDto[] UnknownVisibleSeedCards { get; set; } = Array.Empty<SeedCardDto>();
}

internal sealed class SeedProbeDto
{
    public bool BoardFound { get; set; }
    public bool InitBoardFound { get; set; }
    public bool SeedSelectionActive { get; set; }
    public bool SeedSelectionPanelActive { get; set; }
    public bool ChooseYourPlantsTextActive { get; set; }
    public bool StartButtonActive { get; set; }
    public bool BlockingRewardUiActive { get; set; }
    public bool BoardCardSelectable { get; set; }
    public bool BoardStartMove { get; set; }
    public bool GameplayReady { get; set; }
    public string? GameBoardType { get; set; }
    public int GameBoardLevel { get; set; }
    public int[] ConfiguredPlantTypes { get; set; } = Array.Empty<int>();
    public int CardCount { get; set; }
    public int SelectedOrBankedCount { get; set; }
    public int AvailableSeedPacketCount { get; set; }
    public int SelectedBankVisibleCount { get; set; }
    public int[] SelectedBankPlantTypes { get; set; } = Array.Empty<int>();
    public PlantTypeCountDto[] SelectedBankPlantTypeCounts { get; set; } = Array.Empty<PlantTypeCountDto>();
    public int AvailableCardVisibleCount { get; set; }
    public int[] AvailableCardPlantTypes { get; set; } = Array.Empty<int>();
    public PlantTypeCountDto[] AvailableCardPlantTypeCounts { get; set; } = Array.Empty<PlantTypeCountDto>();
    public int SeedChooserSignalCount { get; set; }
    public UiProbeEntryDto[] SeedChooserSignals { get; set; } = Array.Empty<UiProbeEntryDto>();
    public List<SeedCardDto> Cards { get; } = new();
    public List<SeedCardDto> AvailableSeedCards { get; } = new();
    public List<SeedCardDto> SelectedSeedBankCards { get; } = new();
    public List<SeedCardDto> ActiveGameplayCardBankCards { get; } = new();
    public List<SeedSlotDto> ActiveGameplaySeedSlots { get; } = new();
    public List<SeedCardDto> StalePreselectedCards { get; } = new();
    public List<SeedCardDto> RuntimeCardWrappers { get; } = new();
    public bool StartReadyAvailable { get; set; }
    public string[] KnownHooks { get; set; } = Array.Empty<string>();
    public string? Investigation { get; set; }
    public double seed_probe_ms { get; set; }
    public double ui_scan_ms { get; set; }
}

internal sealed class SeedRuntimeSnapshot
{
    public bool SeedSelectionActive { get; set; }
    public bool SeedSelectionPanelActive { get; set; }
    public bool StartButtonActive { get; set; }
    public bool BlockingRewardUiActive { get; set; }
    public int ActiveGameplayCardBankCount { get; set; }
    public Dictionary<int, int> ActiveGameplayTypeCounts { get; } = new();
    public List<SeedSlotDto> SeedSlots { get; } = new();

    public static SeedRuntimeSnapshot FromProbe(SeedProbeDto probe)
    {
        var snapshot = new SeedRuntimeSnapshot
        {
            SeedSelectionActive = probe.SeedSelectionActive,
            SeedSelectionPanelActive = probe.SeedSelectionPanelActive,
            StartButtonActive = probe.StartButtonActive,
            BlockingRewardUiActive = probe.BlockingRewardUiActive,
            ActiveGameplayCardBankCount = probe.ActiveGameplayCardBankCards.Count
        };
        foreach (var pair in BridgeModVisibleHelpers.BuildTypeCountsForRuntime(probe.ActiveGameplayCardBankCards.Select(card => card.PlantType)))
        {
            snapshot.ActiveGameplayTypeCounts[pair.Key] = pair.Value;
        }
        foreach (var slot in probe.ActiveGameplaySeedSlots)
        {
            snapshot.SeedSlots.Add(slot);
        }

        return snapshot;
    }
}

internal sealed class StartupPopupInfo
{
    public bool StartupPopupVisible { get; set; }
    public bool StartupOkButtonVisible { get; set; }
    public bool MainMenuBlockedByPopup { get; set; }
    public List<UiProbeEntryDto> StartupOkSignals { get; set; } = new();
    public List<UiProbeEntryDto> StartupPopupSignals { get; set; } = new();
}

internal sealed class SeedRuntimeCache
{
    public bool Valid { get; set; }
    public string? InvalidReason { get; set; }
    public bool SeedSelectionActive { get; set; }
    public bool SeedSelectionPanelActive { get; set; }
    public bool StartButtonActive { get; set; }
    public bool BlockingRewardUiActive { get; set; }
    public bool GameplayReady { get; set; }
    public bool ActualGameplayReady { get; set; }
    public int ActiveGameplayCardBankCount { get; set; }
    public Dictionary<int, int> ActiveGameplayTypeCounts { get; } = new();
    public Dictionary<int, int> CachedPlantCosts { get; } = new();
    public Dictionary<int, CardUI> CachedGameplayCards { get; } = new();
    public List<SeedSlotCacheEntry> CachedSeedSlots { get; } = new();
    public List<SeedSlotDto> CachedSeedSlotDtos { get; } = new();

    public SeedRuntimeSnapshot ToSnapshot(bool rawGameplayReady)
    {
        var snapshot = new SeedRuntimeSnapshot
        {
            SeedSelectionActive = SeedSelectionActive,
            SeedSelectionPanelActive = SeedSelectionPanelActive,
            StartButtonActive = StartButtonActive,
            BlockingRewardUiActive = BlockingRewardUiActive,
            ActiveGameplayCardBankCount = ActiveGameplayCardBankCount
        };
        foreach (var pair in ActiveGameplayTypeCounts)
        {
            snapshot.ActiveGameplayTypeCounts[pair.Key] = pair.Value;
        }
        foreach (var slot in CachedSeedSlotDtos)
        {
            snapshot.SeedSlots.Add(slot);
        }

        if (!rawGameplayReady)
        {
            snapshot.ActiveGameplayTypeCounts.Clear();
            snapshot.ActiveGameplayCardBankCount = 0;
            snapshot.SeedSlots.Clear();
        }

        return snapshot;
    }
}

internal sealed class SeedSlotCacheEntry
{
    public int SlotIndex { get; set; }
    public int CardInstanceId { get; set; }
    public CardUI? Card { get; set; }
}

internal static class BridgeModVisibleHelpers
{
    public static Dictionary<int, int> BuildTypeCountsForRuntime(IEnumerable<int> plantTypes)
    {
        var counts = new Dictionary<int, int>();
        foreach (var plantType in plantTypes)
        {
            counts.TryGetValue(plantType, out var current);
            counts[plantType] = current + 1;
        }

        return counts;
    }
}

internal sealed class PlantTypeCountDto
{
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public int Count { get; set; }
}

internal sealed class SeedSelectionAttemptDto
{
    public int AttemptIndex { get; set; }
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public string MethodUsed { get; set; } = "";
    public int CandidateInstanceId { get; set; }
    public int CandidateCostBefore { get; set; }
    public int CandidateCostAfter { get; set; }
    public string? CandidateParentName { get; set; }
    public string? CandidateContainerName { get; set; }
    public string? CandidateHierarchyPath { get; set; }
    public bool ClickInvoked { get; set; }
    public int SelectedBankCountBefore { get; set; }
    public int SelectedBankCountAfter { get; set; }
    public int SelectedBankTypeCountBefore { get; set; }
    public int SelectedBankTypeCountAfter { get; set; }
    public bool SelectedBankCountIncreased { get; set; }
    public bool SelectedBankTypeCountIncreased { get; set; }
    public int DuplicateSelectionIndex { get; set; }
    public bool DuplicateCostIncreaseAccessible { get; set; }
    public bool DuplicateCostIncreaseDetected { get; set; }
    public int[] VisibleCostsBefore { get; set; } = Array.Empty<int>();
    public int[] VisibleCostsAfter { get; set; } = Array.Empty<int>();
    public int[] AvailableCostsBefore { get; set; } = Array.Empty<int>();
    public int[] AvailableCostsAfter { get; set; } = Array.Empty<int>();
    public int[] SelectedBankCostsBefore { get; set; } = Array.Empty<int>();
    public int[] SelectedBankCostsAfter { get; set; } = Array.Empty<int>();
    public bool Success { get; set; }
    public string? Error { get; set; }
}

internal sealed class SeedVerificationDto
{
    public bool Success { get; set; }
    public string? Source { get; set; }
    public int[] RequestedSeedTypes { get; set; } = Array.Empty<int>();
    public int[] SelectedSeedTypes { get; set; } = Array.Empty<int>();
    public int[] MissingSeedTypes { get; set; } = Array.Empty<int>();
    public PlantTypeCountDto[] RequestedPlantTypeCounts { get; set; } = Array.Empty<PlantTypeCountDto>();
    public PlantTypeCountDto[] SelectedPlantTypeCounts { get; set; } = Array.Empty<PlantTypeCountDto>();
}

internal sealed class PlantDto
{
    public int Index { get; set; }
    public int InstanceId { get; set; }
    public int Type { get; set; }
    public string? TypeName { get; set; }
    public int Row { get; set; }
    public int Column { get; set; }
    public int Health { get; set; }
    public int MaxHealth { get; set; }
    public int Level { get; set; }
    public float AttackCooldown { get; set; }
    public float ProduceCooldown { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
}

internal sealed class VisiblePlantDto
{
    public int InstanceId { get; set; }
    public int Type { get; set; }
    public string? TypeName { get; set; }
    public int Row { get; set; }
    public int Column { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public bool ActiveInHierarchy { get; set; }
    public bool InBoardBounds { get; set; }
    public bool InPlantArray { get; set; }
}

internal sealed class VisibleMowerDto
{
    public int InstanceId { get; set; }
    public int Type { get; set; }
    public string? TypeName { get; set; }
    public int Row { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public float MowerX { get; set; }
    public bool ActiveInHierarchy { get; set; }
    public bool InBoardBounds { get; set; }
    public bool InMowerArray { get; set; }
}

internal sealed class ZombieDto
{
    public int Index { get; set; }
    public int Type { get; set; }
    public string? TypeName { get; set; }
    public int Row { get; set; }
    public int Column { get; set; }
    public int Health { get; set; }
    public int MaxHealth { get; set; }
    public int Status { get; set; }
    public string? StatusName { get; set; }
    public float Speed { get; set; }
    public bool Alive { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
}

internal sealed class LaneDto
{
    public int Row { get; set; }
    public int ZombieCount { get; set; }
    public float? NearestZombieX { get; set; }
    public int? NearestZombieHealth { get; set; }
    public int? NearestZombieType { get; set; }
    public int ConeheadCount { get; set; }
    public int BucketheadCount { get; set; }
    public int ToughZombieCount { get; set; }
    public float? ToughZombieNearestX { get; set; }
    public float ToughZombiePressureScore { get; set; }
}

internal sealed class FusionCandidateDto
{
    public int SourceInstanceId { get; set; }
    public string? SourcePlantName { get; set; }
    public int SourcePlantType { get; set; }
    public int SourceRow { get; set; }
    public int SourceCol { get; set; }
    public string? IngredientPlantName { get; set; }
    public int IngredientPlantType { get; set; }
    public int IngredientSeedSlotIndex { get; set; }
    public string? PredictedResultName { get; set; }
    public int PredictedResultType { get; set; } = -1;
    public bool FusionLegal { get; set; }
    public string? FusionBlockedReason { get; set; }
}
