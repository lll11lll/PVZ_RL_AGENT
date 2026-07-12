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
}
