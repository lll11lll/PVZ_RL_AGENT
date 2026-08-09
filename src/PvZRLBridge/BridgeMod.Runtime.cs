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

    internal const int MaintainedActionCount = 701;

    private int GetActionCount(int rows, int columns, int seedSlotCount = 0) => MaintainedActionCount;

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

    private static int SafeReadProfileAdventureLevel(out string source)
    {
        source = "";
        var gameAppType = typeof(GameAPP);
        foreach (var name in new[]
                 {
                     "adventureLevel",
                     "AdventureLevel",
                     "theAdventureLevel",
                     "profileAdventureLevel",
                     "currentAdventureLevel",
                     "CurrentAdventureLevel"
                 })
        {
            if (TryReadStaticIntMember(gameAppType, name, out var level) && level > 0)
            {
                source = $"GameAPP.{name}";
                return level;
            }
        }

        try
        {
            var flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static;
            foreach (var field in gameAppType.GetFields(flags))
            {
                var normalized = field.Name.ToLowerInvariant();
                if (!normalized.Contains("adventure") || !normalized.Contains("level"))
                {
                    continue;
                }
                if (TryConvertNumber(field.GetValue(null), out var number) && number > 0)
                {
                    source = $"GameAPP.{field.Name}";
                    return (int)number;
                }
            }
            foreach (var property in gameAppType.GetProperties(flags))
            {
                var normalized = property.Name.ToLowerInvariant();
                if (!normalized.Contains("adventure") || !normalized.Contains("level") || property.GetIndexParameters().Length > 0)
                {
                    continue;
                }
                if (TryConvertNumber(property.GetValue(null, null), out var number) && number > 0)
                {
                    source = $"GameAPP.{property.Name}";
                    return (int)number;
                }
            }
        }
        catch
        {
            source = "";
        }

        return -1;
    }

    private static bool TryReadStaticIntMember(Type type, string name, out int value)
    {
        value = -1;
        var flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static;
        try
        {
            var field = type.GetField(name, flags);
            if (field != null && TryConvertNumber(field.GetValue(null), out var fieldNumber))
            {
                value = (int)fieldNumber;
                return true;
            }
        }
        catch { }

        try
        {
            var property = type.GetProperty(name, flags);
            if (property != null &&
                property.GetIndexParameters().Length == 0 &&
                TryConvertNumber(property.GetValue(null, null), out var propertyNumber))
            {
                value = (int)propertyNumber;
                return true;
            }
        }
        catch { }

        return false;
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

    private static double? ReadNullableDouble(JsonElement root, params string[] names)
    {
        foreach (var name in names)
        {
            if (!root.TryGetProperty(name, out var element))
            {
                continue;
            }

            if (element.ValueKind == JsonValueKind.Number && element.TryGetDouble(out var numericValue))
            {
                return numericValue;
            }

            if (element.ValueKind == JsonValueKind.String && double.TryParse(element.GetString(), out var parsedValue))
            {
                return parsedValue;
            }
        }

        return null;
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
