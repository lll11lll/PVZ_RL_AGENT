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

    private void ArmBoardSingletonCheck(string reason)
    {
        _boardSingletonCheckArmed = true;
        _lastBoardSingletonCheckFrame = -100000;
        _lastActiveBoardCount = 0;
        _boardSingletonStableChecks = 0;
        if (!string.IsNullOrWhiteSpace(reason))
        {
            LoggerInstance.Msg($"[board-singleton] check armed reason={reason}");
        }
    }

    private Board? EnsureSingleActiveBoard(Board? primaryHint, out int activeBoardCount)
    {
        const int periodicCheckFrames = 30;
        const int armedRetryFrames = 15;
        var frame = 0;
        try { frame = Time.frameCount; } catch { }

        var checkInterval = _boardSingletonCheckArmed ? armedRetryFrames : periodicCheckFrames;
        var shouldCheck = frame - _lastBoardSingletonCheckFrame >= checkInterval;
        if (!shouldCheck)
        {
            activeBoardCount = _lastActiveBoardCount;
            return primaryHint;
        }

        _lastBoardSingletonCheckFrame = frame;
        List<Board> boards;
        try
        {
            boards = Object.FindObjectsOfType<Board>()
                .Where(candidate => candidate != null)
                .ToList();
        }
        catch (Exception ex)
        {
            // A failed safety scan is not proof that only one Board remains.
            // Fail closed for this observation and keep the gate armed so the
            // next observation retries the singleton check.
            _boardSingletonCheckArmed = true;
            _boardSingletonStableChecks = 0;
            activeBoardCount = Math.Max(2, _lastActiveBoardCount);
            _lastActiveBoardCount = activeBoardCount;
            LoggerInstance.Msg($"[board-singleton] scan failed; retrying: {ex.Message}");
            return primaryHint;
        }

        var previousCount = _lastActiveBoardCount;
        activeBoardCount = boards.Count;
        _lastActiveBoardCount = activeBoardCount;
        if (activeBoardCount <= 1)
        {
            if (_boardSingletonCheckArmed)
            {
                // Keep checking for a short grace window after a restart.
                // UIMgr.EnterGame can create the replacement Board a few
                // frames after the old Board disappears; disarming on the
                // first singleton observation would reopen the duplicate
                // spawner race in that gap.
                if (activeBoardCount == 1)
                {
                    _boardSingletonStableChecks++;
                }
                else
                {
                    _boardSingletonStableChecks = 0;
                }

                if (previousCount > 1 || _boardSingletonStableChecks == 1)
                {
                    LoggerInstance.Msg(
                        $"[board-singleton] activeBoards={activeBoardCount}; " +
                        $"stableChecks={_boardSingletonStableChecks}/3");
                }

                if (_boardSingletonStableChecks < 3)
                {
                    return boards.FirstOrDefault() ?? primaryHint;
                }

                LoggerInstance.Msg("[board-singleton] recovered activeBoards=1");
            }
            _boardSingletonCheckArmed = false;
            _boardSingletonStableChecks = 0;
            return boards.FirstOrDefault() ?? primaryHint;
        }

        var primary = SelectPrimaryBoard(boards, primaryHint);
        var actions = new List<string>();
        DestroyDuplicateBoards(primary, actions);
        _boardSingletonCheckArmed = true;
        _boardSingletonStableChecks = 0;
        if (previousCount != activeBoardCount)
        {
            LoggerInstance.Msg(
                $"[board-singleton] blocked gameplay activeBoards={activeBoardCount}; " +
                string.Join(" ", actions));
        }
        return primary;
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
        var primaryGameObjectId = SafeInstanceId(primary.gameObject);
        var destroyed = 0;
        foreach (var duplicate in boards)
        {
            if (SafeInstanceId(duplicate) == primaryId)
            {
                continue;
            }

            try
            {
                // Normally each Board owns a distinct root GameObject. If a
                // malformed scene instead attached two Board components to
                // one root, remove only the duplicate component so the
                // retained Board and its scene object stay alive.
                if (SafeInstanceId(duplicate.gameObject) == primaryGameObjectId)
                {
                    Object.Destroy(duplicate);
                    actions.Add($"DestroyDuplicateBoardComponent({SafeInstanceId(duplicate)})");
                }
                else
                {
                    Object.Destroy(duplicate.gameObject);
                }
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

    private Board SelectPrimaryBoard(List<Board> boards, Board? primaryHint)
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
                    ArmBoardSingletonCheck("loss_menu_restart");
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
            ArmBoardSingletonCheck("restart_fallback");
            return true;
        }

        if (string.IsNullOrWhiteSpace(error))
        {
            error = fallbackError;
        }

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
            ArmBoardSingletonCheck("UIMgr.EnterGame");
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

        if (invoked)
        {
            ArmBoardSingletonCheck("InitBoard.quick_reset");
        }
        return invoked;
    }
}
