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
        var profileAdventureLevel = SafeReadProfileAdventureLevel(out var profileAdventureLevelSource);
        var firstTrophySignal = trophySignals.FirstOrDefault();
        var world = level > 0 ? ((level - 1) / 10) + 1 : 0;
        var dayLevel = level > 0 ? ((level - 1) % 10) + 1 : 0;

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
            profileAdventureLevel,
            profileAdventureLevelSource,
            currentWorldOrStage = world,
            currentDayLevel = dayLevel,
            uiWorldLevelText = world > 0 && dayLevel > 0 ? $"{world}-{dayLevel}" : "",
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
}
