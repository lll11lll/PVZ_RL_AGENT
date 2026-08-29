using System;
using System.Collections.Generic;

namespace PvZRLBridge;

internal readonly record struct GameSpeedTargets(
    float GameSpeed,
    float TimeScale,
    float FixedDeltaTime);

internal static class BridgeObservationHelpers
{
    public static GameSpeedTargets ResolveGameSpeedTargets(
        string mode,
        float requestedGameSpeed,
        float originalGameSpeed,
        float originalTimeScale,
        float originalFixedDeltaTime,
        float currentTimeScale,
        bool gameplayReady)
    {
        var requested = Math.Max(0.01f, requestedGameSpeed);
        if (mode == "time_scale")
        {
            return new GameSpeedTargets(
                originalGameSpeed,
                requested,
                originalFixedDeltaTime * requested);
        }

        if (mode == "game_speed" && gameplayReady)
        {
            // This mirrors the game's native speed setting: GameAPP.gameSpeed
            // stores the selected multiplier and Time.timeScale applies it to
            // the simulation. Preserve a zero time scale while paused; the
            // configured multiplier is restored as soon as play resumes.
            var targetTimeScale = currentTimeScale <= 0.0001f
                ? currentTimeScale
                : requested;
            return new GameSpeedTargets(
                requested,
                targetTimeScale,
                originalFixedDeltaTime);
        }

        // Menu, chooser, transition, terminal, and safe mode retain the
        // captured native timing. The configured multiplier is applied only
        // after the board reaches structural gameplay readiness.
        return new GameSpeedTargets(
            originalGameSpeed,
            originalTimeScale,
            originalFixedDeltaTime);
    }

    public static bool IsMainMenuControlPath(string? hierarchyPath)
    {
        var normalized = (hierarchyPath ?? "")
            .Replace('\\', '/')
            .Trim();
        return normalized.StartsWith(
            "MainMenuCanvas/MainMenu",
            StringComparison.OrdinalIgnoreCase);
    }

    public static bool IsSeedSelectionControlPath(string? hierarchyPath)
    {
        var normalized = (hierarchyPath ?? "")
            .Replace('\\', '/')
            .Trim();
        return normalized.Contains(
            "/InGameUI(Clone)/Bottom/SeedLibrary",
            StringComparison.OrdinalIgnoreCase);
    }

    public static string ClassifyAdventureScreenState(
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
        if (lossDetected)
        {
            return "game_over";
        }
        if (seedSelectionActive)
        {
            return "seed_selection";
        }
        if (rewardActive)
        {
            return "reward_unlock";
        }
        if (gameplayReady)
        {
            return "gameplay";
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

    public static bool IsRawBoardGameplayReady(
        bool boardFound,
        bool createPlantFound,
        bool boardStartMove,
        bool done) =>
        boardFound && createPlantFound && boardStartMove && !done;

    public static bool HasDuplicateActiveBoards(int activeBoardCount) => activeBoardCount > 1;

    public static bool IsActiveGameplaySeedBankReady(
        int activeGameplayCardBankCount,
        IReadOnlyDictionary<int, int> activeGameplayCounts)
    {
        if (activeGameplayCardBankCount <= 0)
        {
            return false;
        }

        foreach (var pair in activeGameplayCounts)
        {
            if (pair.Value > 0)
            {
                return true;
            }
        }

        return false;
    }

    public static void PopulateSeedCompatibilityCollections(
        IReadOnlyList<SeedCardDto> sortedSlotCards,
        Dictionary<int, int> activeGameplayCounts,
        Dictionary<int, int> minimumPositiveCosts)
    {
        activeGameplayCounts.Clear();
        minimumPositiveCosts.Clear();
        foreach (var card in sortedSlotCards)
        {
            activeGameplayCounts.TryGetValue(card.PlantType, out var count);
            activeGameplayCounts[card.PlantType] = count + 1;
            if (card.SeedCost > 0 &&
                (!minimumPositiveCosts.TryGetValue(
                     card.PlantType,
                     out var currentCost) ||
                 card.SeedCost < currentCost))
            {
                minimumPositiveCosts[card.PlantType] = card.SeedCost;
            }
        }
    }

    public static int CellKey(int row, int column, int columnCount) =>
        row * columnCount + column;

    public static HashSet<int> BuildOccupiedCellKeys(
        ObservationDto observation,
        int rowCount,
        int columnCount)
    {
        var occupied = new HashSet<int>();
        foreach (var plant in observation.Plants)
        {
            if (plant.Row >= 0 &&
                plant.Row < rowCount &&
                plant.Column >= 0 &&
                plant.Column < columnCount)
            {
                occupied.Add(CellKey(plant.Row, plant.Column, columnCount));
            }
        }

        foreach (var plant in observation.VisiblePlants)
        {
            if (plant.ActiveInHierarchy &&
                plant.InBoardBounds &&
                plant.Row >= 0 &&
                plant.Row < rowCount &&
                plant.Column >= 0 &&
                plant.Column < columnCount)
            {
                occupied.Add(CellKey(plant.Row, plant.Column, columnCount));
            }
        }

        return occupied;
    }

    public static List<LaneDto> BuildLaneSummaries(
        IReadOnlyList<ZombieDto> zombies,
        int rowCount)
    {
        var safeRowCount = Math.Max(0, rowCount);
        var zombieCounts = new int[safeRowCount];
        var nearest = new ZombieDto?[safeRowCount];
        var coneheadCounts = new int[safeRowCount];
        var bucketheadCounts = new int[safeRowCount];
        var toughCounts = new int[safeRowCount];
        var nearestTough = new ZombieDto?[safeRowCount];
        var toughPressure = new double[safeRowCount];

        foreach (var zombie in zombies)
        {
            var row = zombie.Row;
            if (!zombie.Alive || row < 0 || row >= safeRowCount)
            {
                continue;
            }

            zombieCounts[row]++;
            if (nearest[row] == null ||
                Comparer<float>.Default.Compare(zombie.X, nearest[row]!.X) < 0)
            {
                nearest[row] = zombie;
            }

            var conehead = IsConeheadZombie(zombie);
            var buckethead = IsBucketheadZombie(zombie);
            if (conehead)
            {
                coneheadCounts[row]++;
            }
            if (buckethead)
            {
                bucketheadCounts[row]++;
            }

            var tough = conehead || buckethead || zombie.Health >= 600 || zombie.MaxHealth >= 600;
            if (!tough)
            {
                continue;
            }

            toughCounts[row]++;
            if (nearestTough[row] == null ||
                Comparer<float>.Default.Compare(zombie.X, nearestTough[row]!.X) < 0)
            {
                nearestTough[row] = zombie;
            }
            toughPressure[row] += Math.Max(0f, 1f - zombie.X / 10f);
        }

        var lanes = new List<LaneDto>(safeRowCount);
        for (var row = 0; row < safeRowCount; row++)
        {
            var closest = nearest[row];
            if (closest == null)
            {
                lanes.Add(new LaneDto { Row = row, ZombieCount = 0 });
                continue;
            }

            lanes.Add(new LaneDto
            {
                Row = row,
                ZombieCount = zombieCounts[row],
                NearestZombieX = closest.X,
                NearestZombieHealth = closest.Health,
                NearestZombieType = closest.Type,
                ConeheadCount = coneheadCounts[row],
                BucketheadCount = bucketheadCounts[row],
                ToughZombieCount = toughCounts[row],
                ToughZombieNearestX = nearestTough[row]?.X,
                ToughZombiePressureScore = (float)toughPressure[row]
            });
        }

        return lanes;
    }

    public static bool IsConeheadZombie(ZombieDto zombie)
    {
        var name = (zombie.TypeName ?? "").ToLowerInvariant();
        return zombie.Type is 2 or 12 ||
               name.Contains("cone") ||
               name.Contains("roadblock") ||
               name.Contains("\u8def\u969c");
    }

    public static bool IsBucketheadZombie(ZombieDto zombie)
    {
        var name = (zombie.TypeName ?? "").ToLowerInvariant();
        return zombie.Type is 4 or 13 ||
               name.Contains("bucket") ||
               name.Contains("\u94c1\u6876");
    }

    public static bool IsToughZombie(ZombieDto zombie) =>
        IsConeheadZombie(zombie) ||
        IsBucketheadZombie(zombie) ||
        zombie.Health >= 600 ||
        zombie.MaxHealth >= 600;
}
