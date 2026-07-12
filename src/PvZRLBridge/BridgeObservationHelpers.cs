using System;
using System.Collections.Generic;

namespace PvZRLBridge;

internal static class BridgeObservationHelpers
{
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
