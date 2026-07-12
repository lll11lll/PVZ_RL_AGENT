using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using PvZRLBridge;

internal static class BridgeObservationBenchmark
{
    private static int _checksum;

    public static int Main(string[] args)
    {
        try
        {
            var outputPath = args.Length > 0
                ? Path.GetFullPath(args[0])
                : Path.GetFullPath("phase6_bridge_pure_final.json");
            var observation = DenseObservation();
            var zombies = DenseZombies();

            var legacyOccupancySignature = LegacyOccupancySignature(observation);
            var indexedOccupancySignature = IndexedOccupancySignature(observation);
            if (!legacyOccupancySignature.SequenceEqual(indexedOccupancySignature))
            {
                throw new InvalidOperationException("occupancy benchmark cell projections differ");
            }
            var legacyOccupancyCount = LegacyOccupancyPass(observation, 14);
            var indexedOccupancyCount = IndexedOccupancyPass(observation, 14);
            if (legacyOccupancyCount != indexedOccupancyCount)
            {
                throw new InvalidOperationException("occupancy benchmark projections differ");
            }

            var legacyLanes = LegacyLanes(zombies, 5);
            var indexedLanes = BridgeObservationHelpers.BuildLaneSummaries(zombies, 5);
            if (!LaneSignatures(legacyLanes).SequenceEqual(LaneSignatures(indexedLanes)))
            {
                throw new InvalidOperationException("lane benchmark projections differ");
            }

            var results = new Dictionary<string, object>
            {
                ["occupancy_legacy_slot_cell_scans"] = Measure(
                    () => LegacyOccupancyPass(observation, 14),
                    samples: 50,
                    iterations: 100),
                ["occupancy_indexed_slot_cell_checks"] = Measure(
                    () => IndexedOccupancyPass(observation, 14),
                    samples: 50,
                    iterations: 100),
                ["lanes_legacy_filter_sort"] = Measure(
                    () => LegacyLanes(zombies, 5).Count,
                    samples: 50,
                    iterations: 200),
                ["lanes_single_pass"] = Measure(
                    () => BridgeObservationHelpers.BuildLaneSummaries(zombies, 5).Count,
                    samples: 50,
                    iterations: 200)
            };

            var payload = new
            {
                schema_version = 1,
                methodology = new
                {
                    runtime = Environment.Version.ToString(),
                    clock = "Stopwatch",
                    samples = 50,
                    occupancy_iterations_per_sample = 100,
                    lane_iterations_per_sample = 200,
                    includes_occupancy_index_build = true,
                    limitation = "Pure DTO/helper benchmark; excludes Unity scans, CheckBox, IL2CPP lifetime, socket, and live bridge latency."
                },
                contracts = new
                {
                    rows = observation.RowCount,
                    columns = observation.ColumnCount,
                    slots = 14,
                    plants = observation.Plants.Count,
                    visible_plants = observation.VisiblePlants.Count,
                    zombies = zombies.Count,
                    occupancy_projection_count = legacyOccupancyCount,
                    occupancy_signature = string.Concat(
                        legacyOccupancySignature.Select(occupied => occupied ? "1" : "0")),
                    lane_signature = LaneSignatures(legacyLanes).ToArray()
                },
                results,
                checksum = _checksum
            };
            var json = JsonSerializer.Serialize(
                payload,
                new JsonSerializerOptions { WriteIndented = true });
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
            File.WriteAllText(outputPath, json + Environment.NewLine);
            Console.WriteLine(json);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("Bridge observation benchmark failed: " + ex);
            return 1;
        }
    }

    private static object Measure(Func<int> operation, int samples, int iterations)
    {
        for (var index = 0; index < 20; index++)
        {
            _checksum = unchecked((_checksum * 397) ^ operation());
        }
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        var values = new List<double>(samples);
        for (var sample = 0; sample < samples; sample++)
        {
            var watch = Stopwatch.StartNew();
            var local = 0;
            for (var iteration = 0; iteration < iterations; iteration++)
            {
                local = unchecked((local * 397) ^ operation());
            }
            watch.Stop();
            _checksum = unchecked((_checksum * 397) ^ local);
            values.Add(watch.Elapsed.TotalMilliseconds / iterations);
        }

        values.Sort();
        var p95Index = Math.Min(values.Count - 1, Math.Max(0, (int)Math.Ceiling(values.Count * 0.95) - 1));
        return new
        {
            median_ms = values[values.Count / 2],
            p95_ms = values[p95Index],
            min_ms = values[0],
            max_ms = values[^1]
        };
    }

    private static ObservationDto DenseObservation()
    {
        var observation = new ObservationDto { RowCount = 5, ColumnCount = 10 };
        for (var index = 0; index < 25; index++)
        {
            observation.Plants.Add(new PlantDto
            {
                Row = index % 5,
                Column = (index * 3) % 10
            });
        }
        for (var index = 0; index < 10; index++)
        {
            observation.VisiblePlants.Add(new VisiblePlantDto
            {
                Row = index % 5,
                Column = (index * 7 + 1) % 10,
                ActiveInHierarchy = index % 4 != 0,
                InBoardBounds = true
            });
        }
        return observation;
    }

    private static List<ZombieDto> DenseZombies()
    {
        var zombies = new List<ZombieDto>();
        for (var index = 0; index < 60; index++)
        {
            zombies.Add(new ZombieDto
            {
                Index = index,
                Row = index % 5,
                Type = index % 9 == 0 ? 4 : index % 7 == 0 ? 2 : 99,
                TypeName = index % 11 == 0 ? "ConeZombie" : "NormalZombie",
                Health = 100 + index * 17,
                MaxHealth = 150 + index * 19,
                Alive = index % 8 != 0,
                X = 11f - (index % 20) * 0.43f
            });
        }
        return zombies;
    }

    private static int LegacyOccupancyPass(ObservationDto observation, int slotCount)
    {
        var available = 0;
        for (var slot = 0; slot < slotCount; slot++)
        {
            for (var row = 0; row < observation.RowCount; row++)
            {
                for (var column = 0; column < observation.ColumnCount; column++)
                {
                    var occupied = observation.Plants.Any(
                        plant => plant.Row == row && plant.Column == column) ||
                        observation.VisiblePlants.Any(
                            plant => plant.ActiveInHierarchy &&
                                     plant.InBoardBounds &&
                                     plant.Row == row &&
                                     plant.Column == column);
                    if (!occupied) available++;
                }
            }
        }
        return available;
    }

    private static IReadOnlyList<bool> LegacyOccupancySignature(ObservationDto observation)
    {
        var signature = new List<bool>(observation.RowCount * observation.ColumnCount);
        for (var row = 0; row < observation.RowCount; row++)
        {
            for (var column = 0; column < observation.ColumnCount; column++)
            {
                signature.Add(
                    observation.Plants.Any(plant => plant.Row == row && plant.Column == column) ||
                    observation.VisiblePlants.Any(
                        plant => plant.ActiveInHierarchy &&
                                 plant.InBoardBounds &&
                                 plant.Row == row &&
                                 plant.Column == column));
            }
        }
        return signature;
    }

    private static int IndexedOccupancyPass(ObservationDto observation, int slotCount)
    {
        var occupied = BridgeObservationHelpers.BuildOccupiedCellKeys(
            observation,
            observation.RowCount,
            observation.ColumnCount);
        var available = 0;
        for (var slot = 0; slot < slotCount; slot++)
        {
            for (var row = 0; row < observation.RowCount; row++)
            {
                for (var column = 0; column < observation.ColumnCount; column++)
                {
                    if (!occupied.Contains(
                            BridgeObservationHelpers.CellKey(
                                row,
                                column,
                                observation.ColumnCount)))
                    {
                        available++;
                    }
                }
            }
        }
        return available;
    }

    private static IReadOnlyList<bool> IndexedOccupancySignature(ObservationDto observation)
    {
        var occupied = BridgeObservationHelpers.BuildOccupiedCellKeys(
            observation,
            observation.RowCount,
            observation.ColumnCount);
        var signature = new List<bool>(observation.RowCount * observation.ColumnCount);
        for (var row = 0; row < observation.RowCount; row++)
        {
            for (var column = 0; column < observation.ColumnCount; column++)
            {
                signature.Add(
                    occupied.Contains(
                        BridgeObservationHelpers.CellKey(
                            row,
                            column,
                            observation.ColumnCount)));
            }
        }
        return signature;
    }

    private static List<LaneDto> LegacyLanes(IReadOnlyList<ZombieDto> zombies, int rowCount)
    {
        var lanes = new List<LaneDto>();
        for (var row = 0; row < rowCount; row++)
        {
            var laneZombies = zombies.Where(zombie => zombie.Row == row && zombie.Alive).ToList();
            if (laneZombies.Count == 0)
            {
                lanes.Add(new LaneDto { Row = row, ZombieCount = 0 });
                continue;
            }
            var nearest = laneZombies.OrderBy(zombie => zombie.X).First();
            var tough = laneZombies.Where(BridgeObservationHelpers.IsToughZombie).ToList();
            var nearestTough = tough.OrderBy(zombie => zombie.X).FirstOrDefault();
            lanes.Add(new LaneDto
            {
                Row = row,
                ZombieCount = laneZombies.Count,
                NearestZombieX = nearest.X,
                NearestZombieHealth = nearest.Health,
                NearestZombieType = nearest.Type,
                ConeheadCount = laneZombies.Count(BridgeObservationHelpers.IsConeheadZombie),
                BucketheadCount = laneZombies.Count(BridgeObservationHelpers.IsBucketheadZombie),
                ToughZombieCount = tough.Count,
                ToughZombieNearestX = nearestTough?.X,
                ToughZombiePressureScore = tough.Sum(zombie => Math.Max(0f, 1f - zombie.X / 10f))
            });
        }
        return lanes;
    }

    private static IEnumerable<string> LaneSignatures(IEnumerable<LaneDto> lanes) =>
        lanes.Select(lane => string.Join(
            ":",
            lane.Row.ToString(CultureInfo.InvariantCulture),
            lane.ZombieCount.ToString(CultureInfo.InvariantCulture),
            lane.NearestZombieX?.ToString(CultureInfo.InvariantCulture) ?? "null",
            lane.NearestZombieHealth?.ToString(CultureInfo.InvariantCulture) ?? "null",
            lane.NearestZombieType?.ToString(CultureInfo.InvariantCulture) ?? "null",
            lane.ConeheadCount.ToString(CultureInfo.InvariantCulture),
            lane.BucketheadCount.ToString(CultureInfo.InvariantCulture),
            lane.ToughZombieCount.ToString(CultureInfo.InvariantCulture),
            lane.ToughZombieNearestX?.ToString(CultureInfo.InvariantCulture) ?? "null",
            lane.ToughZombiePressureScore.ToString(CultureInfo.InvariantCulture)));
}
