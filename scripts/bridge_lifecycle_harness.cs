using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using PvZRLBridge;

internal static class BridgeLifecycleHarness
{
    private const long FakeFrequency = 1000;
    private const long FakeStart = 1000;
    private static int _checks;

    public static int Main()
    {
        try
        {
            DeadlineIsMonotonicAndDeterministic();
            ExpiredQueueEntryNeverDispatches();
            ExpiredMutatingCommandsNeverDispatch();
            TimeoutCancellationWinsBeforeDispatch();
            DispatchOwnershipSuppressesTimeout();
            StopDrainsAndRejectsQueuedWork();
            DispatchCancelRaceHasOneOwner();
            DispatchStopRaceHasOneOwner();
            ClientRegistryStopsAndDrainsBoundedly();
            ObservationSchemaIsStable();
            ProtectedActionCountRemainsStable();
            OccupancyIndexMatchesLegacyScans();
            LaneSummariesMatchLegacyProjection();
            SeedCompatibilityCollectionsMatchLegacyProjection();
            RawGameplayReadinessIsStructural();
            DuplicateBoardGateRequiresSingleton();
            GameplaySpeedTargetsMatchNativeSemantics();
            ActiveGameplaySeedBankReadinessUsesRuntimeBank();
            AdventureScreenClassificationPrioritizesLoss();
            MainMenuAlmanacIsNotUnlockContext();
            SeedSelectionAlmanacIsNotUnlockContext();
            Console.WriteLine($"Bridge lifecycle harness passed: {_checks} checks.");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine("Bridge lifecycle harness failed: " + ex);
            return 1;
        }
    }

    private static void DeadlineIsMonotonicAndDeterministic()
    {
        var request = NewRequest(1, timeoutMilliseconds: 100);
        Check(request.RequestId == 1, "request ID must be retained");
        Check(request.CreatedTimestamp == FakeStart, "created timestamp must be retained");
        Check(request.DeadlineTimestamp == FakeStart + 100, "deadline must use the injected monotonic frequency");
        Check(
            Math.Abs(request.RemainingUntilDeadline(FakeStart + 50).TotalMilliseconds - 50.0) < 0.001,
            "remaining duration must derive from monotonic timestamps");
        Check(request.RemainingUntilDeadline(FakeStart + 100) == TimeSpan.Zero, "deadline must expire at equality");
    }

    private static void ExpiredQueueEntryNeverDispatches()
    {
        var queue = new PendingRequestQueue();
        queue.StartAccepting();
        var request = NewRequest(2, timeoutMilliseconds: 100);
        Check(queue.TryEnqueue(request, ServerStopping()), "live queue must accept work");
        Check(
            queue.TryTakeForDispatch(FakeStart + 100, Timeout(), out var claimed),
            "expired queue entry must be consumed");
        Check(claimed == null, "expired queue entry must not be dispatchable");
        Check(request.State == PendingRequestState.Canceled, "expired queue entry must be canceled");
        Check(request.Completion.Task.Result.Error == "timeout", "expired queue entry must complete with timeout");
    }

    private static void TimeoutCancellationWinsBeforeDispatch()
    {
        var request = NewRequest(3);
        Check(request.TryCancel(Timeout()), "queued timeout cancellation must win");
        Check(!request.TryBeginDispatch(FakeStart + 1, Timeout()), "canceled request must never dispatch later");
        Check(!request.TryComplete(BridgeResponse.Success(null)), "canceled request must not complete as dispatched");
        Check(request.Completion.Task.Result.Error == "timeout", "winning timeout response must remain authoritative");
    }

    private static void ExpiredMutatingCommandsNeverDispatch()
    {
        var commands = new[] { "step", "fusion_step", "configure", "reset" };
        for (var index = 0; index < commands.Length; index++)
        {
            var queue = new PendingRequestQueue();
            queue.StartAccepting();
            var request = new PendingRequest(
                100 + index,
                $"{{\"command\":\"{commands[index]}\"}}",
                FakeStart,
                FakeFrequency,
                TimeSpan.FromMilliseconds(10));
            Check(queue.TryEnqueue(request, ServerStopping()), $"{commands[index]} must enqueue before its deadline");
            Check(
                queue.TryTakeForDispatch(FakeStart + 10, Timeout(), out var claimed),
                $"expired {commands[index]} must be consumed deterministically");
            Check(claimed == null, $"expired {commands[index]} must never reach Unity dispatch");
            Check(request.State == PendingRequestState.Canceled, $"expired {commands[index]} must be canceled");
        }
    }

    private static void DispatchOwnershipSuppressesTimeout()
    {
        var request = NewRequest(4);
        Check(request.TryBeginDispatch(FakeStart + 1, Timeout()), "queued request must be claimable before deadline");
        Check(!request.TryCancel(Timeout()), "timeout must lose after dispatch owns the request");
        Check(!request.Completion.Task.IsCompleted, "dispatch winner must not expose an early timeout completion");
        var success = BridgeResponse.Success(new { value = 1 });
        Check(request.TryComplete(success), "dispatch winner must complete once");
        Check(request.Completion.Task.Result.Ok, "real dispatch response must reach the waiter");
        Check(!request.TryComplete(success), "request must not complete twice");
    }

    private static void StopDrainsAndRejectsQueuedWork()
    {
        var queue = new PendingRequestQueue();
        queue.StartAccepting();
        var queued = NewRequest(5);
        Check(queue.TryEnqueue(queued, ServerStopping()), "queue must accept before stop");
        Check(queue.StopAcceptingAndCancel(ServerStopping()) == 1, "stop must cancel queued work exactly once");
        Check(queued.Completion.Task.Result.Error == "server_stopping", "drained work must get server_stopping");

        var rejected = NewRequest(6);
        Check(!queue.TryEnqueue(rejected, ServerStopping()), "stopped queue must reject new work atomically");
        Check(rejected.State == PendingRequestState.Canceled, "rejected work must be completed, not abandoned");
        Check(rejected.Completion.Task.Result.Error == "server_stopping", "rejected work must get server_stopping");
        Check(
            !queue.TryTakeForDispatch(FakeStart + 1, Timeout(), out _),
            "stopped queue must not dispatch");
    }

    private static void DispatchCancelRaceHasOneOwner()
    {
        for (var iteration = 0; iteration < 1000; iteration++)
        {
            var request = NewRequest(10_000 + iteration);
            using var start = new ManualResetEventSlim(false);
            var dispatchTask = Task.Run(() =>
            {
                start.Wait();
                return request.TryBeginDispatch(FakeStart + 1, Timeout());
            });
            var cancelTask = Task.Run(() =>
            {
                start.Wait();
                return request.TryCancel(Timeout());
            });
            start.Set();
            Task.WaitAll(dispatchTask, cancelTask);

            Check(dispatchTask.Result ^ cancelTask.Result, "dispatch/cancel race must have exactly one winner");
            if (dispatchTask.Result)
            {
                Check(request.TryComplete(BridgeResponse.Success(null)), "dispatch race winner must complete");
                Check(request.Completion.Task.Result.Ok, "dispatch race winner must retain real response");
            }
            else
            {
                Check(request.Completion.Task.Result.Error == "timeout", "cancel race winner must retain timeout response");
                Check(!request.TryBeginDispatch(FakeStart + 2, Timeout()), "cancel race winner must remain non-dispatchable");
            }
        }
    }

    private static void DispatchStopRaceHasOneOwner()
    {
        for (var iteration = 0; iteration < 1000; iteration++)
        {
            var queue = new PendingRequestQueue();
            queue.StartAccepting();
            var request = NewRequest(20_000 + iteration);
            Check(queue.TryEnqueue(request, ServerStopping()), "race queue must accept initial request");

            using var start = new ManualResetEventSlim(false);
            PendingRequest? claimed = null;
            var takeTask = Task.Run(() =>
            {
                start.Wait();
                return queue.TryTakeForDispatch(FakeStart + 1, Timeout(), out claimed);
            });
            var stopTask = Task.Run(() =>
            {
                start.Wait();
                return queue.StopAcceptingAndCancel(ServerStopping());
            });
            start.Set();
            Task.WaitAll(takeTask, stopTask);

            if (claimed != null)
            {
                Check(takeTask.Result, "dispatch winner must consume the queued request");
                Check(stopTask.Result == 0, "stop must not cancel a dispatch-owned request");
                Check(claimed.TryComplete(BridgeResponse.Success(null)), "dispatch-owned request must complete normally");
            }
            else
            {
                Check(stopTask.Result == 1, "stop winner must cancel the queued request");
                Check(request.State == PendingRequestState.Canceled, "stop winner must leave request canceled");
                Check(request.Completion.Task.Result.Error == "server_stopping", "stop winner response must be authoritative");
            }
        }
    }

    private static void ClientRegistryStopsAndDrainsBoundedly()
    {
        var registry = new ActiveClientRegistry();
        registry.StartAccepting();
        using var first = new TcpClient();
        using var second = new TcpClient();
        using var rejected = new TcpClient();
        Check(registry.TryRegister(first, out var firstId), "client registry must accept first client");
        Check(registry.TryRegister(second, out var secondId), "client registry must accept second client");
        Check(registry.ActiveCount == 2, "client registry must count active workers");
        Check(registry.StopAcceptingAndSnapshot().Count == 2, "stop must snapshot every active client");
        Check(!registry.TryRegister(rejected, out _), "stopped client registry must reject new workers");
        Check(!registry.WaitForDrain(TimeSpan.FromMilliseconds(1)), "drain wait must be bounded while workers remain");
        registry.Unregister(firstId);
        registry.Unregister(secondId);
        Check(registry.WaitForDrain(TimeSpan.FromSeconds(1)), "registry must signal after the final worker exits");
        Check(registry.ActiveCount == 0, "client registry must be empty after worker exit");
    }

    private static void ObservationSchemaIsStable()
    {
        const string expectedHash = "c0d34a11cb12f53e0e0a84d82bf35022b73bf3eaeb9e83706bc6e4b3b005b6b5";
        var properties = typeof(ObservationDto).GetProperties(
            BindingFlags.Public | BindingFlags.Instance);
        Check(properties.Length == 122, "ObservationDto property count changed");
        var signature = string.Join(
            "\n",
            properties
                .Select(property =>
                    JsonNamingPolicy.CamelCase.ConvertName(property.Name) +
                    ":" + JsonShape(property.PropertyType))
                .OrderBy(value => value, StringComparer.Ordinal));
        var actualHash = Convert.ToHexString(
            SHA256.HashData(Encoding.UTF8.GetBytes(signature))).ToLowerInvariant();
        Check(
            actualHash == expectedHash,
            $"ObservationDto recursive key/type surface changed: {actualHash}");
    }

    private static string JsonShape(Type type)
    {
        var nullable = Nullable.GetUnderlyingType(type);
        if (nullable != null)
        {
            return JsonShape(nullable) + "?";
        }
        if (type.IsArray)
        {
            return "array<" + JsonShape(type.GetElementType()!) + ">";
        }
        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(List<>))
        {
            return "array<" + JsonShape(type.GetGenericArguments()[0]) + ">";
        }
        if (type == typeof(bool)) return "boolean";
        if (type == typeof(byte) || type == typeof(short) || type == typeof(int) || type == typeof(long)) return "integer";
        if (type == typeof(float) || type == typeof(double) || type == typeof(decimal)) return "number";
        if (type == typeof(string)) return "string";
        if (type.Namespace == typeof(ObservationDto).Namespace &&
            type.Name.EndsWith("Dto", StringComparison.Ordinal))
        {
            var properties = type.GetProperties(BindingFlags.Public | BindingFlags.Instance);
            return "object{" + string.Join(
                ",",
                properties
                    .Select(property =>
                        JsonNamingPolicy.CamelCase.ConvertName(property.Name) +
                        ":" + JsonShape(property.PropertyType))
                    .OrderBy(value => value, StringComparer.Ordinal)) + "}";
        }
        return "object";
    }

    private static void OccupancyIndexMatchesLegacyScans()
    {
        var random = new Random(404);
        for (var iteration = 0; iteration < 500; iteration++)
        {
            var observation = new ObservationDto { RowCount = 5, ColumnCount = 10 };
            var plantCount = random.Next(0, 40);
            for (var index = 0; index < plantCount; index++)
            {
                observation.Plants.Add(new PlantDto
                {
                    Row = random.Next(-1, 7),
                    Column = random.Next(-1, 12)
                });
            }
            var visibleCount = random.Next(0, 40);
            for (var index = 0; index < visibleCount; index++)
            {
                observation.VisiblePlants.Add(new VisiblePlantDto
                {
                    Row = random.Next(-1, 7),
                    Column = random.Next(-1, 12),
                    ActiveInHierarchy = random.Next(0, 2) == 1,
                    InBoardBounds = random.Next(0, 2) == 1
                });
            }

            var occupied = BridgeObservationHelpers.BuildOccupiedCellKeys(
                observation,
                observation.RowCount,
                observation.ColumnCount);
            for (var row = 0; row < observation.RowCount; row++)
            {
                for (var column = 0; column < observation.ColumnCount; column++)
                {
                    var legacy = observation.Plants.Any(
                        plant => plant.Row == row && plant.Column == column) ||
                        observation.VisiblePlants.Any(
                            plant => plant.ActiveInHierarchy &&
                                     plant.InBoardBounds &&
                                     plant.Row == row &&
                                     plant.Column == column);
                    var indexed = occupied.Contains(
                        BridgeObservationHelpers.CellKey(
                            row,
                            column,
                            observation.ColumnCount));
                    Check(legacy == indexed, "occupancy index diverged from legacy scan");
                }
            }
        }
    }

    private static void LaneSummariesMatchLegacyProjection()
    {
        var random = new Random(405);
        for (var iteration = 0; iteration < 500; iteration++)
        {
            var zombies = new List<ZombieDto>();
            var count = random.Next(0, 80);
            for (var index = 0; index < count; index++)
            {
                var typeChoice = random.Next(0, 6);
                zombies.Add(new ZombieDto
                {
                    Index = index,
                    Type = typeChoice == 0 ? 2 : typeChoice == 1 ? 4 : 99,
                    TypeName = typeChoice == 2 ? "ConeZombie" : typeChoice == 3 ? "BucketZombie" : "NormalZombie",
                    Row = random.Next(-1, 7),
                    Health = random.Next(0, 1000),
                    MaxHealth = random.Next(0, 1000),
                    Alive = random.Next(0, 4) != 0,
                    X = (float)(random.NextDouble() * 14.0 - 2.0)
                });
            }

            var actual = BridgeObservationHelpers.BuildLaneSummaries(zombies, 5);
            var legacy = LegacyLaneSummaries(zombies, 5);
            Check(actual.Count == legacy.Count, "lane count changed");
            for (var row = 0; row < actual.Count; row++)
            {
                var left = actual[row];
                var right = legacy[row];
                Check(left.Row == right.Row, "lane row changed");
                Check(left.ZombieCount == right.ZombieCount, "lane zombie count changed");
                Check(left.NearestZombieX == right.NearestZombieX, "lane nearest X changed");
                Check(left.NearestZombieHealth == right.NearestZombieHealth, "lane nearest health changed");
                Check(left.NearestZombieType == right.NearestZombieType, "lane nearest type changed");
                Check(left.ConeheadCount == right.ConeheadCount, "lane conehead count changed");
                Check(left.BucketheadCount == right.BucketheadCount, "lane buckethead count changed");
                Check(left.ToughZombieCount == right.ToughZombieCount, "lane tough count changed");
                Check(left.ToughZombieNearestX == right.ToughZombieNearestX, "lane tough nearest X changed");
                Check(
                    left.ToughZombiePressureScore == right.ToughZombiePressureScore,
                    $"lane tough pressure changed: {left.ToughZombiePressureScore:R} != {right.ToughZombiePressureScore:R}");
            }
        }
    }

    private static List<LaneDto> LegacyLaneSummaries(
        IReadOnlyList<ZombieDto> zombies,
        int rowCount)
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

    private static void SeedCompatibilityCollectionsMatchLegacyProjection()
    {
        var random = new Random(406);
        for (var iteration = 0; iteration < 500; iteration++)
        {
            var cards = new List<SeedCardDto>();
            var cardCount = random.Next(0, 40);
            for (var index = 0; index < cardCount; index++)
            {
                cards.Add(new SeedCardDto
                {
                    InstanceId = index + 1,
                    PlantType = random.Next(-1, 16),
                    SeedCost = random.Next(-25, 500)
                });
            }

            var actualCounts = new Dictionary<int, int> { [999] = 1 };
            var actualCosts = new Dictionary<int, int> { [999] = 1 };
            BridgeObservationHelpers.PopulateSeedCompatibilityCollections(
                cards,
                actualCounts,
                actualCosts);

            var expectedCounts = cards
                .GroupBy(card => card.PlantType)
                .ToDictionary(group => group.Key, group => group.Count());
            var expectedCosts = cards
                .Where(card => card.SeedCost > 0)
                .GroupBy(card => card.PlantType)
                .ToDictionary(group => group.Key, group => group.Min(card => card.SeedCost));
            Check(
                actualCounts.OrderBy(pair => pair.Key).SequenceEqual(
                    expectedCounts.OrderBy(pair => pair.Key)),
                "seed compatibility counts changed");
            Check(
                actualCosts.OrderBy(pair => pair.Key).SequenceEqual(
                    expectedCosts.OrderBy(pair => pair.Key)),
                "seed compatibility minimum costs changed");
        }
    }

    private static void ActiveGameplaySeedBankReadinessUsesRuntimeBank()
    {
        var rotatedBank = new Dictionary<int, int>
        {
            [1] = 2,
            [0] = 1,
            [3] = 1
        };
        Check(
            BridgeObservationHelpers.IsActiveGameplaySeedBankReady(4, rotatedBank),
            "a non-empty rotated runtime bank must satisfy gameplay readiness");
        Check(
            !BridgeObservationHelpers.IsActiveGameplaySeedBankReady(0, rotatedBank),
            "an empty runtime bank must not satisfy gameplay readiness");
        Check(
            !BridgeObservationHelpers.IsActiveGameplaySeedBankReady(4, new Dictionary<int, int>()),
            "a bank count without runtime card identities must not satisfy gameplay readiness");
    }

    private static void RawGameplayReadinessIsStructural()
    {
        Check(
            BridgeObservationHelpers.IsRawBoardGameplayReady(
                boardFound: true,
                createPlantFound: true,
                boardStartMove: true,
                done: false),
            "a running board must satisfy structural gameplay readiness");
        Check(
            !BridgeObservationHelpers.IsRawBoardGameplayReady(
                boardFound: true,
                createPlantFound: true,
                boardStartMove: false,
                done: false),
            "a board that has not started must not satisfy gameplay readiness");
        Check(
            !BridgeObservationHelpers.IsRawBoardGameplayReady(
                boardFound: true,
                createPlantFound: true,
                boardStartMove: true,
                done: true),
            "a terminal board must not satisfy gameplay readiness");
    }

    private static void DuplicateBoardGateRequiresSingleton()
    {
        Check(!BridgeObservationHelpers.HasDuplicateActiveBoards(0), "missing board must not be classified as a duplicate");
        Check(!BridgeObservationHelpers.HasDuplicateActiveBoards(1), "one active board must pass the singleton gate");
        Check(BridgeObservationHelpers.HasDuplicateActiveBoards(2), "two active boards must block gameplay");
        Check(BridgeObservationHelpers.HasDuplicateActiveBoards(99), "any board count above one must block gameplay");
    }

    private static void GameplaySpeedTargetsMatchNativeSemantics()
    {
        var transition = BridgeObservationHelpers.ResolveGameSpeedTargets(
            mode: "game_speed",
            requestedGameSpeed: 4f,
            originalGameSpeed: 1f,
            originalTimeScale: 1f,
            originalFixedDeltaTime: 0.02f,
            currentTimeScale: 1f,
            gameplayReady: false);
        Check(transition.GameSpeed == 1f, "transitions must retain native game speed");
        Check(transition.TimeScale == 1f, "transitions must retain native time scale");
        Check(transition.FixedDeltaTime == 0.02f, "transitions must retain native fixed delta");

        var gameplay = BridgeObservationHelpers.ResolveGameSpeedTargets(
            mode: "game_speed",
            requestedGameSpeed: 4f,
            originalGameSpeed: 1f,
            originalTimeScale: 1f,
            originalFixedDeltaTime: 0.02f,
            currentTimeScale: 1f,
            gameplayReady: true);
        Check(gameplay.GameSpeed == 4f, "ready gameplay must store the configured multiplier");
        Check(gameplay.TimeScale == 4f, "ready gameplay must apply the multiplier to simulation time");
        Check(gameplay.FixedDeltaTime == 0.02f, "native game-speed mode must preserve fixed delta");

        var drifted = BridgeObservationHelpers.ResolveGameSpeedTargets(
            mode: "game_speed",
            requestedGameSpeed: 4f,
            originalGameSpeed: 1f,
            originalTimeScale: 1f,
            originalFixedDeltaTime: 0.02f,
            currentTimeScale: 1f,
            gameplayReady: true);
        Check(drifted.TimeScale == 4f, "gameplay drift to 1x must target 4x again");

        var paused = BridgeObservationHelpers.ResolveGameSpeedTargets(
            mode: "game_speed",
            requestedGameSpeed: 4f,
            originalGameSpeed: 1f,
            originalTimeScale: 1f,
            originalFixedDeltaTime: 0.02f,
            currentTimeScale: 0f,
            gameplayReady: true);
        Check(paused.GameSpeed == 4f, "pause must retain the configured multiplier");
        Check(paused.TimeScale == 0f, "speed enforcement must not unpause the game");
    }

    private static void ProtectedActionCountRemainsStable()
    {
        Check(
            BridgeMod.MaintainedActionCount == 841,
            "the bridge must retain the full-Adventure 841-action surface");
    }

    private static void AdventureScreenClassificationPrioritizesLoss()
    {
        string Classify(
            bool lossDetected,
            bool gameplayReady,
            bool rewardActive = false,
            bool seedSelectionActive = false) =>
            BridgeObservationHelpers.ClassifyAdventureScreenState(
                boardFound: true,
                startupPopupVisible: false,
                lossDetected,
                seedSelectionActive,
                gameplayReady,
                rewardActive,
                adventureVisible: false,
                mainMenuVisible: false);

        Check(
            Classify(lossDetected: true, gameplayReady: true) == "game_over",
            "loss/restart evidence must outrank stale gameplay readiness");
        Check(
            Classify(lossDetected: true, gameplayReady: true, rewardActive: true) == "game_over",
            "loss/restart evidence must outrank stale reward evidence");
        Check(
            Classify(lossDetected: false, gameplayReady: true) == "gameplay",
            "ordinary ready boards must remain gameplay");
        Check(
            Classify(
                lossDetected: false,
                gameplayReady: false,
                rewardActive: true,
                seedSelectionActive: true) == "seed_selection",
            "confirmed seed selection must outrank stale reward hints");
    }

    private static void MainMenuAlmanacIsNotUnlockContext()
    {
        Check(
            BridgeObservationHelpers.IsMainMenuControlPath(
                "MainMenuCanvas/MainMenu(Clone)/Grave/Btn/Almanac"),
            "the main-menu Almanac control must be recognized as navigation UI");
        Check(
            !BridgeObservationHelpers.IsMainMenuControlPath(
                "CanvasUp/Award/NewPlant/SeedPacket"),
            "post-win reward UI must not be classified as a main-menu control");
    }

    private static void SeedSelectionAlmanacIsNotUnlockContext()
    {
        Check(
            BridgeObservationHelpers.IsSeedSelectionControlPath(
                "CanvasUp/InGameUI(Clone)/Bottom/SeedLibrary/LookAlmanac/text"),
            "the seed chooser Almanac control must be recognized as selection UI");
        Check(
            !BridgeObservationHelpers.IsSeedSelectionControlPath(
                "CanvasUp/Award/NewPlant/SeedPacket"),
            "post-win seed-packet UI must not be classified as a seed chooser control");
    }

    private static PendingRequest NewRequest(long requestId, int timeoutMilliseconds = 1000) =>
        new(
            requestId,
            "{\"command\":\"step\"}",
            FakeStart,
            FakeFrequency,
            TimeSpan.FromMilliseconds(timeoutMilliseconds));

    private static BridgeResponse Timeout() =>
        BridgeResponse.Fail("timeout", "Unity main thread did not process the request in time.");

    private static BridgeResponse ServerStopping() =>
        BridgeResponse.Fail("server_stopping", "PvZRLBridge is stopping and cannot accept this request.");

    private static void Check(bool condition, string message)
    {
        _checks++;
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}
