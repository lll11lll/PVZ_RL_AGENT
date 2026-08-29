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
    private SeedProbeDto BuildSeedProbe()
    {
        var seedProbeWatch = Stopwatch.StartNew();
        var board = FindBoard();
        var initBoard = FindInitBoard();
        var cardReferences = new List<CardUI>();
        var cards = ScanSeedCards(cardReferences);
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
        UpdateSeedRuntimeCache(probe, cardReferences);
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

            // The configured plant list is only the startup compatibility
            // projection. The caller separately requires a non-empty runtime
            // CardUI bank, which is authoritative after curriculum rotation.
            return BridgeObservationHelpers.IsRawBoardGameplayReady(
                boardFound: true,
                createPlantFound: true,
                boardStartMove: board.startMove,
                done: false);
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

    private void UpdateSeedRuntimeCache(
        SeedProbeDto probe,
        IReadOnlyList<CardUI> cardReferences)
    {
        var sortedSlotCards = SortSeedSlotCards(probe.ActiveGameplayCardBankCards);
        var activeGameplayCounts = new Dictionary<int, int>();
        BridgeObservationHelpers.PopulateSeedCompatibilityCollections(
            sortedSlotCards,
            activeGameplayCounts,
            _seedRuntimeCache.CachedPlantCosts);
        // The configured plant types are the startup model contract, not the
        // active dynamic curriculum loadout. The active CardUI bank is the
        // runtime readiness source; Python validates the requested identities.
        var activeGameplaySeedBankReady = BridgeObservationHelpers.IsActiveGameplaySeedBankReady(
            probe.ActiveGameplayCardBankCards.Count,
            activeGameplayCounts);
        _seedRuntimeCache.Valid = true;
        _seedRuntimeCache.SeedSelectionActive = probe.SeedSelectionActive;
        _seedRuntimeCache.SeedSelectionPanelActive = probe.SeedSelectionPanelActive;
        _seedRuntimeCache.StartButtonActive = probe.StartButtonActive;
        _seedRuntimeCache.BlockingRewardUiActive = probe.BlockingRewardUiActive;
        _seedRuntimeCache.GameplayReady = probe.GameplayReady;
        _seedRuntimeCache.ActualGameplayReady = probe.GameplayReady &&
                                                 !probe.SeedSelectionActive &&
                                                 !probe.BlockingRewardUiActive &&
                                                 activeGameplaySeedBankReady;
        _seedRuntimeCache.ActiveGameplayCardBankCount = probe.ActiveGameplayCardBankCards.Count;
        _seedRuntimeCache.ActiveGameplayTypeCounts.Clear();
        foreach (var pair in activeGameplayCounts)
        {
            _seedRuntimeCache.ActiveGameplayTypeCounts[pair.Key] = pair.Value;
        }

        _seedRuntimeCache.CachedSeedSlotDtos.Clear();
        for (var i = 0; i < sortedSlotCards.Count; i++)
        {
            _seedRuntimeCache.CachedSeedSlotDtos.Add(BuildSeedSlotDto(sortedSlotCards[i], i, "active_gameplay_card_bank"));
        }

        RefreshCachedGameplayCardRefs(sortedSlotCards, cardReferences);
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
        _seedRuntimeCache.CachedSeedSlotsByIndex.Clear();
        _seedRuntimeCache.CachedSeedSlotDtos.Clear();
        _observationsSinceSeedProbe = Math.Max(_observationsSinceSeedProbe, _config.SeedScreenCheckInterval);
    }

    private void RefreshCachedGameplayCardRefs(
        List<SeedCardDto> sortedSlotCards,
        IReadOnlyList<CardUI> cardReferences)
    {
        _seedRuntimeCache.CachedGameplayCards.Clear();
        _seedRuntimeCache.CachedSeedSlots.Clear();
        _seedRuntimeCache.CachedSeedSlotsByIndex.Clear();
        var activeGameplayCardIds = new HashSet<int>(sortedSlotCards.Select(card => card.InstanceId));
        if (activeGameplayCardIds.Count == 0)
        {
            return;
        }

        var cardsById = new Dictionary<int, CardUI>();
        foreach (var card in cardReferences)
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

        for (var i = 0; i < sortedSlotCards.Count; i++)
        {
            if (!cardsById.TryGetValue(sortedSlotCards[i].InstanceId, out var card))
            {
                continue;
            }

            var entry = new SeedSlotCacheEntry
            {
                SlotIndex = i,
                CardInstanceId = sortedSlotCards[i].InstanceId,
                Card = card
            };
            _seedRuntimeCache.CachedSeedSlots.Add(entry);
            _seedRuntimeCache.CachedSeedSlotsByIndex[i] = entry;
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

        var refreshedCount = 0;
        foreach (var entry in _seedRuntimeCache.CachedSeedSlots)
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

                var refreshed = BuildSeedSlotDto(
                    card,
                    entry.SlotIndex,
                    "cached_active_gameplay_card_live",
                    includeHierarchyPath: false);
                if (refreshedCount < _seedRuntimeCache.CachedSeedSlotDtos.Count)
                {
                    _seedRuntimeCache.CachedSeedSlotDtos[refreshedCount] = refreshed;
                }
                else
                {
                    _seedRuntimeCache.CachedSeedSlotDtos.Add(refreshed);
                }
                refreshedCount++;
            }
            catch
            {
                InvalidateSeedRuntimeCache("cached seed slot refresh failed");
                return;
            }
        }

        if (refreshedCount == 0)
        {
            return;
        }

        if (_seedRuntimeCache.CachedSeedSlotDtos.Count > refreshedCount)
        {
            _seedRuntimeCache.CachedSeedSlotDtos.RemoveRange(
                refreshedCount,
                _seedRuntimeCache.CachedSeedSlotDtos.Count - refreshedCount);
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
        // Compatibility key retained for Python diagnostics. Compensation is deprecated,
        // so the counter intentionally remains a serialized constant zero.
        obs.SunSpawnCompensationApplyCount = DeprecatedSunSpawnCompensationApplyCount;
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
            ArmBoardSingletonCheck("lets_rock");
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
                ArmBoardSingletonCheck("lets_rock_fallback");
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

    private List<SeedCardDto> ScanSeedCards(
        List<CardUI>? cardReferences = null)
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
                    cardReferences?.Add(card);
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

    internal static Dictionary<int, int> BuildTypeCounts(IEnumerable<int> plantTypes)
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
        if (TryInvokeTypedStartGameButton(gameObject, actions, out methodUsed))
        {
            return true;
        }

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

    private bool TryInvokeTypedStartGameButton(
        GameObject startObject,
        List<string> actions,
        out string methodUsed)
    {
        methodUsed = "";
        try
        {
            var startPath = BuildHierarchyPath(startObject.transform);
            var candidates = Object.FindObjectsOfType<StartGameBtn>()
                .Where(button =>
                    button != null &&
                    button.gameObject != null &&
                    button.gameObject.activeInHierarchy &&
                    (button.gameObject.GetInstanceID() == startObject.GetInstanceID() ||
                     string.Equals(
                         BuildHierarchyPath(button.transform),
                         startPath,
                         StringComparison.OrdinalIgnoreCase)))
                .ToList();

            foreach (var button in candidates)
            {
                var targetPath = BuildHierarchyPath(button.transform);
                var downInvoked = TryInvokeNoArg(button, "OnMouseDown", out var downError);
                if (downInvoked)
                {
                    actions.Add($"StartGameBtn.OnMouseDown({targetPath})");
                }

                if (TryInvokeNoArg(button, "OnMouseUpAsButton", out var upError))
                {
                    methodUsed = "StartGameBtn.OnMouseUpAsButton";
                    actions.Add($"{methodUsed}({targetPath})");
                    return true;
                }
                if (TryInvokeNoArg(button, "OnMouseUp", out upError))
                {
                    methodUsed = "StartGameBtn.OnMouseUp";
                    actions.Add($"{methodUsed}({targetPath})");
                    return true;
                }
                if (downInvoked)
                {
                    methodUsed = "StartGameBtn.OnMouseDown";
                    return true;
                }

                actions.Add(
                    $"StartGameBtn candidate unavailable: {targetPath} " +
                    $"down={downError} up={upError}");
            }
            actions.Add($"StartGameBtn candidates={candidates.Count} path={startPath}");
        }
        catch (Exception ex)
        {
            actions.Add("FindObjectsOfType<StartGameBtn>() failed: " + ex.Message);
        }
        return false;
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
            // auto_select_seeds(start_level=true) uses this shared helper
            // directly, so arm the transition gate here as well as in the
            // explicit press_lets_rock_once command wrapper.
            ArmBoardSingletonCheck("auto_select_seeds_start");
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

}
