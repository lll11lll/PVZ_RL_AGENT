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
    private ObservationDto BuildObservation(bool includeDebugArrays = false, bool forceSeedProbe = false, bool forceRestartProbe = false)
    {
        var observeWatch = Stopwatch.StartNew();
        var seedProbeMs = 0.0;
        var uiScanMs = 0.0;
        var board = FindBoard();
        var createPlant = FindCreatePlant();
        var restartInfo = DetectRestartScreenInfo(forceRestartProbe || board == null);
        var obs = new ObservationDto
        {
            BoardFound = board != null,
            CreatePlantFound = createPlant != null,
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
            var orderedSeedSlots = obs.SeedSlots.OrderBy(slot => slot.SlotIndex).ToList();
            RefreshCompatibilityFieldsFromSeedSlots(obs, orderedSeedSlots);
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

            AddLegalActions(obs, orderedSeedSlots, createPlant);
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

    private void RefreshCompatibilityFieldsFromSeedSlots(
        ObservationDto obs,
        IReadOnlyList<SeedSlotDto> orderedSeedSlots)
    {
        if (orderedSeedSlots.Count == 0)
        {
            return;
        }

        for (var index = 0; index < orderedSeedSlots.Count; index++)
        {
            var slot = orderedSeedSlots[index];
            var plantCost = new PlantCostDto
            {
                PlantType = slot.PlantType,
                PlantTypeName = slot.PlantTypeName,
                Cost = slot.SeedCost,
                Source = $"seed_slot[{slot.SlotIndex}]/{slot.Source}"
            };
            if (index < obs.PlantCosts.Count)
            {
                obs.PlantCosts[index] = plantCost;
            }
            else
            {
                obs.PlantCosts.Add(plantCost);
            }

            var cooldown = SlotCooldownFromDto(slot);
            cooldown.Source = $"seed_slot[{slot.SlotIndex}]/{slot.Source}";
            if (index < obs.CardCooldowns.Count)
            {
                obs.CardCooldowns[index] = cooldown;
            }
            else
            {
                obs.CardCooldowns.Add(cooldown);
            }
        }

        if (obs.PlantCosts.Count > orderedSeedSlots.Count)
        {
            obs.PlantCosts.RemoveRange(
                orderedSeedSlots.Count,
                obs.PlantCosts.Count - orderedSeedSlots.Count);
        }
        if (obs.CardCooldowns.Count > orderedSeedSlots.Count)
        {
            obs.CardCooldowns.RemoveRange(
                orderedSeedSlots.Count,
                obs.CardCooldowns.Count - orderedSeedSlots.Count);
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
        obs.Lanes.AddRange(
            BridgeObservationHelpers.BuildLaneSummaries(obs.Zombies, obs.RowCount));
    }

    private void AddLegalActions(
        ObservationDto obs,
        IReadOnlyList<SeedSlotDto> orderedSeedSlots,
        CreatePlant? createPlant)
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

        if (createPlant == null)
        {
            return;
        }

        var size = obs.RowCount * obs.ColumnCount;
        var occupiedCellKeys = BridgeObservationHelpers.BuildOccupiedCellKeys(
            obs,
            obs.RowCount,
            obs.ColumnCount);
        foreach (var slot in orderedSeedSlots)
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
                    if (occupiedCellKeys.Contains(
                            BridgeObservationHelpers.CellKey(
                                row,
                                column,
                                obs.ColumnCount)))
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
        obs.VisiblePlants.Any(p =>
            p.ActiveInHierarchy &&
            p.InBoardBounds &&
            p.Row == row &&
            p.Column == column);

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
        var hasFallback = GeneratedPlantRegistry.TryGetBridgeFallbackCost(plantTypeId, out var fallback);

        return new PlantCostInfo
        {
            PlantType = plantTypeId,
            Cost = fallback,
            Source = hasFallback
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
        if (seedSlotIndex < 0)
        {
            return false;
        }

        if (!_seedRuntimeCache.Valid || _seedRuntimeCache.SeedSelectionActive)
        {
            try { BuildSeedProbe(); }
            catch { return false; }
        }

        if (TryBuildCachedSeedSlotForPlacement(seedSlotIndex, out slot, out card))
        {
            return true;
        }

        try { BuildSeedProbe(); }
        catch { return false; }
        return TryBuildCachedSeedSlotForPlacement(seedSlotIndex, out slot, out card);
    }

    private bool TryBuildCachedSeedSlotForPlacement(int seedSlotIndex, out SeedSlotDto slot, out CardUI? card)
    {
        slot = new SeedSlotDto { SlotIndex = seedSlotIndex, PlantType = -1, PlantTypeName = "unknown" };
        card = null;
        if (!_seedRuntimeCache.CachedSeedSlotsByIndex.TryGetValue(
                seedSlotIndex,
                out var entry))
        {
            return false;
        }

        var resolvedCard = entry.Card;
        try
        {
            if (resolvedCard == null ||
                resolvedCard.gameObject == null ||
                !resolvedCard.gameObject.activeInHierarchy ||
                resolvedCard.GetInstanceID() != entry.CardInstanceId)
            {
                return false;
            }

            slot = BuildSeedSlotDto(resolvedCard, seedSlotIndex, "cached_active_gameplay_card", includeHierarchyPath: false);
            card = resolvedCard;
            return true;
        }
        catch
        {
            slot = new SeedSlotDto { SlotIndex = seedSlotIndex, PlantType = -1, PlantTypeName = "unknown" };
            card = null;
            return false;
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
}
