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
    private object ActionCommand(JsonElement root)
    {
        var watch = Stopwatch.StartNew();
        var action = ReadInt(root, 0, "action", "action_id", "actionId");
        var returnObservation = ReadBool(root, true, "return_observation", "returnObservation");
        var before = BuildObservation();
        var decoded = DecodeAction(action, before);
        PlacementResult? placement = null;
        var illegalAction = decoded.Kind == "invalid";
        var illegalReason = decoded.Kind == "invalid" ? decoded.Error : null;

        var lossEvidence = HasLossRestartEvidence(before) &&
                           !HasPostWinEvidence(before) &&
                           before.TerminalHint != "possible_win";
        if (decoded.Kind == "plant" && lossEvidence)
        {
            illegalAction = true;
            illegalReason = "game_over_restart_screen";
            placement = PlacementResult.Fail(
                decoded.PlantType,
                decoded.Row,
                decoded.Column,
                "Game Over / Restart UI is active; plant actions are disabled until reset completes.",
                "game_over_restart_screen");
        }
        else if (decoded.Kind == "plant" && before.SeedSelectionActive)
        {
            illegalAction = true;
            illegalReason = "seed_selection_active";
            placement = PlacementResult.Fail(
                decoded.PlantType,
                decoded.Row,
                decoded.Column,
                "Seed selection UI is active; plant actions are disabled until Let's Rock has completed.",
                "seed_selection_active");
        }
        else if (decoded.Kind == "plant")
        {
            placement = TryPlaceSeedSlot(decoded.SeedSlotIndex, decoded.Row, decoded.Column, true, before);
            illegalAction = placement is { Success: false };
            illegalReason = placement?.IllegalReason;
        }

        var afterObservation = returnObservation ? BuildObservation() : null;
        watch.Stop();
        return new
        {
            action,
            decoded,
            legalBefore = before.LegalActions.Contains(action),
            legalActionReasonBefore = before.LegalActionReason,
            illegalAction,
            illegalReason,
            plantPlaced = placement?.PlantPlaced ?? false,
            costPaid = placement?.CostPaid ?? false,
            cooldownStarted = placement?.CooldownStarted ?? false,
            plantCost = placement?.PlantCost,
            costSource = placement?.CostSource,
            cardCooldown = placement?.CardCooldown,
            sunBefore = placement?.SunBefore,
            sunAfter = placement?.SunAfter,
            placement,
            observation = afterObservation,
            step_ms = _config.DebugPerformance ? Math.Round(watch.Elapsed.TotalMilliseconds, 3) : 0.0,
            observe_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.observe_ms : 0.0,
            bridge_observe_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.bridge_observe_ms : 0.0,
            screen_check_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.screen_check_ms : before.screen_check_ms,
            seed_probe_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.seed_probe_ms : 0.0,
            ui_scan_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.ui_scan_ms : 0.0,
            restartDetectionMode = afterObservation?.RestartDetectionMode ?? before.RestartDetectionMode
        };
    }

    private object FusionProbeCommand(JsonElement root)
    {
        var watch = Stopwatch.StartNew();
        var before = BuildObservation(forceSeedProbe: true);
        var candidates = new List<FusionCandidateDto>();
        var sourceRow = ReadNullableInt(root, "source_row", "sourceRow");
        var sourceCol = ReadNullableInt(root, "source_col", "sourceCol", "source_column", "sourceColumn");
        var sourcePlantType = ReadNullableInt(root, "source_plant_type", "sourcePlantType");
        var ingredientSeedSlotIndex = ReadNullableInt(root, "ingredient_seed_slot_index", "ingredientSeedSlotIndex", "seed_slot_index", "seedSlotIndex");
        foreach (var source in before.Plants)
        {
            if (sourceRow.HasValue && source.Row != sourceRow.Value)
            {
                continue;
            }
            if (sourceCol.HasValue && source.Column != sourceCol.Value)
            {
                continue;
            }
            if (sourcePlantType.HasValue && source.Type != sourcePlantType.Value)
            {
                continue;
            }
            foreach (var slot in before.SeedSlots.OrderBy(slot => slot.SlotIndex))
            {
                if (ingredientSeedSlotIndex.HasValue && slot.SlotIndex != ingredientSeedSlotIndex.Value)
                {
                    continue;
                }
                candidates.Add(ProbeFusionCandidate(before, source, slot));
            }
        }

        watch.Stop();
        return new
        {
            fusionAvailable = candidates.Any(candidate => candidate.FusionLegal),
            fusionCandidateCount = candidates.Count,
            fusionCandidates = candidates,
            gameplayReady = before.GameplayReady,
            seedSelectionActive = before.SeedSelectionActive,
            screenState = before.ScreenState,
            step_ms = _config.DebugPerformance ? Math.Round(watch.Elapsed.TotalMilliseconds, 3) : 0.0
        };
    }

    private object FusionStepCommand(JsonElement root)
    {
        var watch = Stopwatch.StartNew();
        var returnObservation = ReadBool(root, true, "return_observation", "returnObservation");
        var sourceInstanceId = ReadInt(root, 0, "source_instance_id", "sourceInstanceId");
        var sourceRow = ReadInt(root, -1, "source_row", "sourceRow");
        var sourceCol = ReadInt(root, -1, "source_col", "sourceCol", "source_column", "sourceColumn");
        var sourcePlantType = ReadInt(root, -1, "source_plant_type", "sourcePlantType");
        var ingredientSeedSlotIndex = ReadInt(root, -1, "ingredient_seed_slot_index", "ingredientSeedSlotIndex", "seed_slot_index", "seedSlotIndex");
        var ingredientPlantType = ReadInt(root, -1, "ingredient_plant_type", "ingredientPlantType", "target_plant_type", "targetPlantType");
        var requestedPredictedResultType = ReadInt(root, -1, "predicted_result_type", "predictedResultType");
        var requestedPredictedResultName = ReadString(root, "predicted_result_name", "predictedResultName") ?? "";
        var coachCommandId = ReadNullableInt(root, "coach_command_id", "coachCommandId", "executed_coach_command_id", "executedCoachCommandId");
        var executedFromFreshCoachCommand = ReadBool(root, false, "executed_from_fresh_coach_command", "executedFromFreshCoachCommand");
        var coachCommandAgeSeconds = ReadNullableDouble(root, "coach_command_age_seconds", "coachCommandAgeSeconds");
        var startupCommandBlocked = ReadBool(root, false, "startup_command_blocked", "startupCommandBlocked");
        var coachCommandQueueClearedOnReset = ReadBool(root, false, "coach_command_queue_cleared_on_reset", "coachCommandQueueClearedOnReset");

        var effectivePredictedResultType = requestedPredictedResultType;
        var effectivePredictedResultName = requestedPredictedResultName;
        var predictedResultResolutionSource = requestedPredictedResultType > 0 ? "request_payload" : "unresolved";
        var mixLookupFound = false;
        var mixLookupKey = "";

        var before = BuildObservation(forceSeedProbe: true);
        var rejection = ValidateFusionState(before);
        PlacementResult? placement = null;
        var beforeTilePlants = CollectPlantsOnTile(before, sourceRow, sourceCol);
        var sourceTileOccupiedBefore = beforeTilePlants.Count > 0;
        var plantCountOnTileBefore = beforeTilePlants.Count;
        var sourcePlantBefore = beforeTilePlants.FirstOrDefault();
        var candidate = new FusionCandidateDto
        {
            SourceInstanceId = sourceInstanceId,
            SourcePlantType = sourcePlantType,
            SourcePlantName = ((PlantType)sourcePlantType).ToString(),
            SourceRow = sourceRow,
            SourceCol = sourceCol,
            IngredientSeedSlotIndex = ingredientSeedSlotIndex,
            IngredientPlantType = ingredientPlantType,
            IngredientPlantName = ingredientPlantType >= 0 ? ((PlantType)ingredientPlantType).ToString() : "",
            PredictedResultType = effectivePredictedResultType,
            PredictedResultName = effectivePredictedResultName,
            PredictedResultResolutionSource = predictedResultResolutionSource,
            MixLookupFound = mixLookupFound,
            MixLookupKey = mixLookupKey
        };

        if (string.IsNullOrWhiteSpace(rejection))
        {
            var source = FindSourcePlant(before, sourceInstanceId, sourceRow, sourceCol, sourcePlantType);
            if (source == null)
            {
                rejection = "source_not_found";
            }
            else if (!TryGetSeedSlotForPlacement(ingredientSeedSlotIndex, out var slot, out var card))
            {
                rejection = "target_not_available";
            }
            else if (ingredientPlantType >= 0 && slot.PlantType != ingredientPlantType)
            {
                rejection = "target_not_available";
            }
            else
            {
                candidate = ProbeFusionCandidate(before, source, slot);
                mixLookupFound = candidate.MixLookupFound;
                mixLookupKey = !string.IsNullOrWhiteSpace(candidate.MixLookupKey)
                    ? candidate.MixLookupKey ?? ""
                    : BuildMixLookupKey(source.Type, slot.PlantType);
                if (requestedPredictedResultType > 0)
                {
                    effectivePredictedResultType = requestedPredictedResultType;
                    effectivePredictedResultName = !string.IsNullOrWhiteSpace(requestedPredictedResultName)
                        ? requestedPredictedResultName
                        : ResolvePlantTypeName(requestedPredictedResultType);
                    predictedResultResolutionSource = "request_payload";
                }
                else
                {
                    effectivePredictedResultType = candidate.PredictedResultType;
                    effectivePredictedResultName = !string.IsNullOrWhiteSpace(candidate.PredictedResultName)
                        ? candidate.PredictedResultName ?? ""
                        : requestedPredictedResultName;
                    predictedResultResolutionSource = !string.IsNullOrWhiteSpace(candidate.PredictedResultResolutionSource)
                        ? candidate.PredictedResultResolutionSource ?? ""
                        : "probe_unresolved";
                }
                candidate.PredictedResultType = effectivePredictedResultType;
                candidate.PredictedResultName = effectivePredictedResultName;
                candidate.PredictedResultResolutionSource = predictedResultResolutionSource;
                candidate.MixLookupFound = mixLookupFound;
                candidate.MixLookupKey = mixLookupKey;
                if (!candidate.FusionLegal)
                {
                    rejection = candidate.FusionBlockedReason ?? "bridge_rejected";
                }
                else
                {
                    placement = TryFuseSeedSlot(
                        slot,
                        card,
                        source,
                        before,
                        new FusionPredictionInfo
                        {
                            PredictedResultType = effectivePredictedResultType,
                            PredictedResultName = effectivePredictedResultName,
                            PredictedResultResolutionSource = predictedResultResolutionSource,
                            MixLookupFound = mixLookupFound,
                            MixLookupKey = mixLookupKey
                        });
                    if (placement is { Success: false })
                    {
                        rejection = placement.IllegalReason ?? "bridge_rejected";
                    }
                }
            }
        }

        var afterObservation = returnObservation ? BuildObservation() : null;
        if (placement is { Success: true } && effectivePredictedResultType > 0)
        {
            var predictedMatched = placement.ResultingPlantAfter != null && placement.ResultingPlantAfter.PlantType == effectivePredictedResultType;
            if (!predictedMatched)
            {
                rejection = "fusion_result_mismatch";
                placement.Success = false;
                placement.IllegalReason = rejection;
                placement.Message = "Fusion step did not produce the predicted result plant type at the source tile.";
                placement.BridgeResultReason = rejection;
            }
        }
        watch.Stop();
        var success = placement?.Success ?? false;
        var fusionExecutionMode = placement?.FusionExecutionMode ?? "dedicated_fusion";
        var sourceTileOccupiedBeforeValue = placement?.SourceTileOccupiedBefore ?? sourceTileOccupiedBefore;
        var plantCountOnTileBeforeValue = placement?.PlantCountOnTileBefore ?? plantCountOnTileBefore;
        var plantCountOnTileAfterValue = placement?.PlantCountOnTileAfter ?? plantCountOnTileBeforeValue;
        var sourcePlantBeforeValue = placement?.SourcePlantBefore ?? sourcePlantBefore;
        var resultingPlantAfter = placement?.ResultingPlantAfter;
        var duplicateStackDetected = placement?.DuplicateStackDetected ?? false;
        var bridgeMethodUsed = placement?.BridgeMethodUsed ?? "";
        var bridgeResultReason = placement?.BridgeResultReason ?? (success ? "success" : (rejection ?? "bridge_rejected"));
        var predictedResultResolutionSourceValue = placement?.PredictedResultResolutionSource ?? predictedResultResolutionSource;
        var mixLookupFoundValue = placement?.MixLookupFound ?? mixLookupFound;
        var mixLookupKeyValue = placement?.MixLookupKey ?? mixLookupKey;
        var preSourceTypeValue = placement?.PreSourceType ?? sourcePlantBeforeValue?.PlantType ?? sourcePlantType;
        var preSourceNameValue = placement?.PreSourceName
            ?? sourcePlantBeforeValue?.PlantTypeName
            ?? ResolvePlantTypeName(preSourceTypeValue);
        var ingredientTypeValue = placement?.IngredientType ?? ingredientPlantType;
        var ingredientNameValue = placement?.IngredientName ?? ResolvePlantTypeName(ingredientTypeValue);
        var postResultTypeValue = placement?.PostResultType ?? resultingPlantAfter?.PlantType ?? -1;
        var postResultNameValue = placement?.PostResultName
            ?? resultingPlantAfter?.PlantTypeName
            ?? ResolvePlantTypeName(postResultTypeValue);
        var noEffectReasonValue = placement?.NoEffectReason ?? "";
        var requestedSourceRowValue = placement?.RequestedSourceRow ?? sourceRow;
        var requestedSourceColValue = placement?.RequestedSourceCol ?? sourceCol;
        var requestedSourceInstanceIdValue = placement?.RequestedSourceInstanceId ?? sourceInstanceId;
        var changedTileCountValue = placement?.ChangedTileCount ?? 0;
        var changedTilesValue = placement?.ChangedTiles ?? new List<FusionTileChangeDto>();
        var nonSourceTilesChangedValue = placement?.NonSourceTilesChanged ?? false;
        var globalFusionSideEffectValue = placement?.GlobalFusionSideEffect ?? false;
        var fusionScopeValue = placement?.FusionScope ?? "unknown";

        return new
        {
            action = -1,
            decoded = new
            {
                kind = "fusion",
                sourcePlantType,
                ingredientPlantType,
                resultPlantType = effectivePredictedResultType,
                row = sourceRow,
                column = sourceCol
            },
            fusionAttempted = placement != null,
            fusionSucceeded = success,
            fusionRejectedReason = success ? null : rejection,
            illegalAction = !success,
            illegalReason = success ? null : rejection,
            plantPlaced = placement?.PlantPlaced ?? false,
            costPaid = placement?.CostPaid ?? false,
            cooldownStarted = placement?.CooldownStarted ?? false,
            plantCost = placement?.PlantCost,
            costSource = placement?.CostSource,
            cardCooldown = placement?.CardCooldown,
            sunBefore = placement?.SunBefore,
            sunAfter = placement?.SunAfter,
            fusionExecutionMode = fusionExecutionMode,
            sourceTileOccupiedBefore = sourceTileOccupiedBeforeValue,
            source_tile_occupied_before = sourceTileOccupiedBeforeValue,
            plantCountOnTileBefore = plantCountOnTileBeforeValue,
            plantCountOnTileAfter = plantCountOnTileAfterValue,
            sourcePlantBefore = sourcePlantBeforeValue,
            source_plant_before = sourcePlantBeforeValue,
            resultingPlantAfter,
            resulting_plant_after = resultingPlantAfter,
            duplicateStackDetected,
            duplicate_stack_detected = duplicateStackDetected,
            bridgeMethodUsed,
            bridgeResultReason,
            predictedResultResolutionSource = predictedResultResolutionSourceValue,
            mixLookupFound = mixLookupFoundValue,
            mixLookupKey = mixLookupKeyValue,
            preSourceType = preSourceTypeValue,
            preSourceName = preSourceNameValue,
            ingredientType = ingredientTypeValue,
            ingredientName = ingredientNameValue,
            postResultType = postResultTypeValue,
            postResultName = postResultNameValue,
            noEffectReason = noEffectReasonValue,
            requestedSourceRow = requestedSourceRowValue,
            requestedSourceCol = requestedSourceColValue,
            requestedSourceInstanceId = requestedSourceInstanceIdValue,
            changedTileCount = changedTileCountValue,
            changedTiles = changedTilesValue,
            nonSourceTilesChanged = nonSourceTilesChangedValue,
            globalFusionSideEffect = globalFusionSideEffectValue,
            fusionScope = fusionScopeValue,
            requested_source_row = requestedSourceRowValue,
            requested_source_col = requestedSourceColValue,
            requested_source_instance_id = requestedSourceInstanceIdValue,
            changed_tile_count = changedTileCountValue,
            changed_tiles = changedTilesValue,
            non_source_tiles_changed = nonSourceTilesChangedValue,
            global_fusion_side_effect = globalFusionSideEffectValue,
            fusion_scope = fusionScopeValue,
            bridge_method_used = bridgeMethodUsed,
            bridge_result_reason = bridgeResultReason,
            executed_from_fresh_coach_command = executedFromFreshCoachCommand,
            coach_command_age_seconds = coachCommandAgeSeconds,
            startup_command_blocked = startupCommandBlocked,
            coach_command_queue_cleared_on_reset = coachCommandQueueClearedOnReset,
            executed_coach_command_id = coachCommandId,
            sourceInstanceId,
            sourceRow,
            sourceCol,
            sourcePlantType,
            ingredientSeedSlotIndex,
            ingredientPlantType,
            predictedResultType = effectivePredictedResultType,
            predictedResultName = effectivePredictedResultName,
            fusionCandidate = candidate,
            placement,
            observation = afterObservation,
            step_ms = _config.DebugPerformance ? Math.Round(watch.Elapsed.TotalMilliseconds, 3) : 0.0,
            observe_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.observe_ms : 0.0,
            bridge_observe_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.bridge_observe_ms : 0.0,
            screen_check_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.screen_check_ms : before.screen_check_ms,
            seed_probe_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.seed_probe_ms : 0.0,
            ui_scan_ms = _config.DebugPerformance && afterObservation != null ? afterObservation.ui_scan_ms : 0.0,
            restartDetectionMode = afterObservation?.RestartDetectionMode ?? before.RestartDetectionMode
        };
    }

    private string ValidateFusionState(ObservationDto obs)
    {
        if (obs.Done || obs.Over)
        {
            return "terminal_or_transition_state";
        }
        if (obs.SeedSelectionActive || obs.ScreenState == "seed_selection")
        {
            return "seed_selection_active";
        }
        if (obs.BlockingRewardUiActive || obs.ScreenState is "reward_unlock" or "reward_screen" or "level_complete_trophy")
        {
            return "reward_or_unlock_screen_active";
        }
        if (!obs.BoardFound)
        {
            return "not_gameplay";
        }
        if (!obs.GameplayReady)
        {
            return "gameplay_not_ready";
        }
        return "";
    }

    private PlantDto? FindSourcePlant(ObservationDto obs, int instanceId, int row, int column, int plantType)
    {
        foreach (var plant in obs.Plants)
        {
            if (plant.Row != row || plant.Column != column || plant.Type != plantType)
            {
                continue;
            }
            if (instanceId != 0 && plant.InstanceId != 0 && plant.InstanceId != instanceId)
            {
                continue;
            }
            return plant;
        }
        return null;
    }

    private FusionCandidateDto ProbeFusionCandidate(ObservationDto obs, PlantDto source, SeedSlotDto slot)
    {
        var mixLookupKey = BuildMixLookupKey(source.Type, slot.PlantType);
        var candidate = new FusionCandidateDto
        {
            SourceInstanceId = source.InstanceId,
            SourcePlantType = source.Type,
            SourcePlantName = source.TypeName,
            SourceRow = source.Row,
            SourceCol = source.Column,
            IngredientSeedSlotIndex = slot.SlotIndex,
            IngredientPlantType = slot.PlantType,
            IngredientPlantName = slot.PlantTypeName,
            PredictedResultType = -1,
            PredictedResultName = "",
            PredictedResultResolutionSource = "probe_unresolved",
            MixLookupFound = false,
            MixLookupKey = mixLookupKey,
            FusionLegal = false,
            FusionBlockedReason = ""
        };
        var stateRejection = ValidateFusionState(obs);
        if (!string.IsNullOrWhiteSpace(stateRejection))
        {
            candidate.FusionBlockedReason = stateRejection;
            return candidate;
        }
        if (!slot.Usable || slot.Disabled || !slot.Ready || obs.Sun < slot.SeedCost)
        {
            candidate.FusionBlockedReason = "target_not_available";
            return candidate;
        }
        var createPlant = FindCreatePlant();
        if (createPlant == null)
        {
            candidate.FusionBlockedReason = "fusion_not_available";
            return candidate;
        }
        var prediction = ResolveFusionPrediction(createPlant, source, slot);
        candidate.PredictedResultType = prediction.PredictedResultType;
        candidate.PredictedResultName = prediction.PredictedResultName;
        candidate.PredictedResultResolutionSource = prediction.PredictedResultResolutionSource;
        candidate.MixLookupFound = prediction.MixLookupFound;
        candidate.MixLookupKey = prediction.MixLookupKey;
        candidate.FusionLegal = true;
        candidate.FusionBlockedReason = "";
        return candidate;
    }

    private PlacementResult TryFuseSeedSlot(
        SeedSlotDto slot,
        CardUI? card,
        PlantDto source,
        ObservationDto precheckedObservation,
        FusionPredictionInfo prediction)
    {
        var predictedResultType = prediction?.PredictedResultType ?? -1;
        var predictedResultName = prediction?.PredictedResultName ?? "";
        var predictedResultResolutionSource = !string.IsNullOrWhiteSpace(prediction?.PredictedResultResolutionSource)
            ? prediction!.PredictedResultResolutionSource
            : "unresolved";
        var mixLookupFound = prediction?.MixLookupFound ?? false;
        var mixLookupKey = prediction?.MixLookupKey ?? BuildMixLookupKey(source.Type, slot.PlantType);
        var plantTypeId = slot.PlantType;
        var board = FindBoard();
        var sourcePlantBefore = BuildTilePlantSnapshot(source);
        var preSourceType = sourcePlantBefore.PlantType;
        var preSourceName = sourcePlantBefore.PlantTypeName ?? ResolvePlantTypeName(preSourceType);
        var ingredientType = slot.PlantType;
        var ingredientName = slot.PlantTypeName ?? ResolvePlantTypeName(ingredientType);
        var beforeTilePlants = CollectPlantsOnTile(precheckedObservation, source.Row, source.Column);
        var beforeBoardTileMap = CollectBoardTilePlantMap(precheckedObservation);
        var plantCountOnTileBefore = beforeTilePlants.Count;
        var sourceTileOccupiedBefore = plantCountOnTileBefore > 0;
        var afterFallbackCount = plantCountOnTileBefore;
        var requestedSourceRow = source.Row;
        var requestedSourceCol = source.Column;
        var requestedSourceInstanceId = source.InstanceId;
        var changedTiles = new List<FusionTileChangeDto>();
        var changedTileCount = 0;
        var nonSourceTilesChanged = false;
        var globalFusionSideEffect = false;
        var fusionScope = "unknown";

        PlacementResult WithFusionDiagnostics(
            PlacementResult result,
            int plantCountOnTileAfter,
            TilePlantSnapshotDto? resultingPlantAfter,
            bool duplicateStackDetected,
            string bridgeMethodUsed,
            string bridgeResultReason,
            int changedTileCountValue = 0,
            List<FusionTileChangeDto>? changedTilesValue = null,
            bool nonSourceTilesChangedValue = false,
            bool globalFusionSideEffectValue = false,
            string fusionScopeValue = "unknown")
        {
            result.FusionExecutionMode = "dedicated_fusion";
            result.SourceTileOccupiedBefore = sourceTileOccupiedBefore;
            result.PlantCountOnTileBefore = plantCountOnTileBefore;
            result.PlantCountOnTileAfter = plantCountOnTileAfter;
            result.SourcePlantBefore = sourcePlantBefore;
            result.ResultingPlantAfter = resultingPlantAfter;
            result.DuplicateStackDetected = duplicateStackDetected;
            result.BridgeMethodUsed = bridgeMethodUsed;
            result.BridgeResultReason = bridgeResultReason;
            if (string.IsNullOrWhiteSpace(result.PredictedResultResolutionSource))
            {
                result.PredictedResultResolutionSource = predictedResultResolutionSource;
            }
            result.MixLookupFound = result.MixLookupFound || mixLookupFound;
            if (string.IsNullOrWhiteSpace(result.MixLookupKey))
            {
                result.MixLookupKey = mixLookupKey;
            }
            result.PreSourceType = preSourceType;
            result.PreSourceName = preSourceName;
            result.IngredientType = ingredientType;
            result.IngredientName = ingredientName;
            result.PostResultType = resultingPlantAfter?.PlantType ?? result.PostResultType;
            result.PostResultName = resultingPlantAfter?.PlantTypeName ?? result.PostResultName;
            result.RequestedSourceRow = requestedSourceRow;
            result.RequestedSourceCol = requestedSourceCol;
            result.RequestedSourceInstanceId = requestedSourceInstanceId;
            result.ChangedTileCount = Math.Max(0, changedTileCountValue);
            result.ChangedTiles = changedTilesValue != null
                ? new List<FusionTileChangeDto>(changedTilesValue)
                : new List<FusionTileChangeDto>();
            result.NonSourceTilesChanged = nonSourceTilesChangedValue;
            result.GlobalFusionSideEffect = globalFusionSideEffectValue;
            result.FusionScope = string.IsNullOrWhiteSpace(fusionScopeValue) ? "unknown" : fusionScopeValue;
            return result;
        }

        if (board == null)
        {
            return WithFusionDiagnostics(
                    PlacementResult.Fail(plantTypeId, source.Row, source.Column, "Board not found.", "board_not_found")
                        .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId),
                    afterFallbackCount,
                    null,
                    false,
                    "none",
                    "board_not_found");
        }
        var createPlant = FindCreatePlant();
        if (createPlant == null)
        {
            return WithFusionDiagnostics(
                    PlacementResult.Fail(plantTypeId, source.Row, source.Column, "CreatePlant.Instance not found.", "create_plant_not_found")
                        .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId),
                    afterFallbackCount,
                    null,
                    false,
                    "none",
                    "create_plant_not_found");
        }
        var stateRejection = ValidateFusionState(precheckedObservation);
        if (!string.IsNullOrWhiteSpace(stateRejection))
        {
            return WithFusionDiagnostics(
                    PlacementResult.Fail(plantTypeId, source.Row, source.Column, "Fusion blocked by unsafe state.", stateRejection)
                        .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId),
                    afterFallbackCount,
                    null,
                    false,
                    "none",
                    stateRejection);
        }
        if (!sourceTileOccupiedBefore)
        {
            return WithFusionDiagnostics(
                    PlacementResult.Fail(plantTypeId, source.Row, source.Column, "Source tile is not occupied by a plant before fusion.", "source_tile_not_occupied")
                        .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId),
                    afterFallbackCount,
                    null,
                    false,
                    "none",
                    "source_tile_not_occupied");
        }
        var costInfo = new PlantCostInfo
        {
            PlantType = plantTypeId,
            Cost = Math.Max(0, slot.SeedCost),
            Source = slot.Source ?? "seed_slot",
            Warning = slot.Warning
        };
        var sunBefore = board.theSun;
        var cooldownInfo = card != null
            ? BuildCardCooldownDto(card, plantTypeId, $"seed_slot[{slot.SlotIndex}]")
            : SlotCooldownFromDto(slot);
        if (!slot.Usable || slot.Disabled)
        {
            return WithFusionDiagnostics(
                    PlacementResult.Fail(
                            plantTypeId,
                            source.Row,
                            source.Column,
                            "Seed slot is not usable.",
                            slot.Disabled ? "slot_disabled" : "slot_not_usable",
                            costInfo,
                            sunBefore,
                            sunBefore,
                            cooldownInfo)
                        .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId),
                    afterFallbackCount,
                    null,
                    false,
                    "none",
                    slot.Disabled ? "slot_disabled" : "slot_not_usable");
        }
        if (sunBefore < costInfo.Cost)
        {
            return WithFusionDiagnostics(
                    PlacementResult.Fail(
                            plantTypeId,
                            source.Row,
                            source.Column,
                            "Insufficient sun for fusion.",
                            "target_not_available",
                            costInfo,
                            sunBefore,
                            sunBefore,
                            cooldownInfo)
                        .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId),
                    afterFallbackCount,
                    null,
                    false,
                    "none",
                    "target_not_available");
        }
        if (!cooldownInfo.Ready)
        {
            return WithFusionDiagnostics(
                    PlacementResult.Fail(
                            plantTypeId,
                            source.Row,
                            source.Column,
                            "Seed slot is on cooldown.",
                            "target_not_available",
                            costInfo,
                            sunBefore,
                            sunBefore,
                            cooldownInfo)
                        .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId),
                    afterFallbackCount,
                    null,
                    false,
                    "none",
                    "target_not_available");
        }

        var execution = TryExecuteDedicatedFusion(card, createPlant, source, slot, beforeBoardTileMap);
        ObservationDto? postFusionObservation = null;
        try
        {
            postFusionObservation = BuildObservation(forceSeedProbe: true);
        }
        catch
        {
            postFusionObservation = null;
        }
        var afterTilePlants = postFusionObservation != null
            ? CollectPlantsOnTile(postFusionObservation, source.Row, source.Column)
            : beforeTilePlants;
        var afterBoardTileMap = postFusionObservation != null
            ? CollectBoardTilePlantMap(postFusionObservation)
            : beforeBoardTileMap;
        changedTiles = ComputeChangedTiles(beforeBoardTileMap, afterBoardTileMap);
        changedTileCount = changedTiles.Count;
        nonSourceTilesChanged = changedTiles.Any(tile => tile.Row != source.Row || tile.Column != source.Column);
        globalFusionSideEffect = nonSourceTilesChanged;
        if (postFusionObservation == null)
        {
            fusionScope = "unknown";
        }
        else if (globalFusionSideEffect)
        {
            fusionScope = "global_side_effect_detected";
        }
        else
        {
            fusionScope = "tile_scoped";
        }
        var changedSourceTile = changedTiles.Any(tile => tile.Row == source.Row && tile.Column == source.Column);
        var exactlyOneSourceTileChanged = changedTileCount == 1 && changedSourceTile && !nonSourceTilesChanged;
        var plantCountOnTileAfter = afterTilePlants.Count;
        var resultingPlantAfter = ResolveResultingTilePlant(afterTilePlants, predictedResultType, source.InstanceId);
        var duplicateStackDetected = plantCountOnTileAfter > 1;
        var observableTileChange = DidFusionProduceObservableChange(
            beforeTilePlants,
            afterTilePlants,
            sourcePlantBefore,
            resultingPlantAfter);
        var postResultType = resultingPlantAfter?.PlantType ?? -1;
        var postResultName = resultingPlantAfter?.PlantTypeName ?? ResolvePlantTypeName(postResultType);
        var postPlantDiffersFromSource = resultingPlantAfter != null && (
            resultingPlantAfter.PlantType != preSourceType ||
            !PlantNamesEquivalent(resultingPlantAfter.PlantTypeName, preSourceName));
        var postMatchesDiscoveredMix = mixLookupFound &&
                                       ((predictedResultType > 0 && postResultType == predictedResultType) ||
                                        (!string.IsNullOrWhiteSpace(predictedResultName) &&
                                         PlantNamesEquivalent(postResultName, predictedResultName)));
        var inferredSuccessWithoutPrediction =
            sourceTileOccupiedBefore &&
            plantCountOnTileAfter == 1 &&
            !duplicateStackDetected &&
            resultingPlantAfter != null &&
            (postPlantDiffersFromSource || postMatchesDiscoveredMix);
        var noEffectReason = "";

        if (!execution.Success)
        {
            var failReason = string.IsNullOrWhiteSpace(execution.Reason) ? "bridge_rejected" : execution.Reason;
            var failBridgeReason = failReason;
            if (globalFusionSideEffect)
            {
                failReason = "global_fusion_side_effect";
                failBridgeReason = "fusion_mutated_non_source_tiles";
            }
            return WithFusionDiagnostics(
                    PlacementResult.Fail(
                            plantTypeId,
                            source.Row,
                            source.Column,
                            $"Dedicated fusion execution failed: {failReason}.",
                            failReason,
                            costInfo,
                            sunBefore,
                            board.theSun,
                            cooldownInfo)
                        .WithSeedSlot(slot.SlotIndex, slot.CardInstanceId),
                    plantCountOnTileAfter,
                    resultingPlantAfter,
                    duplicateStackDetected,
                    execution.MethodUsed,
                    failBridgeReason,
                    changedTileCount,
                    changedTiles,
                    nonSourceTilesChanged,
                    globalFusionSideEffect,
                    fusionScope);
        }

        var sunAfterExecution = board.theSun;
        var costPaid = TryPaySun(board, costInfo.Cost, sunBefore, out var sunAfter, out var paymentSource);
        var cooldownStarted = false;
        var cooldownSource = "not_started";
        if (costPaid && card != null)
        {
            cooldownStarted = TryStartSeedSlotCooldown(card, slot.SlotIndex, out cooldownSource);
            InvalidateSeedRuntimeCache($"fusion_seed_slot[{slot.SlotIndex}] placed");
        }

        var failureReason = "";
        var failureMessage = "";
        if (!costPaid)
        {
            failureReason = "cost_payment_failed";
            failureMessage = "Fusion executed but failed to pay sun cost.";
        }
        else if (globalFusionSideEffect)
        {
            failureReason = "global_fusion_side_effect";
            failureMessage = "Fusion mutated non-source tiles.";
            noEffectReason = "fusion_mutated_non_source_tiles";
        }
        else if (!exactlyOneSourceTileChanged)
        {
            failureReason = "fusion_no_effect";
            failureMessage = "Dedicated fusion execution completed without an isolated source-tile mutation.";
            noEffectReason = changedTileCount == 0
                ? "no_changed_tiles_detected"
                : changedSourceTile
                    ? "multiple_changed_tiles_detected"
                    : "source_tile_not_in_changed_tiles";
        }
        else if (duplicateStackDetected)
        {
            failureReason = "duplicate_stack_detected";
            failureMessage = "Fusion created duplicate plant occupancy on the source tile.";
        }
        else if (resultingPlantAfter == null)
        {
            failureReason = "source_not_found_after_fusion";
            failureMessage = "Fusion execution left no resulting source-tile plant snapshot.";
        }
        else if (predictedResultType > 0 && (resultingPlantAfter == null || resultingPlantAfter.PlantType != predictedResultType))
        {
            failureReason = "fusion_result_mismatch";
            failureMessage = "Fusion step did not produce the predicted result plant type at the source tile.";
        }
        else if (predictedResultType <= 0 && !inferredSuccessWithoutPrediction)
        {
            failureReason = "fusion_no_effect";
            failureMessage = "Dedicated fusion execution completed without any observable source-tile state change.";
            noEffectReason = BuildFusionNoEffectReason(
                sourcePlantBefore,
                resultingPlantAfter,
                plantCountOnTileAfter,
                observableTileChange,
                postMatchesDiscoveredMix);
        }
        else if (predictedResultType > 0 && !observableTileChange)
        {
            // Preserve diagnostics for predicted fusions that did resolve to the expected type but looked visually unchanged.
            noEffectReason = BuildFusionNoEffectReason(
                sourcePlantBefore,
                resultingPlantAfter,
                plantCountOnTileAfter,
                observableTileChange,
                postMatchesDiscoveredMix);
        }

        var success = string.IsNullOrEmpty(failureReason);
        var bridgeResultReason = success
            ? "success"
            : globalFusionSideEffect
                ? "fusion_mutated_non_source_tiles"
                : failureReason;
        var resultType = resultingPlantAfter?.PlantType ?? (predictedResultType > 0 ? predictedResultType : plantTypeId);
        var resultTypeName = resultingPlantAfter?.PlantTypeName ?? ResolvePlantTypeName(resultType);
        var placementResult = new PlacementResult
        {
            Success = success,
            PlantPlaced = true,
            CostPaid = costPaid,
            CooldownStarted = cooldownStarted,
            PlantCost = costInfo.Cost,
            CostSource = costInfo.Source,
            CostWarning = costInfo.Warning,
            PaymentSource = paymentSource,
            CooldownSource = cooldownSource,
            CardCooldown = card != null ? BuildCardCooldownDto(card, plantTypeId, $"seed_slot[{slot.SlotIndex}]") : SlotCooldownFromDto(slot),
            SunBefore = sunBefore,
            SunAfter = success ? sunAfter : sunAfterExecution,
            PlantType = resultType,
            PlantTypeName = resultTypeName,
            SeedSlotIndex = slot.SlotIndex,
            CardInstanceId = slot.CardInstanceId,
            Row = source.Row,
            Column = source.Column,
            Message = success
                ? $"Fusion executed through dedicated bridge path ({execution.MethodUsed})."
                : failureMessage,
            IllegalReason = success ? null : failureReason,
            PostResultType = resultType,
            PostResultName = resultTypeName,
            NoEffectReason = noEffectReason,
            FusionScope = fusionScope
        };
        if (success && resultType >= 0)
        {
            var cacheKey = BuildMixLookupKey(preSourceType, ingredientType);
            var cacheResolutionSource = predictedResultType > 0 && !string.IsNullOrWhiteSpace(predictedResultResolutionSource)
                ? predictedResultResolutionSource
                : "post_fusion_observation";
            _fusionPredictionCache[cacheKey] = new FusionPredictionInfo
            {
                PredictedResultType = resultType,
                PredictedResultName = resultTypeName,
                PredictedResultResolutionSource = cacheResolutionSource,
                MixLookupFound = true,
                MixLookupKey = cacheKey
            };
            if (string.IsNullOrWhiteSpace(placementResult.PredictedResultResolutionSource))
            {
                placementResult.PredictedResultResolutionSource = cacheResolutionSource;
            }
            placementResult.MixLookupFound = true;
            placementResult.MixLookupKey = cacheKey;
        }
        return WithFusionDiagnostics(
                placementResult,
                plantCountOnTileAfter,
                resultingPlantAfter,
                duplicateStackDetected,
                execution.MethodUsed,
                bridgeResultReason,
                changedTileCount,
                changedTiles,
                nonSourceTilesChanged,
                globalFusionSideEffect,
                fusionScope);
    }

    private FusionPredictionInfo ResolveFusionPrediction(CreatePlant createPlant, PlantDto source, SeedSlotDto slot)
    {
        var mixLookupKey = BuildMixLookupKey(source.Type, slot.PlantType);
        if (_fusionPredictionCache.TryGetValue(mixLookupKey, out var cached) && cached != null && cached.PredictedResultType >= 0)
        {
            return new FusionPredictionInfo
            {
                PredictedResultType = cached.PredictedResultType,
                PredictedResultName = cached.PredictedResultName,
                PredictedResultResolutionSource = "runtime_cache",
                MixLookupFound = true,
                MixLookupKey = mixLookupKey
            };
        }

        return new FusionPredictionInfo
        {
            PredictedResultType = -1,
            PredictedResultName = "",
            PredictedResultResolutionSource = "read_only_metadata_unavailable",
            MixLookupFound = false,
            MixLookupKey = mixLookupKey
        };
    }

    private static bool TryExtractMixResultPlantType(object mixObject, out int plantType, out string resolutionSource)
    {
        if (TryExtractPlantTypeFromObject(mixObject, "checkmix_object", out plantType, out resolutionSource))
        {
            return true;
        }

        var nestedMemberNames = new[]
        {
            "result",
            "resultPlant",
            "resultPlantData",
            "mixResult",
            "mixData",
            "target",
            "targetPlant",
            "newPlant",
            "fusionPlant",
            "plant",
            "data"
        };

        foreach (var memberName in nestedMemberNames)
        {
            if (!TryGetMemberValue(mixObject, memberName, out var nested) || nested == null)
            {
                continue;
            }

            if (TryExtractPlantTypeFromObject(nested, $"checkmix_object.{memberName}", out plantType, out resolutionSource))
            {
                return true;
            }
        }

        plantType = -1;
        resolutionSource = "";
        return false;
    }

    private static bool TryExtractPlantTypeFromObject(object target, string sourcePrefix, out int plantType, out string resolutionSource)
    {
        if (target == null)
        {
            plantType = -1;
            resolutionSource = "";
            return false;
        }

        if (TryConvertNumber(target, out var directNumber))
        {
            var directPlantType = (int)directNumber;
            if (IsLikelyPlantType(directPlantType))
            {
                plantType = directPlantType;
                resolutionSource = $"{sourcePrefix}.direct_numeric";
                return true;
            }
        }

        var namedTypeMembers = new[]
        {
            "resultPlantType",
            "resultType",
            "resultPlantID",
            "resultPlantId",
            "resultId",
            "mixPlantType",
            "targetPlantType",
            "newPlantType",
            "fusionPlantType",
            "fusedPlantType",
            "plantType",
            "plantID",
            "plantId",
            "id"
        };

        foreach (var memberName in namedTypeMembers)
        {
            if (!TryGetMemberValue(target, memberName, out var raw) || raw == null)
            {
                continue;
            }

            if (!TryConvertNumber(raw, out var rawNumber))
            {
                continue;
            }

            var candidateType = (int)rawNumber;
            if (!IsLikelyPlantType(candidateType))
            {
                continue;
            }

            plantType = candidateType;
            resolutionSource = $"{sourcePrefix}.{memberName}";
            return true;
        }

        // Fallback heuristic: inspect all numeric members likely to represent fusion outputs.
        try
        {
            var flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;
            var type = target.GetType();
            foreach (var field in type.GetFields(flags))
            {
                var name = field.Name ?? "";
                if (!LooksLikePlantTypeMemberName(name))
                {
                    continue;
                }
                var raw = field.GetValue(target);
                if (!TryConvertNumber(raw, out var numeric))
                {
                    continue;
                }
                var candidateType = (int)numeric;
                if (!IsLikelyPlantType(candidateType))
                {
                    continue;
                }
                plantType = candidateType;
                resolutionSource = $"{sourcePrefix}.{name}";
                return true;
            }

            foreach (var property in type.GetProperties(flags))
            {
                if (!property.CanRead || property.GetIndexParameters().Length > 0)
                {
                    continue;
                }
                var name = property.Name ?? "";
                if (!LooksLikePlantTypeMemberName(name))
                {
                    continue;
                }
                object? raw;
                try
                {
                    raw = property.GetValue(target, null);
                }
                catch
                {
                    continue;
                }
                if (!TryConvertNumber(raw, out var numeric))
                {
                    continue;
                }
                var candidateType = (int)numeric;
                if (!IsLikelyPlantType(candidateType))
                {
                    continue;
                }
                plantType = candidateType;
                resolutionSource = $"{sourcePrefix}.{name}";
                return true;
            }
        }
        catch
        {
            // Ignore reflection failures and report unresolved.
        }

        plantType = -1;
        resolutionSource = "";
        return false;
    }

    private static bool LooksLikePlantTypeMemberName(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            return false;
        }
        var normalized = name.Trim().ToLowerInvariant();
        if (normalized.Contains("row") || normalized.Contains("col") || normalized.Contains("lane"))
        {
            return false;
        }
        var hasPlantHint = normalized.Contains("plant");
        var hasResultHint = normalized.Contains("result") || normalized.Contains("mix") || normalized.Contains("target") || normalized.Contains("fusion") || normalized.Contains("fuse");
        var hasTypeHint = normalized.Contains("type") || normalized.Contains("id");
        return hasPlantHint && (hasResultHint || hasTypeHint);
    }

    private static bool IsLikelyPlantType(int plantType)
    {
        return plantType >= 0 && plantType < 50000;
    }

    private static string BuildMixLookupKey(int sourcePlantType, int ingredientPlantType)
    {
        return $"{sourcePlantType}+{ingredientPlantType}";
    }

    private static string ResolvePlantTypeName(int plantType)
    {
        if (plantType < 0)
        {
            return "";
        }
        var runtimeName = ((PlantType)plantType).ToString();
        return GeneratedPlantRegistry.ResolveCanonicalNameFallback(plantType, runtimeName);
    }

    private static bool PlantNamesEquivalent(string? left, string? right)
    {
        var normalizedLeft = NormalizePlantName(left);
        var normalizedRight = NormalizePlantName(right);
        if (string.IsNullOrWhiteSpace(normalizedLeft) || string.IsNullOrWhiteSpace(normalizedRight))
        {
            return false;
        }
        return string.Equals(normalizedLeft, normalizedRight, StringComparison.OrdinalIgnoreCase);
    }

    private static string NormalizePlantName(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return "";
        }
        return value.Trim().Replace(" ", "").Replace("_", "").ToLowerInvariant();
    }

    private static string BuildFusionNoEffectReason(
        TilePlantSnapshotDto? sourcePlantBefore,
        TilePlantSnapshotDto? resultingPlantAfter,
        int plantCountOnTileAfter,
        bool observableTileChange,
        bool postMatchesDiscoveredMix)
    {
        if (plantCountOnTileAfter <= 0 || resultingPlantAfter == null)
        {
            return "post_result_missing";
        }
        if (plantCountOnTileAfter > 1)
        {
            return "duplicate_stack_detected";
        }
        if (sourcePlantBefore == null)
        {
            return "source_snapshot_missing";
        }
        if (postMatchesDiscoveredMix)
        {
            return "mix_result_matched_without_identity_delta";
        }
        if (sourcePlantBefore.PlantType == resultingPlantAfter.PlantType &&
            PlantNamesEquivalent(sourcePlantBefore.PlantTypeName, resultingPlantAfter.PlantTypeName) &&
            sourcePlantBefore.InstanceId == resultingPlantAfter.InstanceId)
        {
            return "post_matches_pre_identity";
        }
        if (!observableTileChange)
        {
            return "no_observable_tile_change";
        }
        return "unknown";
    }

    private TilePlantSnapshotDto BuildTilePlantSnapshot(PlantDto plant, string source = "logical")
    {
        return new TilePlantSnapshotDto
        {
            InstanceId = plant.InstanceId,
            PlantType = plant.Type,
            PlantTypeName = plant.TypeName ?? ((PlantType)plant.Type).ToString(),
            Row = plant.Row,
            Column = plant.Column,
            Source = source
        };
    }

    private List<TilePlantSnapshotDto> CollectPlantsOnTile(ObservationDto obs, int row, int column)
    {
        var result = new List<TilePlantSnapshotDto>();
        if (obs == null || row < 0 || column < 0)
        {
            return result;
        }

        var seenInstanceIds = new HashSet<int>();
        foreach (var plant in obs.Plants)
        {
            if (plant.Row != row || plant.Column != column)
            {
                continue;
            }
            if (plant.InstanceId != 0 && !seenInstanceIds.Add(plant.InstanceId))
            {
                continue;
            }
            result.Add(BuildTilePlantSnapshot(plant, "logical"));
        }

        foreach (var visible in obs.VisiblePlants)
        {
            if (!visible.ActiveInHierarchy || !visible.InBoardBounds || visible.Row != row || visible.Column != column)
            {
                continue;
            }
            if (visible.InstanceId != 0 && !seenInstanceIds.Add(visible.InstanceId))
            {
                continue;
            }
            result.Add(new TilePlantSnapshotDto
            {
                InstanceId = visible.InstanceId,
                PlantType = visible.Type,
                PlantTypeName = visible.TypeName ?? ((PlantType)visible.Type).ToString(),
                Row = visible.Row,
                Column = visible.Column,
                Source = visible.InPlantArray ? "visible_in_plant_array" : "visible_only"
            });
        }

        return result;
    }

    private Dictionary<(int Row, int Column), List<TilePlantSnapshotDto>> CollectBoardTilePlantMap(ObservationDto obs)
    {
        var tilePlants = new Dictionary<(int Row, int Column), List<TilePlantSnapshotDto>>();
        var tileIdentityKeys = new Dictionary<(int Row, int Column), HashSet<string>>();
        if (obs == null)
        {
            return tilePlants;
        }

        void AddSnapshot(TilePlantSnapshotDto snapshot)
        {
            if (snapshot.Row < 0 || snapshot.Column < 0)
            {
                return;
            }

            var key = (snapshot.Row, snapshot.Column);
            if (!tilePlants.TryGetValue(key, out var entries))
            {
                entries = new List<TilePlantSnapshotDto>();
                tilePlants[key] = entries;
                tileIdentityKeys[key] = new HashSet<string>(StringComparer.Ordinal);
            }

            var identityKey = TilePlantIdentityKey(snapshot);
            if (!tileIdentityKeys[key].Add(identityKey))
            {
                return;
            }

            entries.Add(CloneTilePlantSnapshot(snapshot));
        }

        foreach (var plant in obs.Plants)
        {
            AddSnapshot(BuildTilePlantSnapshot(plant, "logical"));
        }

        foreach (var visible in obs.VisiblePlants)
        {
            if (!visible.ActiveInHierarchy || !visible.InBoardBounds)
            {
                continue;
            }

            AddSnapshot(new TilePlantSnapshotDto
            {
                InstanceId = visible.InstanceId,
                PlantType = visible.Type,
                PlantTypeName = visible.TypeName ?? ((PlantType)visible.Type).ToString(),
                Row = visible.Row,
                Column = visible.Column,
                Source = visible.InPlantArray ? "visible_in_plant_array" : "visible_only"
            });
        }

        return tilePlants;
    }

    private static List<FusionTileChangeDto> ComputeChangedTiles(
        IReadOnlyDictionary<(int Row, int Column), List<TilePlantSnapshotDto>> beforeTileMap,
        IReadOnlyDictionary<(int Row, int Column), List<TilePlantSnapshotDto>> afterTileMap)
    {
        var changed = new List<FusionTileChangeDto>();
        var allKeys = new HashSet<(int Row, int Column)>();
        foreach (var key in beforeTileMap.Keys)
        {
            allKeys.Add(key);
        }
        foreach (var key in afterTileMap.Keys)
        {
            allKeys.Add(key);
        }

        foreach (var key in allKeys.OrderBy(item => item.Row).ThenBy(item => item.Column))
        {
            beforeTileMap.TryGetValue(key, out var beforePlantsRaw);
            afterTileMap.TryGetValue(key, out var afterPlantsRaw);
            var beforePlants = beforePlantsRaw ?? new List<TilePlantSnapshotDto>();
            var afterPlants = afterPlantsRaw ?? new List<TilePlantSnapshotDto>();
            if (AreEquivalentTilePlantSets(beforePlants, afterPlants))
            {
                continue;
            }

            changed.Add(new FusionTileChangeDto
            {
                Row = key.Row,
                Column = key.Column,
                BeforePlantCount = beforePlants.Count,
                AfterPlantCount = afterPlants.Count,
                BeforePlants = beforePlants.Select(CloneTilePlantSnapshot).ToList(),
                AfterPlants = afterPlants.Select(CloneTilePlantSnapshot).ToList(),
            });
        }

        return changed;
    }

    private static TilePlantSnapshotDto CloneTilePlantSnapshot(TilePlantSnapshotDto plant)
    {
        return new TilePlantSnapshotDto
        {
            InstanceId = plant.InstanceId,
            PlantType = plant.PlantType,
            PlantTypeName = plant.PlantTypeName,
            Row = plant.Row,
            Column = plant.Column,
            Source = plant.Source
        };
    }

    private static TilePlantSnapshotDto? ResolveResultingTilePlant(
        IReadOnlyList<TilePlantSnapshotDto> plantsOnTileAfter,
        int predictedResultType,
        int sourceInstanceId)
    {
        if (plantsOnTileAfter.Count == 0)
        {
            return null;
        }
        if (predictedResultType > 0)
        {
            var predicted = plantsOnTileAfter.FirstOrDefault(plant => plant.PlantType == predictedResultType);
            if (predicted != null)
            {
                return predicted;
            }
        }
        if (sourceInstanceId != 0)
        {
            var sourceInstance = plantsOnTileAfter.FirstOrDefault(plant => plant.InstanceId == sourceInstanceId);
            if (sourceInstance != null)
            {
                return sourceInstance;
            }
        }
        return plantsOnTileAfter[0];
    }

    private static bool DidFusionProduceObservableChange(
        IReadOnlyList<TilePlantSnapshotDto> beforeTilePlants,
        IReadOnlyList<TilePlantSnapshotDto> afterTilePlants,
        TilePlantSnapshotDto? sourcePlantBefore,
        TilePlantSnapshotDto? resultingPlantAfter)
    {
        if (beforeTilePlants.Count != afterTilePlants.Count)
        {
            return true;
        }

        if (!AreEquivalentTilePlantSets(beforeTilePlants, afterTilePlants))
        {
            return true;
        }

        if (sourcePlantBefore != null && resultingPlantAfter != null)
        {
            if (sourcePlantBefore.PlantType != resultingPlantAfter.PlantType)
            {
                return true;
            }

            if (sourcePlantBefore.InstanceId != 0 &&
                resultingPlantAfter.InstanceId != 0 &&
                sourcePlantBefore.InstanceId != resultingPlantAfter.InstanceId)
            {
                return true;
            }
        }

        return false;
    }

    private static bool AreEquivalentTilePlantSets(
        IReadOnlyList<TilePlantSnapshotDto> leftPlants,
        IReadOnlyList<TilePlantSnapshotDto> rightPlants)
    {
        if (leftPlants.Count != rightPlants.Count)
        {
            return false;
        }

        var left = leftPlants.Select(TilePlantIdentityKey).OrderBy(value => value, StringComparer.Ordinal).ToArray();
        var right = rightPlants.Select(TilePlantIdentityKey).OrderBy(value => value, StringComparer.Ordinal).ToArray();
        if (left.Length != right.Length)
        {
            return false;
        }

        for (var index = 0; index < left.Length; index++)
        {
            if (!string.Equals(left[index], right[index], StringComparison.Ordinal))
            {
                return false;
            }
        }

        return true;
    }

    private static string TilePlantIdentityKey(TilePlantSnapshotDto plant)
    {
        var instanceId = plant?.InstanceId ?? 0;
        var plantType = plant?.PlantType ?? -1;
        return $"{plantType}:{instanceId}";
    }

    private FusionExecutionAttempt TryExecuteDedicatedFusion(
        CardUI? card,
        CreatePlant createPlant,
        PlantDto source,
        SeedSlotDto slot,
        IReadOnlyDictionary<(int Row, int Column), List<TilePlantSnapshotDto>> beforeBoardTileMap)
    {
        if (card != null)
        {
            var mouseAttempt = TryExecuteDedicatedFusionViaMouseCard(card, source, slot.PlantType, beforeBoardTileMap);
            if (mouseAttempt.Success)
            {
                return mouseAttempt;
            }
        }

        // Disable global mix API fallback to guarantee tile-scoped fusion behavior.
        // We retain postcondition checks in TryFuseSeedSlot as a defense-in-depth guard.
        return new FusionExecutionAttempt
        {
            Success = false,
            MethodUsed = "dedicated_fusion_mouse_only",
            Reason = "global_fusion_api_not_tile_scoped"
        };
    }

    private FusionExecutionAttempt TryExecuteDedicatedFusionViaMouseCard(
        CardUI card,
        PlantDto source,
        int ingredientPlantType,
        IReadOnlyDictionary<(int Row, int Column), List<TilePlantSnapshotDto>> beforeBoardTileMap)
    {
        var mouse = FindMouseController();
        if (mouse == null)
        {
            return new FusionExecutionAttempt
            {
                Success = false,
                MethodUsed = "mouse_card",
                Reason = "mouse_controller_not_found"
            };
        }

        // Clear any stale held item before selecting the ingredient card.
        _ = TryInvokeWithCompatibleSignature(mouse, "ClearItemOnMouse", out _, true);
        _ = TryInvokeWithCompatibleSignature(mouse, "ClearItemOnMouse", out _, false);
        _ = TryInvokeWithCompatibleSignature(mouse, "ClearItemOnMouse", out _);

        var selected = false;
        var selectMethodUsed = "";
        if (TryInvokeWithCompatibleSignature(mouse, "ClickCard", out var clickCardResult, card) &&
            (clickCardResult is not bool clickCardBool || clickCardBool))
        {
            selected = true;
            selectMethodUsed = "Mouse.ClickCard";
        }
        else if (TryInvokeWithCompatibleSignature(mouse, "ClickOnCard", out _, card))
        {
            selected = true;
            selectMethodUsed = "Mouse.ClickOnCard";
        }
        else
        {
            try
            {
                card.OnMouseDown();
                selected = true;
                selectMethodUsed = "CardUI.OnMouseDown";
            }
            catch
            {
                selected = false;
            }
        }

        if (!selected)
        {
            return new FusionExecutionAttempt
            {
                Success = false,
                MethodUsed = "mouse_card_select",
                Reason = "card_selection_failed"
            };
        }

        var targetState = PrepareMouseForSourceTile(mouse, source, ingredientPlantType);
        var attemptedMethods = new List<string>();
        var dropMethods = new[]
        {
            ("TryToSetPlantByCard", "Mouse.TryToSetPlantByCard"),
            ("ReinforcePlant", "Mouse.ReinforcePlant"),
            ("LeftClickWithSomeThing", "Mouse.LeftClickWithSomeThing"),
            ("PutDownItem", "Mouse.PutDownItem"),
            ("LeftUp", "Mouse.LeftUp")
        };
        foreach (var (methodName, methodLabel) in dropMethods)
        {
            _ = PrepareMouseForSourceTile(mouse, source, ingredientPlantType);
            if (!TryInvokeWithCompatibleSignature(mouse, methodName, out var methodResult))
            {
                continue;
            }
            attemptedMethods.Add(methodLabel);
            var executed = methodResult is not bool boolResult || boolResult;
            if (!executed)
            {
                continue;
            }

            Thread.Sleep(35);
            if (MouseAttemptMutatedBoard(
                    beforeBoardTileMap,
                    source,
                    out var sourceTileChanged,
                    out var nonSourceTileChanged))
            {
                return new FusionExecutionAttempt
                {
                    Success = true,
                    MethodUsed = $"{selectMethodUsed}->{methodLabel}",
                    Reason = nonSourceTileChanged
                        ? "mouse_method_mutated_non_source_tiles"
                        : sourceTileChanged
                            ? "success"
                            : "mouse_method_mutated_board"
                };
            }
        }

        var attemptedSummary = attemptedMethods.Count > 0
            ? string.Join("|", attemptedMethods)
            : "none";
        return new FusionExecutionAttempt
        {
            Success = attemptedMethods.Count > 0,
            MethodUsed = $"{selectMethodUsed}->{attemptedSummary}",
            Reason = attemptedMethods.Count > 0
                ? $"no_observable_effect_after_mouse_methods({targetState})"
                : $"mouse_drop_methods_unavailable({targetState})"
        };
    }

    private string PrepareMouseForSourceTile(object mouse, PlantDto source, int ingredientPlantType)
    {
        var details = new List<string>();
        try
        {
            var board = FindBoard();
            if (board != null && TrySetMemberValue(mouse, "board", board))
            {
                details.Add("board_assigned");
            }
        }
        catch
        {
            // Best-effort targeting only; the postcondition decides success.
        }

        var boxX = source.X;
        var boxY = source.Y;
        if (TryInvokeWithCompatibleSignature(mouse, "GetBoxXFromColumn", out var boxXObj, source.Column) &&
            TryObjectToFloat(boxXObj, out var resolvedBoxX))
        {
            boxX = resolvedBoxX;
            details.Add("box_x_from_column");
        }
        if (TryInvokeWithCompatibleSignature(mouse, "GetBoxYFromRow", out var boxYObj, source.Row) &&
            TryObjectToFloat(boxYObj, out var resolvedBoxY))
        {
            boxY = resolvedBoxY;
            details.Add("box_y_from_row");
        }

        var colAssigned = TrySetMemberValue(mouse, "theMouseColumn", source.Column);
        var rowAssigned = TrySetMemberValue(mouse, "theMouseRow", source.Row);
        var boxXAssigned = TrySetMemberValue(mouse, "theBoxXofMouse", boxX);
        var boxYAssigned = TrySetMemberValue(mouse, "theBoxYofMouse", boxY);
        var mouseXAssigned = TrySetMemberValue(mouse, "mouseX", boxX);
        var mouseYAssigned = TrySetMemberValue(mouse, "mouseY", boxY);
        var lastClickAssigned = TrySetMemberValue(mouse, "lastClickPosition", new Vector2(boxX, boxY));
        var typeAssigned = TrySetMemberValue(mouse, "thePlantTypeOnMouse", (PlantType)ingredientPlantType);
        var runtimeSource = FindRuntimePlant(source);
        var plantAssigned = runtimeSource != null && TrySetMemberValue(mouse, "plantSelected", runtimeSource);
        _ = TryInvokeWithCompatibleSignature(mouse, "PreviewPositionUpdate", out _);
        _ = TryInvokeWithCompatibleSignature(mouse, "LightUpPlantUnderMouse", out _);
        _ = TryInvokeWithCompatibleSignature(mouse, "PlantPreviewUpdate", out _);
        details.Add($"row={rowAssigned}");
        details.Add($"col={colAssigned}");
        details.Add($"box=({boxXAssigned},{boxYAssigned})");
        details.Add($"mouse=({mouseXAssigned},{mouseYAssigned})");
        details.Add($"last_click={lastClickAssigned}");
        details.Add($"type={typeAssigned}");
        details.Add($"plant_selected={plantAssigned}");
        return string.Join(",", details);
    }

    private bool MouseAttemptMutatedBoard(
        IReadOnlyDictionary<(int Row, int Column), List<TilePlantSnapshotDto>> beforeBoardTileMap,
        PlantDto source,
        out bool sourceTileChanged,
        out bool nonSourceTileChanged)
    {
        sourceTileChanged = false;
        nonSourceTileChanged = false;
        try
        {
            var afterObservation = BuildObservation(forceSeedProbe: true);
            var changedTiles = ComputeChangedTiles(beforeBoardTileMap, CollectBoardTilePlantMap(afterObservation));
            foreach (var tile in changedTiles)
            {
                if (tile.Row == source.Row && tile.Column == source.Column)
                {
                    sourceTileChanged = true;
                }
                else
                {
                    nonSourceTileChanged = true;
                }
            }
            return changedTiles.Count > 0;
        }
        catch
        {
            return false;
        }
    }

    private Plant? FindRuntimePlant(PlantDto source)
    {
        bool Matches(Plant plant, bool requireInstanceMatch)
        {
            try
            {
                if (plant.thePlantRow != source.Row ||
                    plant.thePlantColumn != source.Column ||
                    (int)plant.thePlantType != source.Type)
                {
                    return false;
                }
                return !requireInstanceMatch || source.InstanceId == 0 || plant.GetInstanceID() == source.InstanceId;
            }
            catch
            {
                return false;
            }
        }

        Plant? FindInBoardArray(bool requireInstanceMatch)
        {
            try
            {
                var board = FindBoard();
                var plants = board?.plantArray;
                if (plants != null)
                {
                    for (var index = 0; index < plants.Count; index++)
                    {
                        var plant = plants[index];
                        if (plant != null && Matches(plant, requireInstanceMatch))
                        {
                            return plant;
                        }
                    }
                }
            }
            catch
            {
                // Fall back to scene scan.
            }
            return null;
        }

        Plant? FindInScene(bool requireInstanceMatch)
        {
            try
            {
                foreach (var plant in Object.FindObjectsOfType<Plant>())
                {
                    if (plant != null && Matches(plant, requireInstanceMatch))
                    {
                        return plant;
                    }
                }
            }
            catch
            {
                // Ignore destroyed IL2CPP wrappers.
            }
            return null;
        }

        var exactBoardPlant = FindInBoardArray(requireInstanceMatch: true);
        if (exactBoardPlant != null)
        {
            return exactBoardPlant;
        }
        var exactScenePlant = FindInScene(requireInstanceMatch: true);
        if (exactScenePlant != null)
        {
            return exactScenePlant;
        }

        // If the Unity instance ID changed between observation and execution, stay tile-scoped
        // by falling back only to the same row/column/type source plant.
        var tileBoardPlant = FindInBoardArray(requireInstanceMatch: false);
        if (tileBoardPlant != null)
        {
            return tileBoardPlant;
        }
        return FindInScene(requireInstanceMatch: false);
    }

    private static bool TryObjectToFloat(object? value, out float result)
    {
        try
        {
            result = Convert.ToSingle(value);
            return true;
        }
        catch
        {
            result = 0.0f;
            return false;
        }
    }

    private FusionExecutionAttempt TryExecuteDedicatedFusionViaMixApi(CreatePlant createPlant, PlantDto source, int ingredientPlantType)
    {
        return new FusionExecutionAttempt
        {
            Success = false,
            MethodUsed = "global_mix_api_disabled",
            Reason = "global_fusion_api_not_tile_scoped"
        };
    }

    private object? FindMouseController()
    {
        try
        {
            var assembly = typeof(Board).Assembly;
            var mouseType = assembly.GetTypes().FirstOrDefault(type => string.Equals(type.Name, "Mouse", StringComparison.Ordinal));
            if (mouseType == null)
            {
                return null;
            }

            var flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.Instance;
            var instanceProperty = mouseType.GetProperty("Instance", flags);
            if (instanceProperty != null)
            {
                var instance = instanceProperty.GetValue(null, null);
                if (instance != null)
                {
                    return instance;
                }
            }

            var instanceField = mouseType.GetField("Instance", flags);
            if (instanceField != null)
            {
                var instance = instanceField.GetValue(null);
                if (instance != null)
                {
                    return instance;
                }
            }

            // Avoid FindObjectOfType(Type) here because Il2Cpp expects Il2CppSystem.Type.
            // For this bridge path, Mouse.Instance is the authoritative access point.
            return null;
        }
        catch
        {
            return null;
        }
    }

    private static bool TryInvokeWithCompatibleSignature(object target, string methodName, out object? result, params object?[] args)
    {
        result = null;
        if (target == null)
        {
            return false;
        }

        var flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;
        MethodInfo[] methods;
        try
        {
            methods = target.GetType()
                .GetMethods(flags)
                .Where(candidate => candidate.Name == methodName && candidate.GetParameters().Length == args.Length)
                .ToArray();
        }
        catch
        {
            return false;
        }

        foreach (var method in methods)
        {
            if (!TryConvertArguments(args, method.GetParameters(), out var convertedArgs))
            {
                continue;
            }

            try
            {
                result = method.Invoke(target, convertedArgs);
                return true;
            }
            catch
            {
                // Continue trying overloads.
            }
        }

        return false;
    }

    private static bool TryConvertArguments(object?[] args, IReadOnlyList<ParameterInfo> parameters, out object?[] convertedArgs)
    {
        convertedArgs = new object?[args.Length];
        for (var index = 0; index < args.Length; index++)
        {
            var parameterType = parameters[index].ParameterType;
            if (parameterType.IsByRef)
            {
                parameterType = parameterType.GetElementType() ?? parameterType;
            }
            if (!TryConvertArgument(args[index], parameterType, out var converted))
            {
                return false;
            }
            convertedArgs[index] = converted;
        }

        return true;
    }

    private static bool TryConvertArgument(object? value, Type targetType, out object? converted)
    {
        converted = null;
        if (value == null)
        {
            if (!targetType.IsValueType || Nullable.GetUnderlyingType(targetType) != null)
            {
                return true;
            }
            return false;
        }

        var nullableTarget = Nullable.GetUnderlyingType(targetType) ?? targetType;
        if (nullableTarget.IsInstanceOfType(value))
        {
            converted = value;
            return true;
        }

        try
        {
            if (nullableTarget.IsEnum)
            {
                if (value is string enumName)
                {
                    converted = Enum.Parse(nullableTarget, enumName, ignoreCase: true);
                    return true;
                }
                var numericValue = Convert.ToInt32(value);
                converted = Enum.ToObject(nullableTarget, numericValue);
                return true;
            }

            converted = Convert.ChangeType(value, nullableTarget);
            return true;
        }
        catch
        {
            converted = null;
            return false;
        }
    }

    private static bool TrySetMemberValue(object target, string name, object? value)
    {
        if (target == null)
        {
            return false;
        }

        var flags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;
        try
        {
            var field = target.GetType().GetField(name, flags);
            if (field != null && TryConvertArgument(value, field.FieldType, out var convertedField))
            {
                field.SetValue(target, convertedField);
                return true;
            }
        }
        catch
        {
            // Ignore and continue with property fallback.
        }

        try
        {
            var property = target.GetType().GetProperty(name, flags);
            if (property != null &&
                property.CanWrite &&
                property.GetIndexParameters().Length == 0 &&
                TryConvertArgument(value, property.PropertyType, out var convertedProperty))
            {
                property.SetValue(target, convertedProperty, null);
                return true;
            }
        }
        catch
        {
            return false;
        }

        return false;
    }
}
