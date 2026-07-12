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
    private PlacementResult TryPlacePlant(int plantTypeId, int row, int column, bool checkLegal, ObservationDto? precheckedObservation = null)
    {
        var board = FindBoard();
        if (board == null)
        {
            return PlacementResult.Fail(plantTypeId, row, column, "Board not found.", "board_not_found");
        }

        var createPlant = FindCreatePlant();
        if (createPlant == null)
        {
            return PlacementResult.Fail(plantTypeId, row, column, "CreatePlant.Instance not found.", "create_plant_not_found");
        }

        var gateObservation = precheckedObservation ?? BuildObservation();
        if (gateObservation.SeedSelectionActive)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                "Seed selection UI is active; plant actions are disabled until Let's Rock has completed.",
                "seed_selection_active");
        }

        var plantType = (PlantType)plantTypeId;
        var costInfo = GetPlantCost(plantTypeId);
        var sunBefore = board.theSun;

        if (sunBefore < costInfo.Cost)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                $"Insufficient sun: need {costInfo.Cost}, have {sunBefore}.",
                "insufficient_sun",
                costInfo,
                sunBefore,
                sunBefore);
        }

        var cooldownInfo = GetCardCooldown(plantTypeId);
        if (!cooldownInfo.Ready)
        {
            var reason = cooldownInfo.Found ? "cooldown" : "card_not_found";
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                cooldownInfo.Found
                    ? $"Card is on cooldown: {cooldownInfo.CurrentCooldown:0.###}/{cooldownInfo.FullCooldown:0.###}."
                    : "No matching CardUI seed packet was found.",
                reason,
                costInfo,
                sunBefore,
                sunBefore,
                cooldownInfo);
        }

        if (checkLegal)
        {
            var occupancy = gateObservation;
            if (IsCellOccupied(occupancy, row, column))
            {
                return PlacementResult.Fail(
                    plantTypeId,
                    row,
                    column,
                    "Cell is occupied by a logical or visible plant object.",
                    "occupied_cell",
                    costInfo,
                    sunBefore,
                    sunBefore,
                    cooldownInfo);
            }

            try
            {
                if (!createPlant.CheckBox(column, row, plantType))
                {
                    return PlacementResult.Fail(
                        plantTypeId,
                        row,
                        column,
                        "CreatePlant.CheckBox returned false.",
                        "invalid_cell_or_terrain",
                        costInfo,
                        sunBefore,
                        sunBefore);
                }
            }
            catch (Exception ex)
            {
                return PlacementResult.Fail(
                    plantTypeId,
                    row,
                    column,
                    "CreatePlant.CheckBox failed: " + ex.Message,
                    "checkbox_failed",
                    costInfo,
                    sunBefore,
                    sunBefore);
            }
        }

        try
        {
            var obj = createPlant.SetPlant(column, row, plantType, null, Vector2.zero, true, true, null);
            var plantPlaced = obj != null;
            var costPaid = false;
            var paymentSource = "not_paid";
            var sunAfter = board.theSun;

            if (plantPlaced)
            {
                costPaid = TryPaySun(board, costInfo.Cost, sunBefore, out sunAfter, out paymentSource);
            }

            var cooldownStarted = false;
            var cooldownSource = "not_started";
            if (plantPlaced && costPaid)
            {
                cooldownStarted = TryStartCardCooldown(plantTypeId, out cooldownSource);
                InvalidateSeedRuntimeCache($"plant_type[{plantTypeId}] placed");
            }

            return new PlacementResult
            {
                Success = plantPlaced && costPaid,
                PlantPlaced = plantPlaced,
                CostPaid = costPaid,
                CooldownStarted = cooldownStarted,
                PlantCost = costInfo.Cost,
                CostSource = costInfo.Source,
                PaymentSource = paymentSource,
                CooldownSource = cooldownSource,
                CardCooldown = GetCardCooldown(plantTypeId),
                SunBefore = sunBefore,
                SunAfter = sunAfter,
                PlantType = plantTypeId,
                PlantTypeName = plantType.ToString(),
                Row = row,
                Column = column,
                Message = plantPlaced ? "Plant placed and cost paid." : "SetPlant returned null.",
                IllegalReason = plantPlaced && costPaid ? null : plantPlaced ? "cost_payment_failed" : "setplant_returned_null"
            };
        }
        catch (Exception ex)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                "CreatePlant.SetPlant failed: " + ex.Message,
                "setplant_failed",
                costInfo,
                sunBefore,
                board.theSun);
        }
    }

    private PlacementResult TryPlaceSeedSlot(int seedSlotIndex, int row, int column, bool checkLegal, ObservationDto? precheckedObservation = null)
    {
        if (!TryGetSeedSlotForPlacement(seedSlotIndex, out var slot, out var card))
        {
            return PlacementResult.Fail(
                -1,
                row,
                column,
                $"Seed slot {seedSlotIndex} is not available in the active gameplay card bank.",
                "seed_slot_not_found")
                .WithSeedSlot(seedSlotIndex);
        }

        var plantTypeId = slot.PlantType;
        var board = FindBoard();
        if (board == null)
        {
            return PlacementResult.Fail(plantTypeId, row, column, "Board not found.", "board_not_found")
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }

        var createPlant = FindCreatePlant();
        if (createPlant == null)
        {
            return PlacementResult.Fail(plantTypeId, row, column, "CreatePlant.Instance not found.", "create_plant_not_found")
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }

        var gateObservation = precheckedObservation ?? BuildObservation();
        if (gateObservation.SeedSelectionActive)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                "Seed selection UI is active; plant actions are disabled until Let's Rock has completed.",
                "seed_selection_active")
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }

        var plantType = (PlantType)plantTypeId;
        var costInfo = new PlantCostInfo
        {
            PlantType = plantTypeId,
            Cost = Math.Max(0, slot.SeedCost),
            Source = slot.Source ?? "seed_slot",
            Warning = slot.Warning
        };
        var sunBefore = board.theSun;

        if (!slot.Usable || slot.Disabled)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                $"Seed slot {seedSlotIndex} is not usable.",
                slot.Disabled ? "slot_disabled" : "slot_not_usable",
                costInfo,
                sunBefore,
                sunBefore,
                SlotCooldownFromDto(slot))
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }

        if (sunBefore < costInfo.Cost)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                $"Insufficient sun for seed slot {seedSlotIndex}: need {costInfo.Cost}, have {sunBefore}.",
                "insufficient_sun",
                costInfo,
                sunBefore,
                sunBefore,
                SlotCooldownFromDto(slot))
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }

        var cooldownInfo = card != null
            ? BuildCardCooldownDto(card, plantTypeId, $"seed_slot[{seedSlotIndex}]")
            : SlotCooldownFromDto(slot);
        if (!cooldownInfo.Ready)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                $"Seed slot {seedSlotIndex} is on cooldown: {cooldownInfo.CurrentCooldown:0.###}/{cooldownInfo.FullCooldown:0.###}.",
                "cooldown",
                costInfo,
                sunBefore,
                sunBefore,
                cooldownInfo)
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }

        if (checkLegal)
        {
            if (IsCellOccupied(gateObservation, row, column))
            {
                return PlacementResult.Fail(
                    plantTypeId,
                    row,
                    column,
                    "Cell is occupied by a logical or visible plant object.",
                    "occupied_cell",
                    costInfo,
                    sunBefore,
                    sunBefore,
                    cooldownInfo)
                    .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
            }

            try
            {
                if (!createPlant.CheckBox(column, row, plantType))
                {
                    return PlacementResult.Fail(
                        plantTypeId,
                        row,
                        column,
                        "CreatePlant.CheckBox returned false.",
                        "invalid_cell_or_terrain",
                        costInfo,
                        sunBefore,
                        sunBefore,
                        cooldownInfo)
                        .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
                }
            }
            catch (Exception ex)
            {
                return PlacementResult.Fail(
                    plantTypeId,
                    row,
                    column,
                    "CreatePlant.CheckBox failed: " + ex.Message,
                    "checkbox_failed",
                    costInfo,
                    sunBefore,
                    sunBefore,
                    cooldownInfo)
                    .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
            }
        }

        try
        {
            var obj = createPlant.SetPlant(column, row, plantType, null, Vector2.zero, true, true, null);
            var plantPlaced = obj != null;
            var costPaid = false;
            var paymentSource = "not_paid";
            var sunAfter = board.theSun;

            if (plantPlaced)
            {
                costPaid = TryPaySun(board, costInfo.Cost, sunBefore, out sunAfter, out paymentSource);
            }

            var cooldownStarted = false;
            var cooldownSource = "not_started";
            if (plantPlaced && costPaid && card != null)
            {
                cooldownStarted = TryStartSeedSlotCooldown(card, seedSlotIndex, out cooldownSource);
                InvalidateSeedRuntimeCache($"seed_slot[{seedSlotIndex}] placed");
            }

            return new PlacementResult
            {
                Success = plantPlaced && costPaid,
                PlantPlaced = plantPlaced,
                CostPaid = costPaid,
                CooldownStarted = cooldownStarted,
                PlantCost = costInfo.Cost,
                CostSource = costInfo.Source,
                CostWarning = costInfo.Warning,
                PaymentSource = paymentSource,
                CooldownSource = cooldownSource,
                CardCooldown = card != null ? BuildCardCooldownDto(card, plantTypeId, $"seed_slot[{seedSlotIndex}]") : SlotCooldownFromDto(slot),
                SunBefore = sunBefore,
                SunAfter = sunAfter,
                PlantType = plantTypeId,
                PlantTypeName = plantType.ToString(),
                SeedSlotIndex = seedSlotIndex,
                CardInstanceId = slot.CardInstanceId,
                Row = row,
                Column = column,
                Message = plantPlaced ? $"Plant placed from seed slot {seedSlotIndex} and cost paid." : "SetPlant returned null.",
                IllegalReason = plantPlaced && costPaid ? null : plantPlaced ? "cost_payment_failed" : "setplant_returned_null"
            };
        }
        catch (Exception ex)
        {
            return PlacementResult.Fail(
                plantTypeId,
                row,
                column,
                "CreatePlant.SetPlant failed: " + ex.Message,
                "setplant_failed",
                costInfo,
                sunBefore,
                board.theSun)
                .WithSeedSlot(seedSlotIndex, slot.CardInstanceId);
        }
    }
}
