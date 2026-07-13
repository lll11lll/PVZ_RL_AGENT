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


internal sealed class ProofReport
{
    public bool BoardFound { get; set; }
    public bool CreatePlantFound { get; set; }
    public bool CanReadBoard { get; set; }
    public bool CanReadPlants { get; set; }
    public bool CanReadZombies { get; set; }
    public bool GoNoGo { get; set; }
    public string? NextStep { get; set; }
    public int Sun { get; set; }
    public int Wave { get; set; }
    public int MaxWave { get; set; }
    public int PlantCount { get; set; }
    public int ZombieCount { get; set; }
    public string? ReadError { get; set; }
    public PlacementResult? PlacementAttempt { get; set; }
}

internal sealed class TilePlantSnapshotDto
{
    public int InstanceId { get; set; }
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public int Row { get; set; }
    public int Column { get; set; }
    public string? Source { get; set; }
}

internal sealed class FusionTileChangeDto
{
    public int Row { get; set; }
    public int Column { get; set; }
    public int BeforePlantCount { get; set; }
    public int AfterPlantCount { get; set; }
    public List<TilePlantSnapshotDto> BeforePlants { get; set; } = new();
    public List<TilePlantSnapshotDto> AfterPlants { get; set; } = new();
}

internal sealed class FusionExecutionAttempt
{
    public bool Success { get; set; }
    public string MethodUsed { get; set; } = "none";
    public string Reason { get; set; } = "bridge_rejected";
}

internal sealed class FusionPredictionInfo
{
    public int PredictedResultType { get; set; } = -1;
    public string PredictedResultName { get; set; } = "";
    public string PredictedResultResolutionSource { get; set; } = "unresolved";
    public bool MixLookupFound { get; set; }
    public string MixLookupKey { get; set; } = "";
}

internal sealed class PlacementResult
{
    public bool Success { get; set; }
    public bool PlantPlaced { get; set; }
    public bool CostPaid { get; set; }
    public bool CooldownStarted { get; set; }
    public int PlantCost { get; set; }
    public string? CostSource { get; set; }
    public string? CostWarning { get; set; }
    public string? PaymentSource { get; set; }
    public string? CooldownSource { get; set; }
    public CardCooldownDto? CardCooldown { get; set; }
    public int SunBefore { get; set; }
    public int SunAfter { get; set; }
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public int SeedSlotIndex { get; set; } = -1;
    public int CardInstanceId { get; set; }
    public int Row { get; set; }
    public int Column { get; set; }
    public string? Message { get; set; }
    public string? IllegalReason { get; set; }
    public string? FusionExecutionMode { get; set; }
    public bool SourceTileOccupiedBefore { get; set; }
    public int PlantCountOnTileBefore { get; set; }
    public int PlantCountOnTileAfter { get; set; }
    public TilePlantSnapshotDto? SourcePlantBefore { get; set; }
    public TilePlantSnapshotDto? ResultingPlantAfter { get; set; }
    public bool DuplicateStackDetected { get; set; }
    public string? BridgeMethodUsed { get; set; }
    public string? BridgeResultReason { get; set; }
    public string? PredictedResultResolutionSource { get; set; }
    public bool MixLookupFound { get; set; }
    public string? MixLookupKey { get; set; }
    public int PreSourceType { get; set; } = -1;
    public string? PreSourceName { get; set; }
    public int IngredientType { get; set; } = -1;
    public string? IngredientName { get; set; }
    public int PostResultType { get; set; } = -1;
    public string? PostResultName { get; set; }
    public string? NoEffectReason { get; set; }
    public int RequestedSourceRow { get; set; } = -1;
    public int RequestedSourceCol { get; set; } = -1;
    public int RequestedSourceInstanceId { get; set; }
    public int ChangedTileCount { get; set; }
    public List<FusionTileChangeDto> ChangedTiles { get; set; } = new();
    public bool NonSourceTilesChanged { get; set; }
    public bool GlobalFusionSideEffect { get; set; }
    public string FusionScope { get; set; } = "unknown";

    public static PlacementResult Fail(
        int plantType,
        int row,
        int column,
        string message,
        string illegalReason,
        PlantCostInfo? costInfo = null,
        int sunBefore = 0,
        int sunAfter = 0,
        CardCooldownDto? cooldown = null) =>
        new()
        {
            Success = false,
            PlantPlaced = false,
            CostPaid = false,
            CooldownStarted = false,
            PlantCost = costInfo?.Cost ?? 0,
            CostSource = costInfo?.Source,
            CostWarning = costInfo?.Warning,
            PaymentSource = "not_paid",
            CooldownSource = "not_started",
            CardCooldown = cooldown,
            SunBefore = sunBefore,
            SunAfter = sunAfter,
            PlantType = plantType,
            PlantTypeName = ((PlantType)plantType).ToString(),
            Row = row,
            Column = column,
            Message = message,
            IllegalReason = illegalReason
        };

    public PlacementResult WithSeedSlot(int seedSlotIndex, int cardInstanceId = 0)
    {
        SeedSlotIndex = seedSlotIndex;
        CardInstanceId = cardInstanceId;
        return this;
    }
}

internal sealed class PlantCostInfo
{
    public int PlantType { get; set; }
    public int Cost { get; set; }
    public string Source { get; set; } = "unknown";
    public string? Warning { get; set; }
}

internal sealed class DecodedAction
{
    public string Kind { get; set; } = "wait";
    public int PlantType { get; set; } = -1;
    public string? PlantTypeName { get; set; }
    public int SeedSlotIndex { get; set; } = -1;
    public int CardInstanceId { get; set; }
    public int SeedCost { get; set; }
    public int Row { get; set; } = -1;
    public int Column { get; set; } = -1;
    public string? Error { get; set; }

    public static DecodedAction Wait() => new() { Kind = "wait" };

    public static DecodedAction Invalid(int action, string error) =>
        new() { Kind = "invalid", Error = $"Action {action}: {error}" };
}

internal sealed class ObservationDto
{
    public bool BoardFound { get; set; }
    public bool CreatePlantFound { get; set; }
    public bool GameplayReady { get; set; }
    public bool ActualGameplayReady { get; set; }
    public bool SeedSelectionActive { get; set; }
    public bool SeedSelectionPanelActive { get; set; }
    public bool StartButtonActive { get; set; }
    public bool BlockingRewardUiActive { get; set; }
    public int ActiveGameplayCardBankCount { get; set; }
    public string? LegalActionReason { get; set; }
    public int Sun { get; set; }
    public int Wave { get; set; }
    public int MaxWave { get; set; }
    public int RowCount { get; set; }
    public int ColumnCount { get; set; }
    public int KillCount { get; set; }
    public bool Over { get; set; }
    public bool Done { get; set; }
    public bool BoardCardSelectable { get; set; }
    public bool BoardStartMove { get; set; }
    public bool MoreZombiesComing { get; set; }
    public float Time { get; set; }
    public int FrameCount { get; set; }
    public float RealtimeSinceStartup { get; set; }
    public float GameSpeed { get; set; }
    public float RequestedGameSpeed { get; set; }
    public string? GameSpeedMode { get; set; }
    public float UnityTimeScale { get; set; }
    public float FixedDeltaTime { get; set; }
    public float EffectiveGameSpeed { get; set; }
    public float? SkySunSpawnInterval { get; set; }
    public float? SkySunSpawnTimer { get; set; }
    public int? ActiveFallingSunCount { get; set; }
    public float? SunSpawnCountPerMinute { get; set; }
    public int? SunCollectedCount { get; set; }
    public long SpeedApplyCount { get; set; }
    public long ValidSpeedModeApplyCount { get; set; }
    public long SunSpawnCompensationApplyCount { get; set; }
    public long BridgeUpdateLoopCount { get; set; }
    public long ResetCount { get; set; }
    public long LetsRockClickCount { get; set; }
    public int BoardInstanceId { get; set; }
    public int? ActiveBoardCount { get; set; }
    public int? ActiveSkySunSpawnerCount { get; set; }
    public int? ActiveSunObjectCount { get; set; }
    public int? ActiveCoroutineCount { get; set; }
    public float NewZombieWaveCountDown { get; set; }
    public float NextZombieWaveCountDown { get; set; }
    public float HugeWaveCountDown { get; set; }
    public int PlantCount { get; set; }
    public int VisiblePlantObjectCount { get; set; }
    public int StaleVisiblePlantObjectCount { get; set; }
    public int LogicalMowerCount { get; set; }
    public int VisibleMowerObjectCount { get; set; }
    public int StaleVisibleMowerObjectCount { get; set; }
    public int DuplicateMowerRowCount { get; set; }
    public int ZombieCount { get; set; }
    public int BulletCount { get; set; }
    public int TotalPlantHealth { get; set; }
    public float MinBoardX { get; set; }
    public float MaxBoardX { get; set; }
    public float ZombieMinX { get; set; }
    public float ZombieMaxX { get; set; }
    public string? TerminalHint { get; set; }
    public string? ScreenState { get; set; }
    public string? CurrentMode { get; set; }
    public bool IsMainMenu { get; set; }
    public bool IsAdventureButtonVisible { get; set; }
    public bool StartupPopupVisible { get; set; }
    public bool StartupOkButtonVisible { get; set; }
    public bool MainMenuBlockedByPopup { get; set; }
    public bool IsSeedSelectionScreen { get; set; }
    public bool IsGameplayReady { get; set; }
    public bool IsLevelComplete { get; set; }
    public bool IsRewardScreen { get; set; }
    public bool IsNewPlantUnlockedScreen { get; set; }
    public bool IsAlmanacOrSeedPacketScreen { get; set; }
    public bool IsGameOverScreen { get; set; }
    public int CurrentAdventureLevel { get; set; }
    public int CurrentWorldOrStage { get; set; }
    public int CurrentDayLevel { get; set; }
    public string[] UnlockedSeedNames { get; set; } = Array.Empty<string>();
    public string[] AvailableSeedNames { get; set; } = Array.Empty<string>();
    public string[] SelectedSeedNames { get; set; } = Array.Empty<string>();
    public string[] UnknownVisibleSeedNames { get; set; } = Array.Empty<string>();
    public string? ReadError { get; set; }
    public int LegalActionCount { get; set; }
    public int ActionCount { get; set; }
    public int SeedSlotCount { get; set; }
    public int[] SlotPlantTypes { get; set; } = Array.Empty<int>();
    public bool CanReadBoard { get; set; }
    public bool OnGameOverScreen { get; set; }
    public bool LossMenuActive { get; set; }
    public bool GameOverTextVisible { get; set; }
    public bool OnLossScreen { get; set; }
    public bool OnRestartScreen { get; set; }
    public bool RestartButtonActive { get; set; }
    public string? RestartButtonName { get; set; }
    public string? RestartButtonPath { get; set; }
    public string? RestartDetectionReason { get; set; }
    public string? RestartDetectionMode { get; set; }
    public bool OnPauseMenu { get; set; }
    public bool PauseMenuActive { get; set; }
    public bool PauseRestartButtonActive { get; set; }
    public bool OnSeedSelectionScreen { get; set; }
    public string? NextStep { get; set; }
    public string? DebugMessage { get; set; }
    public double observe_ms { get; set; }
    public double bridge_observe_ms { get; set; }
    public double screen_check_ms { get; set; }
    public double seed_probe_ms { get; set; }
    public double ui_scan_ms { get; set; }
    public List<PlantDto> Plants { get; } = new();
    public List<VisiblePlantDto> VisiblePlants { get; } = new();
    public List<VisibleMowerDto> VisibleMowers { get; } = new();
    public List<int> DuplicateMowerRows { get; } = new();
    public List<ZombieDto> Zombies { get; } = new();
    public List<LaneDto> Lanes { get; } = new();
    public List<PlantCostDto> PlantCosts { get; } = new();
    public List<CardCooldownDto> CardCooldowns { get; } = new();
    public List<SeedSlotDto> SeedSlots { get; } = new();
    public List<int> LegalActions { get; } = new();
}

internal sealed class RestartScreenInfo
{
    public bool OnGameOverScreen { get; set; }
    public bool LossMenuActive { get; set; }
    public bool GameOverTextVisible { get; set; }
    public bool OnRestartScreen { get; set; }
    public bool RestartButtonActive { get; set; }
    public string? RestartButtonName { get; set; }
    public string? RestartButtonPath { get; set; }
    public string? RestartDetectionReason { get; set; }
    public string? RestartDetectionMode { get; set; }
    public bool OnPauseMenu { get; set; }
    public bool PauseMenuActive { get; set; }
    public bool PauseRestartButtonActive { get; set; }
    public double screen_check_ms { get; set; }
    public GameObject? RestartButtonObject;
}

internal sealed class RestartUiCache
{
    public bool Valid { get; set; }
    public string? SceneKey { get; set; }
    public string? InvalidReason { get; set; }
    public LoseMenuBtn? LossRestartButton { get; set; }
    public GameObject? LossMenuRoot { get; set; }
    public GameObject? GameOverTextObject { get; set; }
    public PauseMenu_Btn? PauseRestartButton { get; set; }
    public GameObject? PauseMenuRoot { get; set; }
}

internal sealed class SeedSlotDto
{
    public int SlotIndex { get; set; }
    public int CardInstanceId { get; set; }
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public int SeedCost { get; set; }
    public bool Ready { get; set; }
    public bool Disabled { get; set; }
    public bool IsAvailable { get; set; }
    public float RawCooldown { get; set; }
    public float FullCooldown { get; set; }
    public float CurrentCooldown { get; set; }
    public bool Usable { get; set; }
    public string? Source { get; set; }
    public string? Warning { get; set; }
    public string? HierarchyPath { get; set; }
}

internal sealed class ResetCleanupReport
{
    public bool BoardFound { get; set; }
    public int LogicalPlantCount { get; set; }
    public int VisiblePlantObjectCount { get; set; }
    public int StaleVisiblePlantObjectCount { get; set; }
    public int LogicalMowerCount { get; set; }
    public int VisibleMowerObjectCount { get; set; }
    public int StaleVisibleMowerObjectCount { get; set; }
    public int DuplicateMowerRowCount { get; set; }
    public int LogicalZombieCount { get; set; }
    public int SceneZombieObjectCount { get; set; }
    public int LogicalBulletCount { get; set; }
    public int SceneBulletObjectCount { get; set; }
    public int LogicalGridItemCount { get; set; }
    public int SceneGridItemObjectCount { get; set; }
    public int SceneMowerObjectCount { get; set; }
    public List<VisiblePlantDto> VisiblePlants { get; } = new();
    public List<VisibleMowerDto> VisibleMowers { get; } = new();
    public List<int> DuplicateMowerRows { get; } = new();
}

internal sealed class PlantCostDto
{
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public int Cost { get; set; }
    public string? Source { get; set; }
}

internal sealed class CardCooldownDto
{
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public bool Found { get; set; }
    public bool Ready { get; set; }
    public float RawCooldown { get; set; }
    public float CurrentCooldown { get; set; }
    public float FullCooldown { get; set; }
    public bool IsAvailable { get; set; }
    public bool Disabled { get; set; }
    public bool OnCardBank { get; set; }
    public bool IsSelected { get; set; }
    public int SeedCost { get; set; }
    public int MatchingCardCount { get; set; }
    public string? Source { get; set; }
}

internal sealed class SeedCardDto
{
    public int InstanceId { get; set; }
    public bool Active { get; set; }
    public bool UiVisible { get; set; }
    public bool RendererVisible { get; set; }
    public bool HasVisibleUiComponent { get; set; }
    public bool InScreenBounds { get; set; }
    public bool InScreenSpaceBounds { get; set; }
    public float ScreenX { get; set; }
    public float ScreenY { get; set; }
    public float ScreenZ { get; set; }
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public string? DisplayName { get; set; }
    public int SeedCost { get; set; }
    public bool OnCardBank { get; set; }
    public bool IsSelected { get; set; }
    public bool PreSelected { get; set; }
    public bool IsAvailable { get; set; }
    public bool Disabled { get; set; }
    public bool Selectable { get; set; }
    public float Cd { get; set; }
    public float FullCd { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public float LocalX { get; set; }
    public float LocalY { get; set; }
    public string? GameObjectName { get; set; }
    public string? ParentName { get; set; }
    public string? CardParentName { get; set; }
    public string? RootName { get; set; }
    public string? HierarchyPath { get; set; }
    public string? Text { get; set; }
    public string? TextBg { get; set; }
    public string Classification { get; set; } = "runtimeCardWrapper";
    public string? MatchedRegistryKey { get; set; }
}

internal sealed class UiProbeEntryDto
{
    public string? Name { get; set; }
    public string? ClassName { get; set; }
    public string? Text { get; set; }
    public bool ActiveSelf { get; set; }
    public bool ActiveInHierarchy { get; set; }
    public bool RendererVisible { get; set; }
    public bool UiVisible { get; set; }
    public string? ParentName { get; set; }
    public string? RootName { get; set; }
    public string? HierarchyPath { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public float Z { get; set; }
    public float LocalX { get; set; }
    public float LocalY { get; set; }
    public float ScreenX { get; set; }
    public float ScreenY { get; set; }
    public float ScreenZ { get; set; }
    public bool InScreenBounds { get; set; }
}

internal sealed class UnlockScreenSnapshotDto
{
    public bool RewardScreenVisible { get; set; }
    public bool UnlockScreenVisible { get; set; }
    public bool NewPlantUnlockedVisible { get; set; }
    public string NewPlantUnlockedName { get; set; } = "";
    public int NewPlantUnlockedPlantType { get; set; } = -1;
    public string[] VisibleRewardTexts { get; set; } = Array.Empty<string>();
    public string[] VisibleSeedCardNames { get; set; } = Array.Empty<string>();
    public int[] VisibleSeedPlantTypes { get; set; } = Array.Empty<int>();
    public UiProbeEntryDto[] UnknownUnlockObjects { get; set; } = Array.Empty<UiProbeEntryDto>();
    public SeedCardDto[] UnknownVisibleSeedCards { get; set; } = Array.Empty<SeedCardDto>();
}

internal sealed class SeedProbeDto
{
    public bool BoardFound { get; set; }
    public bool InitBoardFound { get; set; }
    public bool SeedSelectionActive { get; set; }
    public bool SeedSelectionPanelActive { get; set; }
    public bool ChooseYourPlantsTextActive { get; set; }
    public bool StartButtonActive { get; set; }
    public bool BlockingRewardUiActive { get; set; }
    public bool BoardCardSelectable { get; set; }
    public bool BoardStartMove { get; set; }
    public bool GameplayReady { get; set; }
    public string? GameBoardType { get; set; }
    public int GameBoardLevel { get; set; }
    public int[] ConfiguredPlantTypes { get; set; } = Array.Empty<int>();
    public int CardCount { get; set; }
    public int SelectedOrBankedCount { get; set; }
    public int AvailableSeedPacketCount { get; set; }
    public int SelectedBankVisibleCount { get; set; }
    public int[] SelectedBankPlantTypes { get; set; } = Array.Empty<int>();
    public PlantTypeCountDto[] SelectedBankPlantTypeCounts { get; set; } = Array.Empty<PlantTypeCountDto>();
    public int AvailableCardVisibleCount { get; set; }
    public int[] AvailableCardPlantTypes { get; set; } = Array.Empty<int>();
    public PlantTypeCountDto[] AvailableCardPlantTypeCounts { get; set; } = Array.Empty<PlantTypeCountDto>();
    public int SeedChooserSignalCount { get; set; }
    public UiProbeEntryDto[] SeedChooserSignals { get; set; } = Array.Empty<UiProbeEntryDto>();
    public List<SeedCardDto> Cards { get; } = new();
    public List<SeedCardDto> AvailableSeedCards { get; } = new();
    public List<SeedCardDto> SelectedSeedBankCards { get; } = new();
    public List<SeedCardDto> ActiveGameplayCardBankCards { get; } = new();
    public List<SeedSlotDto> ActiveGameplaySeedSlots { get; } = new();
    public List<SeedCardDto> StalePreselectedCards { get; } = new();
    public List<SeedCardDto> RuntimeCardWrappers { get; } = new();
    public bool StartReadyAvailable { get; set; }
    public string[] KnownHooks { get; set; } = Array.Empty<string>();
    public string? Investigation { get; set; }
    public double seed_probe_ms { get; set; }
    public double ui_scan_ms { get; set; }
}

internal sealed class SeedRuntimeSnapshot
{
    public bool SeedSelectionActive { get; set; }
    public bool SeedSelectionPanelActive { get; set; }
    public bool StartButtonActive { get; set; }
    public bool BlockingRewardUiActive { get; set; }
    public int ActiveGameplayCardBankCount { get; set; }
    public Dictionary<int, int> ActiveGameplayTypeCounts { get; } = new();
    public List<SeedSlotDto> SeedSlots { get; } = new();

    public static SeedRuntimeSnapshot FromProbe(SeedProbeDto probe)
    {
        var snapshot = new SeedRuntimeSnapshot
        {
            SeedSelectionActive = probe.SeedSelectionActive,
            SeedSelectionPanelActive = probe.SeedSelectionPanelActive,
            StartButtonActive = probe.StartButtonActive,
            BlockingRewardUiActive = probe.BlockingRewardUiActive,
            ActiveGameplayCardBankCount = probe.ActiveGameplayCardBankCards.Count
        };
        foreach (var pair in BridgeMod.BuildTypeCounts(probe.ActiveGameplayCardBankCards.Select(card => card.PlantType)))
        {
            snapshot.ActiveGameplayTypeCounts[pair.Key] = pair.Value;
        }
        foreach (var slot in probe.ActiveGameplaySeedSlots)
        {
            snapshot.SeedSlots.Add(slot);
        }

        return snapshot;
    }
}

internal sealed class StartupPopupInfo
{
    public bool StartupPopupVisible { get; set; }
    public bool StartupOkButtonVisible { get; set; }
    public bool MainMenuBlockedByPopup { get; set; }
    public List<UiProbeEntryDto> StartupOkSignals { get; set; } = new();
    public List<UiProbeEntryDto> StartupPopupSignals { get; set; } = new();
}

internal sealed class SeedRuntimeCache
{
    public bool Valid { get; set; }
    public string? InvalidReason { get; set; }
    public bool SeedSelectionActive { get; set; }
    public bool SeedSelectionPanelActive { get; set; }
    public bool StartButtonActive { get; set; }
    public bool BlockingRewardUiActive { get; set; }
    public bool GameplayReady { get; set; }
    public bool ActualGameplayReady { get; set; }
    public int ActiveGameplayCardBankCount { get; set; }
    public Dictionary<int, int> ActiveGameplayTypeCounts { get; } = new();
    public Dictionary<int, int> CachedPlantCosts { get; } = new();
    public Dictionary<int, CardUI> CachedGameplayCards { get; } = new();
    public List<SeedSlotCacheEntry> CachedSeedSlots { get; } = new();
    public Dictionary<int, SeedSlotCacheEntry> CachedSeedSlotsByIndex { get; } = new();
    public List<SeedSlotDto> CachedSeedSlotDtos { get; } = new();

    public SeedRuntimeSnapshot ToSnapshot(bool rawGameplayReady)
    {
        var snapshot = new SeedRuntimeSnapshot
        {
            SeedSelectionActive = SeedSelectionActive,
            SeedSelectionPanelActive = SeedSelectionPanelActive,
            StartButtonActive = StartButtonActive,
            BlockingRewardUiActive = BlockingRewardUiActive,
            ActiveGameplayCardBankCount = ActiveGameplayCardBankCount
        };
        foreach (var pair in ActiveGameplayTypeCounts)
        {
            snapshot.ActiveGameplayTypeCounts[pair.Key] = pair.Value;
        }
        foreach (var slot in CachedSeedSlotDtos)
        {
            snapshot.SeedSlots.Add(slot);
        }

        if (!rawGameplayReady)
        {
            snapshot.ActiveGameplayTypeCounts.Clear();
            snapshot.ActiveGameplayCardBankCount = 0;
            snapshot.SeedSlots.Clear();
        }

        return snapshot;
    }
}

internal sealed class SeedSlotCacheEntry
{
    public int SlotIndex { get; set; }
    public int CardInstanceId { get; set; }
    public CardUI? Card { get; set; }
}

internal sealed class PlantTypeCountDto
{
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public int Count { get; set; }
}

internal sealed class SeedSelectionAttemptDto
{
    public int AttemptIndex { get; set; }
    public int PlantType { get; set; }
    public string? PlantTypeName { get; set; }
    public string MethodUsed { get; set; } = "";
    public int CandidateInstanceId { get; set; }
    public int CandidateCostBefore { get; set; }
    public int CandidateCostAfter { get; set; }
    public string? CandidateParentName { get; set; }
    public string? CandidateContainerName { get; set; }
    public string? CandidateHierarchyPath { get; set; }
    public bool ClickInvoked { get; set; }
    public int SelectedBankCountBefore { get; set; }
    public int SelectedBankCountAfter { get; set; }
    public int SelectedBankTypeCountBefore { get; set; }
    public int SelectedBankTypeCountAfter { get; set; }
    public bool SelectedBankCountIncreased { get; set; }
    public bool SelectedBankTypeCountIncreased { get; set; }
    public int DuplicateSelectionIndex { get; set; }
    public bool DuplicateCostIncreaseAccessible { get; set; }
    public bool DuplicateCostIncreaseDetected { get; set; }
    public int[] VisibleCostsBefore { get; set; } = Array.Empty<int>();
    public int[] VisibleCostsAfter { get; set; } = Array.Empty<int>();
    public int[] AvailableCostsBefore { get; set; } = Array.Empty<int>();
    public int[] AvailableCostsAfter { get; set; } = Array.Empty<int>();
    public int[] SelectedBankCostsBefore { get; set; } = Array.Empty<int>();
    public int[] SelectedBankCostsAfter { get; set; } = Array.Empty<int>();
    public bool Success { get; set; }
    public string? Error { get; set; }
}

internal sealed class SeedVerificationDto
{
    public bool Success { get; set; }
    public string? Source { get; set; }
    public int[] RequestedSeedTypes { get; set; } = Array.Empty<int>();
    public int[] SelectedSeedTypes { get; set; } = Array.Empty<int>();
    public int[] MissingSeedTypes { get; set; } = Array.Empty<int>();
    public PlantTypeCountDto[] RequestedPlantTypeCounts { get; set; } = Array.Empty<PlantTypeCountDto>();
    public PlantTypeCountDto[] SelectedPlantTypeCounts { get; set; } = Array.Empty<PlantTypeCountDto>();
}

internal sealed class PlantDto
{
    public int Index { get; set; }
    public int InstanceId { get; set; }
    public int Type { get; set; }
    public string? TypeName { get; set; }
    public int Row { get; set; }
    public int Column { get; set; }
    public int Health { get; set; }
    public int MaxHealth { get; set; }
    public int Level { get; set; }
    public float AttackCooldown { get; set; }
    public float ProduceCooldown { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
}

internal sealed class VisiblePlantDto
{
    public int InstanceId { get; set; }
    public int Type { get; set; }
    public string? TypeName { get; set; }
    public int Row { get; set; }
    public int Column { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public bool ActiveInHierarchy { get; set; }
    public bool InBoardBounds { get; set; }
    public bool InPlantArray { get; set; }
}

internal sealed class VisibleMowerDto
{
    public int InstanceId { get; set; }
    public int Type { get; set; }
    public string? TypeName { get; set; }
    public int Row { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
    public float MowerX { get; set; }
    public bool ActiveInHierarchy { get; set; }
    public bool InBoardBounds { get; set; }
    public bool InMowerArray { get; set; }
}

internal sealed class ZombieDto
{
    public int Index { get; set; }
    public int Type { get; set; }
    public string? TypeName { get; set; }
    public int Row { get; set; }
    public int Column { get; set; }
    public int Health { get; set; }
    public int MaxHealth { get; set; }
    public int Status { get; set; }
    public string? StatusName { get; set; }
    public float Speed { get; set; }
    public bool Alive { get; set; }
    public float X { get; set; }
    public float Y { get; set; }
}

internal sealed class LaneDto
{
    public int Row { get; set; }
    public int ZombieCount { get; set; }
    public float? NearestZombieX { get; set; }
    public int? NearestZombieHealth { get; set; }
    public int? NearestZombieType { get; set; }
    public int ConeheadCount { get; set; }
    public int BucketheadCount { get; set; }
    public int ToughZombieCount { get; set; }
    public float? ToughZombieNearestX { get; set; }
    public float ToughZombiePressureScore { get; set; }
}

internal sealed class FusionCandidateDto
{
    public int SourceInstanceId { get; set; }
    public string? SourcePlantName { get; set; }
    public int SourcePlantType { get; set; }
    public int SourceRow { get; set; }
    public int SourceCol { get; set; }
    public string? IngredientPlantName { get; set; }
    public int IngredientPlantType { get; set; }
    public int IngredientSeedSlotIndex { get; set; }
    public string? PredictedResultName { get; set; }
    public int PredictedResultType { get; set; } = -1;
    public string? PredictedResultResolutionSource { get; set; }
    public bool MixLookupFound { get; set; }
    public string? MixLookupKey { get; set; }
    public bool FusionLegal { get; set; }
    public string? FusionBlockedReason { get; set; }
}
