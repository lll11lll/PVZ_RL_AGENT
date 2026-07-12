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

[assembly: MelonInfo(typeof(PvZRLBridge.BridgeMod), "PvZRLBridge", "0.1.0", "Codex")]
[assembly: MelonGame("LanPiaoPiao", "PlantsVsZombiesRH")]
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("PvZRLBridgeLifecycleHarness")]
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("PvZRLBridgeObservationBenchmark")]

namespace PvZRLBridge;

public sealed partial class BridgeMod : MelonMod
{
    private const int DefaultPort = 32323;
    private const long DeprecatedSunSpawnCompensationApplyCount = 0;
    private const uint MouseEventLeftDown = 0x0002;
    private const uint MouseEventLeftUp = 0x0004;
    private static readonly TimeSpan ServerThreadShutdownTimeout = TimeSpan.FromSeconds(2);
    private static readonly TimeSpan ClientWorkerShutdownTimeout = TimeSpan.FromSeconds(2);
    private readonly PendingRequestQueue _pending = new();
    private readonly ActiveClientRegistry _activeClients = new();
    private readonly object _serverStateGate = new();
    private readonly BridgeConfig _config = new();
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false
    };

    private TcpListener? _listener;
    private Thread? _serverThread;
    private volatile bool _stopping;
    private long _nextRequestId;
    private int _loggedBoardReady;
    private readonly SeedRuntimeCache _seedRuntimeCache = new();
    private readonly RestartUiCache _restartUiCache = new();
    private readonly Dictionary<string, FusionPredictionInfo> _fusionPredictionCache = new(StringComparer.Ordinal);
    private int _observationsSinceSeedProbe;
    private bool _hasOriginalSpeed;
    private float _originalGameSpeed = 1f;
    private float _originalTimeScale = 1f;
    private float _originalFixedDeltaTime = 0.02f;
    private long _bridgeUpdateLoopCount;
    private long _speedApplyCount;
    private long _validSpeedModeApplyCount;
    private long _resetCount;
    private long _letsRockClickCount;
    private int _lastSunDebugFrame = -100000;
    private SunDebugSnapshot _sunDebugSnapshot = new();
    private bool _speedConfigDirty = true;
    private string _lastAppliedSpeedMode = "";
    private float _lastRequestedGameSpeed = float.NaN;
    private float _lastAppliedGameSpeed = float.NaN;
    private float _lastAppliedTimeScale = float.NaN;
    private float _lastAppliedFixedDeltaTime = float.NaN;

    [StructLayout(LayoutKind.Sequential)]
    private struct NativePoint
    {
        public int X;
        public int Y;
    }

    [DllImport("user32.dll")]
    private static extern bool ClientToScreen(IntPtr hWnd, ref NativePoint lpPoint);

    [DllImport("user32.dll")]
    private static extern IntPtr GetForegroundWindow();

    [DllImport("user32.dll")]
    private static extern bool SetCursorPos(int x, int y);

    [DllImport("user32.dll")]
    private static extern bool SetForegroundWindow(IntPtr hWnd);

    [DllImport("user32.dll")]
    private static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
}
