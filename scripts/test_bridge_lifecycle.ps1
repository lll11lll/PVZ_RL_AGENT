param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$buildScript = Join-Path $PSScriptRoot "build_bridge.ps1"
$bridgeOutDir = Join-Path $root "src\PvZRLBridge\bin\Release\net6.0"
$bridgeDll = Join-Path $bridgeOutDir "PvZRLBridge.dll"
$harnessSource = Join-Path $PSScriptRoot "bridge_lifecycle_harness.cs"
$harnessOutDir = Join-Path $bridgeOutDir "lifecycle-harness"
$harnessDll = Join-Path $harnessOutDir "PvZRLBridgeLifecycleHarness.dll"
$runtimeConfig = Join-Path $harnessOutDir "PvZRLBridgeLifecycleHarness.runtimeconfig.json"
$csc = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\Roslyn\csc.exe"
$net6 = "C:\Program Files\dotnet\shared\Microsoft.NETCore.App\6.0.6"

$buildOutput = & $buildScript 2>&1
$buildExitCode = $LASTEXITCODE
$buildOutput | ForEach-Object { Write-Host $_ }
if ($buildExitCode -ne 0) {
    throw "Bridge build failed before the lifecycle harness."
}
if (($buildOutput | Out-String) -match "\bwarning CS\d+") {
    throw "Bridge build emitted a C# warning; lifecycle verification requires zero warnings."
}

if (-not (Test-Path $csc)) {
    throw "C# compiler was not found at $csc"
}
if (-not (Test-Path $bridgeDll)) {
    throw "Expected bridge build output missing: $bridgeDll"
}

New-Item -ItemType Directory -Force -Path $harnessOutDir | Out-Null

function Test-ManagedAssembly($path) {
    try {
        [void][System.Reflection.AssemblyName]::GetAssemblyName($path)
        return $true
    } catch {
        return $false
    }
}

$refs = Get-ChildItem $net6 -Filter *.dll |
    Where-Object { Test-ManagedAssembly $_.FullName } |
    ForEach-Object { $_.FullName }
$refArgs = $refs | ForEach-Object { "/reference:$_" }
$refArgs += "/reference:$bridgeDll"

& $csc `
    /noconfig `
    /nostdlib+ `
    /target:exe `
    /optimize+ `
    /debug:portable `
    /langversion:10.0 `
    /nullable:enable `
    "/out:$harnessDll" `
    $refArgs `
    $harnessSource

if ($LASTEXITCODE -ne 0) {
    throw "Bridge lifecycle harness compilation failed."
}

Copy-Item -LiteralPath $bridgeDll -Destination $harnessOutDir -Force
@{
    runtimeOptions = @{
        tfm = "net6.0"
        framework = @{
            name = "Microsoft.NETCore.App"
            version = "6.0.0"
        }
    }
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $runtimeConfig -Encoding utf8

& dotnet $harnessDll
if ($LASTEXITCODE -ne 0) {
    throw "Bridge lifecycle harness failed."
}
