param(
    [string]$OutputPath = "runs\benchmarks\phase6_bridge_pure_final.json"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$buildScript = Join-Path $PSScriptRoot "build_bridge.ps1"
$source = Join-Path $PSScriptRoot "bridge_observation_benchmark.cs"
$bridgeOutDir = Join-Path $root "src\PvZRLBridge\bin\Release\net6.0"
$bridgeDll = Join-Path $bridgeOutDir "PvZRLBridge.dll"
$benchmarkDir = Join-Path $bridgeOutDir "observation-benchmark"
$benchmarkDll = Join-Path $benchmarkDir "PvZRLBridgeObservationBenchmark.dll"
$runtimeConfig = Join-Path $benchmarkDir "PvZRLBridgeObservationBenchmark.runtimeconfig.json"
$resolvedOutput = if ([System.IO.Path]::IsPathRooted($OutputPath)) {
    [System.IO.Path]::GetFullPath($OutputPath)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $root $OutputPath))
}
$csc = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\Roslyn\csc.exe"
$net6 = "C:\Program Files\dotnet\shared\Microsoft.NETCore.App\6.0.6"

$buildOutput = & $buildScript 2>&1
$buildExitCode = $LASTEXITCODE
$buildOutput | ForEach-Object { Write-Host $_ }
if ($buildExitCode -ne 0) {
    throw "Bridge build failed before the observation benchmark."
}
if (($buildOutput | Out-String) -match "\bwarning CS\d+") {
    throw "Bridge build emitted a C# warning; benchmark requires zero warnings."
}
if (-not (Test-Path -LiteralPath $csc)) {
    throw "C# compiler was not found at $csc"
}
if (-not (Test-Path -LiteralPath $net6)) {
    throw ".NET 6 reference directory was not found at $net6"
}
if (-not (Test-Path -LiteralPath $bridgeDll)) {
    throw "Expected bridge build output missing: $bridgeDll"
}

New-Item -ItemType Directory -Force -Path $benchmarkDir | Out-Null

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
    "/out:$benchmarkDll" `
    $refArgs `
    $source

if ($LASTEXITCODE -ne 0) {
    throw "Bridge observation benchmark compilation failed."
}

Copy-Item -LiteralPath $bridgeDll -Destination $benchmarkDir -Force
@{
    runtimeOptions = @{
        tfm = "net6.0"
        framework = @{
            name = "Microsoft.NETCore.App"
            version = "6.0.0"
        }
    }
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $runtimeConfig -Encoding utf8

& dotnet $benchmarkDll $resolvedOutput
if ($LASTEXITCODE -ne 0) {
    throw "Bridge observation benchmark failed."
}
