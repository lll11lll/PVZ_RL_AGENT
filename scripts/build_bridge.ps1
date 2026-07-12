param(
    [switch]$CopyToMods
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$generatedRegistrySource = Join-Path $root "src\PvZRLBridge\GeneratedPlantRegistry.cs"
$registryGenerator = Join-Path $PSScriptRoot "generate_bridge_registry.py"
$plantRegistry = Join-Path $root "configs\plant_registry.json"
$sourceDir = Join-Path $root "src\PvZRLBridge"
$outDir = Join-Path $root "src\PvZRLBridge\bin\Release\net6.0"
$dll = Join-Path $outDir "PvZRLBridge.dll"
$csc = "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\Roslyn\csc.exe"

if (-not (Test-Path $csc)) {
    throw "C# compiler was not found at $csc"
}

$python = (Get-Command python -ErrorAction Stop).Source
& $python $registryGenerator --registry $plantRegistry --output $generatedRegistrySource
if ($LASTEXITCODE -ne 0) {
    throw "Bridge registry generation failed."
}

$sources = @(
    Get-ChildItem -LiteralPath $sourceDir -Filter *.cs -File |
        Sort-Object -Property Name |
        ForEach-Object { $_.FullName }
)
if ($sources.Count -eq 0) {
    throw "No bridge C# sources were found under $sourceDir"
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$net6 = "C:\Program Files\dotnet\shared\Microsoft.NETCore.App\6.0.6"
$melonNet6 = Join-Path $root "Game Files\MelonLoader\net6"
$il2cpp = Join-Path $root "Game Files\MelonLoader\Il2CppAssemblies"

function Test-ManagedAssembly($path) {
    try {
        [void][System.Reflection.AssemblyName]::GetAssemblyName($path)
        return $true
    } catch {
        return $false
    }
}

$refs = @()
$refs += Get-ChildItem $net6 -Filter *.dll |
    Where-Object { Test-ManagedAssembly $_.FullName } |
    ForEach-Object { $_.FullName }
$refs += @(
    (Join-Path $melonNet6 "MelonLoader.dll"),
    (Join-Path $melonNet6 "Il2CppInterop.Runtime.dll"),
    (Join-Path $il2cpp "Il2Cppmscorlib.dll"),
    (Join-Path $il2cpp "Il2CppSystem.dll"),
    (Join-Path $il2cpp "Assembly-CSharp.dll"),
    (Join-Path $il2cpp "UnityEngine.dll"),
    (Join-Path $il2cpp "UnityEngine.CoreModule.dll"),
    (Join-Path $il2cpp "UnityEngine.Physics2DModule.dll"),
    (Join-Path $il2cpp "UnityEngine.UIModule.dll"),
    (Join-Path $il2cpp "Unity.TextMeshPro.dll")
)

$refArgs = $refs | ForEach-Object { "/reference:$_" }

& $csc `
    /noconfig `
    /nostdlib+ `
    /target:library `
    /optimize+ `
    /deterministic+ `
    /debug:portable `
    /langversion:10.0 `
    /nullable:enable `
    "/out:$dll" `
    $refArgs `
    $sources

if ($LASTEXITCODE -ne 0) {
    throw "Bridge build failed."
}

if (-not (Test-Path $dll)) {
    throw "Expected build output missing: $dll"
}

if ($CopyToMods) {
    $mods = Join-Path $root "Game Files\Mods"
    Copy-Item -LiteralPath $dll -Destination $mods -Force
    Write-Host "Copied PvZRLBridge.dll to $mods"
} else {
    Write-Host "Built $dll"
}
