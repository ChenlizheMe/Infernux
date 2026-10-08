param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$')]
    [string]$Version
)

$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$ReleaseRoot = [IO.Path]::GetFullPath((Join-Path $Root 'dist\releases'))
$ReleaseDir = [IO.Path]::GetFullPath((Join-Path $ReleaseRoot $Version))
if (-not $ReleaseDir.StartsWith($ReleaseRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe release output path: $ReleaseDir"
}

Set-Location $Root
$IdentityJson = & python (Join-Path $PSScriptRoot 'local_release_identity.py') --version $Version
if ($LASTEXITCODE -ne 0) { throw 'Could not resolve the requested release identity.' }
$Identity = $IdentityJson | ConvertFrom-Json
$BuildNumber = $Identity.build_number
$HubVersion = $Identity.hub_version

New-Item -ItemType Directory -Path $ReleaseDir -Force | Out-Null

Write-Host '[1/4] Configuring the Windows release preset...' -ForegroundColor Cyan
& cmake --preset windows-msvc-release
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }

Write-Host '[2/4] Building the Windows engine wheel...' -ForegroundColor Cyan
& cmake --build --preset windows-msvc-wheel --parallel
if ($LASTEXITCODE -ne 0) { throw 'Release wheel build failed.' }

Write-Host '[3/4] Building the Hub distribution and installer...' -ForegroundColor Cyan
& cmake --build --preset windows-hub-installer
if ($LASTEXITCODE -ne 0) { throw 'Hub release build failed.' }

Write-Host '[4/4] Validating local release assets...' -ForegroundColor Cyan
$RequiredNames = @(
    "infernux-$Version-$BuildNumber-cp313-cp313-win_amd64.whl",
    "InfernuxHubInstaller-$HubVersion-windows-x64.exe",
    "InfernuxHub-$HubVersion-windows-x64-full.zip",
    'InfernuxHub-windows-x64-manifest.json'
)
foreach ($Name in $RequiredNames) {
    $Asset = Join-Path $ReleaseDir $Name
    if (-not (Test-Path -LiteralPath $Asset -PathType Leaf) -or (Get-Item -LiteralPath $Asset).Length -eq 0) {
        throw "Expected a non-empty local release asset: $Name"
    }
}
$Manifest = Get-Content -LiteralPath (Join-Path $ReleaseDir 'InfernuxHub-windows-x64-manifest.json') -Raw | ConvertFrom-Json
if ($Manifest.version -cne $HubVersion -or $Manifest.platform -cne 'windows-x64' -or
    $Manifest.product -cne 'InfernuxHub' -or $Manifest.'$schema' -cne 'infernux.hub_update') {
    throw "Local Hub manifest does not describe InfernuxHub $HubVersion for windows-x64."
}

Get-ChildItem -LiteralPath $ReleaseDir -File | Sort-Object Name | ForEach-Object {
    Write-Host ("  {0,-72} {1,10:N1} MB" -f $_.Name, ($_.Length / 1MB))
}
Write-Host 'Local artifacts are ready. Official releases must use .github/workflows/release.yml so SignPath can verify their origin and approve signing.' -ForegroundColor Green
