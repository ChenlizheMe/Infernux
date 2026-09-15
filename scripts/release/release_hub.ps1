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
$ProjectText = Get-Content -LiteralPath (Join-Path $Root 'pyproject.toml') -Raw
$VersionMatch = [regex]::Match($ProjectText, '(?m)^version\s*=\s*"([^"]+)"')
if (-not $VersionMatch.Success) { throw 'Could not read project.version from pyproject.toml.' }
if ($VersionMatch.Groups[1].Value -ne $Version) {
    throw "Requested version $Version does not match pyproject.toml version $($VersionMatch.Groups[1].Value)."
}

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
$RequiredPatterns = @(
    "infernux-$Version-*-cp313-cp313-win_amd64.whl",
    "InfernuxHubInstaller-$Version-windows-x64.exe",
    "InfernuxHub-$Version-windows-x64-full.zip",
    'InfernuxHub-windows-x64-manifest.json'
)
foreach ($Pattern in $RequiredPatterns) {
    $Matches = @(Get-ChildItem -LiteralPath $ReleaseDir -Filter $Pattern -File)
    if ($Matches.Count -ne 1 -or $Matches[0].Length -eq 0) {
        throw "Expected exactly one non-empty local release asset matching: $Pattern"
    }
}

Get-ChildItem -LiteralPath $ReleaseDir -File | Sort-Object Name | ForEach-Object {
    Write-Host ("  {0,-72} {1,10:N1} MB" -f $_.Name, ($_.Length / 1MB))
}
Write-Host 'Local artifacts are ready. Official releases must use .github/workflows/release.yml so SignPath can verify their origin and approve signing.' -ForegroundColor Green
