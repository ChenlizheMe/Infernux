[CmdletBinding(SupportsShouldProcess)]
param(
    [ValidateSet('All', 'TestArtifacts')]
    [string]$Scope = 'All'
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$RootPrefix = $Root.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
$RemovedCount = 0
$RemovedBytes = 0L

function Get-GitPaths([string]$Repository, [string[]]$Arguments) {
    $Result = @(& git -C $Repository -c core.quotePath=false @Arguments)
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot inspect tracked/generated files in $Repository"
    }
    return $Result
}

function Remove-GeneratedPath([string]$Repository, [string]$RelativePath) {
    $Target = [IO.Path]::GetFullPath((Join-Path $Repository $RelativePath))
    if (-not $Target.StartsWith($RootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the workspace: $Target"
    }
    if (-not (Test-Path -LiteralPath $Target)) { return }

    # A junction at any level could redirect an otherwise local-looking path.
    $Ancestor = $Target
    while ($Ancestor -ne $Root) {
        if ((Get-Item -LiteralPath $Ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing to follow a reparse point: $Ancestor"
        }
        $Ancestor = Split-Path -Parent $Ancestor
    }
    $Tracked = @(Get-GitPaths $Repository @('ls-files', '--', $RelativePath))
    if ($Tracked.Count -gt 0) {
        throw "Refusing to remove tracked source files: $Target"
    }
    $Item = Get-Item -LiteralPath $Target -Force
    $Children = if ($Item.PSIsContainer) {
        @(Get-ChildItem -LiteralPath $Target -Recurse -Force)
    } else { @() }
    foreach ($Child in $Children) {
        if ($Child.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Refusing to remove a nested reparse point: $($Child.FullName)"
        }
    }
    if ($PSCmdlet.ShouldProcess($Target, 'Delete generated workspace files')) {
        $Size = if ($Item.PSIsContainer) {
            ($Children | Where-Object { -not $_.PSIsContainer } | Measure-Object Length -Sum).Sum
        } else { $Item.Length }
        Remove-Item -LiteralPath $Target -Recurse
        $script:RemovedCount += 1
        $script:RemovedBytes += $Size
    }
}

# Discover initialized submodules recursively. Each index protects its source
# files; a parent's index contains just the gitlink for each child repository.
$Repositories = [Collections.Generic.List[string]]::new()
$Repositories.Add($Root)
for ($Index = 0; $Index -lt $Repositories.Count; $Index++) {
    $ParentRepository = $Repositories[$Index]
    if (-not (Test-Path -LiteralPath (Join-Path $ParentRepository '.gitmodules'))) { continue }
    $SubmodulePaths = @(& git -C $ParentRepository config --file .gitmodules --get-regexp '^submodule\..*\.path$')
    if ($LASTEXITCODE -notin @(0, 1)) { throw "Cannot read submodules in $ParentRepository" }
    foreach ($Entry in $SubmodulePaths) {
        $Relative = ($Entry -split '\s+', 2)[1]
        $Repository = [IO.Path]::GetFullPath((Join-Path $ParentRepository $Relative))
        if (-not $Repository.StartsWith($RootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Submodule outside workspace: $Repository"
        }
        if ((Test-Path -LiteralPath (Join-Path $Repository '.git')) -and -not $Repositories.Contains($Repository)) {
            $Repositories.Add($Repository)
        }
    }
}

# Current and old local releases are disposable, just like assembly trees.
$GeneratedRoots = @(
    'out', 'build', 'dist', 'Library', 'mcp_captures',
    'packaging/runtime', 'packaging/Nuitka', 'packaging/_vendor',
    'packaging/InfernuxHubData', 'packaging/nuitka-crash-report.xml',
    'python/infernux.egg-info', 'python/infernux/_runtime_packs',
    'python/infernux/_runtime_modules', 'python/infernux/resources/player_runtime'
)
foreach ($Relative in $GeneratedRoots) {
    if ($Scope -eq 'All') {
        Remove-GeneratedPath $Root $Relative
    }
}

if ($Scope -eq 'TestArtifacts') {
    # These are test state, not native build trees or acceptance evidence.
    foreach ($Relative in @('.pytest_cache', '.pytest_cache-installed-wheel', 'out/cache/pytest')) {
        Remove-GeneratedPath $Root $Relative
    }
}

foreach ($Repository in $Repositories) {
    # Git enumerates ignored output without descending into .git. Do not use a
    # blanket git clean: editor settings and local release tools are not outputs.
    $Ignored = @(Get-GitPaths $Repository @('ls-files', '--others', '--ignored', '--exclude-standard', '--directory'))
    foreach ($Relative in $Ignored) {
        $Path = $Relative.Replace('\', '/')
        if ($Repository -eq $Root -and ($Path -eq 'dev' -or $Path.StartsWith('dev/'))) {
            continue
        }
        if ($Scope -eq 'TestArtifacts') {
            $TestPath = $Repository -eq $Root -and $Path -match '^tests/'
            $TestCache = $TestPath -and $Path -match '(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache)(/|$)'
            $TestBytecode = $TestPath -and $Path -match '\.(pyc|pyo)$'
            $FixtureOutput = $Repository -eq $Root -and $Path -match '^tests/fixtures/[^/]+/(Cache|Library|Logs|\.runtime)(/|$)'
            if ($TestCache -or $TestBytecode -or $FixtureOutput) {
                Remove-GeneratedPath $Repository $Relative
            }
            continue
        }
        $GeneratedDirectory = $Path -match '(^|/)(out|build|dist|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.gradle|node_modules|\.wrangler|CMakeFiles)(/|$)'
        $GeneratedDirectory = $GeneratedDirectory -or $Path -match '(^|/)[^/]+\.(egg-info|build|dist|onefile-build)(/|$)'
        $GeneratedDirectory = $GeneratedDirectory -or $Path -match '(^|/)(cmake-build-[^/]+|[^/]*_InxBuild|[^/]*_InfBuild)(/|$)'
        $PluginPayload = $Path -match '^package/editor/infernux_[^/]+/(player|tools)(/|$)'
        $CompilerPayload = $Path -match '^package/runtime/infernux_taichi(/|$)'
        $FixtureOutput = $Repository -eq $Root -and $Path -match '^tests/fixtures/[^/]+/(Cache|Library|Logs|\.runtime)(/|$)'
        $GeneratedFile = $Path -match '\.(pyc|pyo|dll|pyd|so(?:\.\d+)*|dylib|lib|pdb|o|obj|whl|inxpkg|log|tmp|bak|orig|meta)$'
        if ($GeneratedDirectory -or $PluginPayload -or $CompilerPayload -or $FixtureOutput -or $GeneratedFile) {
            Remove-GeneratedPath $Repository $Relative
        }
    }
}

if ($WhatIfPreference) {
    Write-Host 'Cleanup preview complete. No files were deleted.'
} else {
    if ($Scope -eq 'TestArtifacts') {
        Write-Host ("Deleted {0} test-generated paths ({1:N1} MiB). Build outputs and acceptance evidence were retained." -f $RemovedCount, ($RemovedBytes / 1MB))
    } else {
        Write-Host ("Deleted {0} generated paths ({1:N1} MiB). No local release archives were retained." -f $RemovedCount, ($RemovedBytes / 1MB))
    }
}
