param(
    [string]$Source = "$(Resolve-Path "$PSScriptRoot\..\..\external\taichi_for_infernux")",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

# Keep the compiler build reproducible for contributors and CI.  PowerShell
# can expose cl.exe without exporting MSVC's INCLUDE/LIB variables; invoking
# the VS developer command file in the same cmd process supplies the complete
# standard-library environment before CMake/Ninja starts.
$vsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
$condaActivate = "C:\ProgramData\anaconda3\Scripts\activate.bat"
if (-not (Test-Path -LiteralPath $vsDevCmd)) {
    throw "Visual Studio developer command file not found: $vsDevCmd"
}
if (-not (Test-Path -LiteralPath $condaActivate)) {
    throw "Conda activation script not found: $condaActivate"
}

$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$cleanCommand = if ($Clean) { " && cmake --build --preset infernux-jit --clean-first" } else { "" }
$command = "call `"$vsDevCmd`" -arch=x64 -host_arch=x64 && call `"$condaActivate`" infernux && cmake --preset infernux-jit && cmake --build --preset infernux-jit$cleanCommand"
$process = Start-Process -FilePath "cmd.exe" -ArgumentList "/d", "/s", "/c", $command -WorkingDirectory $sourcePath -NoNewWindow -Wait -PassThru
if ($process.ExitCode -ne 0) {
    throw "Taichi Infernux JIT build failed with exit code $($process.ExitCode)"
}
