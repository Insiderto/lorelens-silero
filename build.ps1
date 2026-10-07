# Builds silero_voice.dll and gathers everything the voice needs into package/:
# the DLL, libtorch's DLLs and the exported model. With -Install the package is also
# copied to where LoreLens looks for it (voices\silero next to its config.toml).
param([switch]$Install)
$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    & "$PSScriptRoot\vendor.ps1"
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
    foreach ($file in "tts.jit", "accentor.jit", "homosolver.jit", "data.json") {
        if (-not (Test-Path "model\$file")) { throw "model\$file is missing: run export_silero.py first (see README.md)" }
    }

    New-Item -ItemType Directory -Force package | Out-Null
    Copy-Item "target\release\silero_voice.dll" package -Force
    $lib = Get-ChildItem "target\release\build\torch-sys-*\out\libtorch\libtorch\lib" -Directory |
        Sort-Object LastWriteTime | Select-Object -Last 1
    # torch_python.dll and shm.dll belong to the Python bindings and are not needed.
    Get-ChildItem $lib.FullName -Filter *.dll | Where-Object { $_.Name -notin "torch_python.dll", "shm.dll" } |
        Copy-Item -Destination package -Force
    Copy-Item "model\tts.jit", "model\accentor.jit", "model\homosolver.jit", "model\data.json" package -Force
    Write-Host "Package: $PSScriptRoot\package"

    if ($Install) {
        $voices = Join-Path $env:APPDATA "LoreLens\voices\silero"
        New-Item -ItemType Directory -Force $voices | Out-Null
        Copy-Item "package\*" $voices -Force
        Write-Host "Installed: $voices"
    }
} finally {
    Pop-Location
}
