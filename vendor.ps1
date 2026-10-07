# Makes vendor/torch-sys: the torch-sys crate with the two changes Visual Studio needs.
#  - libtorch 2.13 headers want C++20, the crate asks for C++17;
#  - in C++20 a line may not start with the word "module", and the crate's C API
#    has a type of that name: it becomes "tch_module".
$ErrorActionPreference = "Stop"
$version = "0.26.0"
$target = Join-Path $PSScriptRoot "vendor\torch-sys"
if (Test-Path (Join-Path $target "Cargo.toml")) { return }

$vendor = Split-Path $target
New-Item -ItemType Directory -Force $vendor | Out-Null
$archive = Join-Path $vendor "torch-sys.crate"
Invoke-WebRequest "https://static.crates.io/crates/torch-sys/torch-sys-$version.crate" -OutFile $archive
# Windows' own tar: the one from Git takes "C:" in a path for a host name.
& (Join-Path $env:SystemRoot "System32/tar.exe") -xzf $archive -C $vendor
if ($LASTEXITCODE -ne 0) { throw "could not unpack $archive" }
Remove-Item $archive
Move-Item (Join-Path $vendor "torch-sys-$version") $target

foreach ($file in "libtch\torch_api.h", "libtch\torch_api.cpp") {
    $path = Join-Path $target $file
    (Get-Content $path -Raw) -creplace '\bmodule\b', 'tch_module' | Set-Content $path -NoNewline
}
$build = Join-Path $target "build.rs"
(Get-Content $build -Raw).Replace("/std:c++17", "/std:c++20") | Set-Content $build -NoNewline
Write-Host "vendor/torch-sys is ready"
