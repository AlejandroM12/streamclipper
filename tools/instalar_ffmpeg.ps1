# Descarga el build portable de FFmpeg (gyan.dev, "essentials": incluye libx264 y libass) en tools\ffmpeg.
# No modifica el PATH ni instala nada global.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$dest = Join-Path $root "ffmpeg"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("streamclipper-ffmpeg-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force $tmp | Out-Null
try {
    $zip = Join-Path $tmp "ffmpeg.zip"
    Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $zip -UseBasicParsing
    Expand-Archive $zip (Join-Path $tmp "x") -Force
    $bin = Get-ChildItem (Join-Path $tmp "x") -Directory | Select-Object -First 1
    New-Item -ItemType Directory -Force $dest | Out-Null
    Copy-Item (Join-Path $bin.FullName "bin\*") $dest -Force
    Write-Host "FFmpeg instalado en $dest"
} finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
