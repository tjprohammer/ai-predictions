param(
    [string]$InstallDir = (Split-Path -Parent $MyInvocation.MyCommand.Path)
)

$ErrorActionPreference = 'Stop'

$desktopShortcut = Join-Path ([Environment]::GetFolderPath('Desktop')) 'MLB Predictor.lnk'
$startMenuDir = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\MLB Predictor'

if (Test-Path $desktopShortcut) {
    Remove-Item $desktopShortcut -Force
}
if (Test-Path $startMenuDir) {
    Remove-Item $startMenuDir -Recurse -Force
}
if (Test-Path $InstallDir) {
    Remove-Item $InstallDir -Recurse -Force
}

Write-Host "Removed MLB Predictor from $InstallDir"