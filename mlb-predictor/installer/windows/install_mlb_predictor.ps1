param(
    [string]$SourceDir = (Join-Path $PSScriptRoot "MLBPredictor"),
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "Programs\MLBPredictor"),
    [switch]$NoDesktopShortcut,
    [switch]$NoStartMenuShortcut,
    [switch]$NoLaunch
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $SourceDir)) {
    throw "Source directory not found: $SourceDir"
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -Path (Join-Path $SourceDir '*') -Destination $InstallDir -Recurse -Force

$wsh = New-Object -ComObject WScript.Shell

if (-not $NoStartMenuShortcut) {
    $startMenuDir = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\MLB Predictor'
    New-Item -ItemType Directory -Force -Path $startMenuDir | Out-Null
    $shortcut = $wsh.CreateShortcut((Join-Path $startMenuDir 'MLB Predictor.lnk'))
    $shortcut.TargetPath = Join-Path $InstallDir 'MLBPredictor.exe'
    $shortcut.WorkingDirectory = $InstallDir
    $shortcut.Save()
}

if (-not $NoDesktopShortcut) {
    $desktopPath = [Environment]::GetFolderPath('Desktop')
    $shortcut = $wsh.CreateShortcut((Join-Path $desktopPath 'MLB Predictor.lnk'))
    $shortcut.TargetPath = Join-Path $InstallDir 'MLBPredictor.exe'
    $shortcut.WorkingDirectory = $InstallDir
    $shortcut.Save()
}

Copy-Item -Path (Join-Path $PSScriptRoot 'uninstall_mlb_predictor.ps1') -Destination $InstallDir -Force

Write-Host "Installed MLB Predictor to $InstallDir"

if (-not $NoLaunch) {
    Start-Process -FilePath (Join-Path $InstallDir 'MLBPredictor.exe') -WorkingDirectory $InstallDir
}