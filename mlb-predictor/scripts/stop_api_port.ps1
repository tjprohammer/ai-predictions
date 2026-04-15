# Stop whatever is listening on the MLB Predictor API port (default 8000).
# Run:  .\scripts\stop_api_port.ps1
# Use -WhatIf to only list listeners.
#
# Note: Windows sometimes lists LISTENING rows with OwningProcess PIDs that no longer
# exist (stale TCP table). This script uses a TCP connect probe as the final truth.

param(
    [int] $Port = 8000,
    [switch] $WhatIf
)

$ErrorActionPreference = 'Continue'

function Test-ProcessRecordExists {
    param([int] $ProcessId)
    if (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) { return $true }
    return [bool](Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue)
}

function Test-LocalTcpPortOpen {
    param([int] $ListenPort)
    # True if something accepts connections on 127.0.0.1 (authoritative for "can I bind?").
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect([string]'127.0.0.1', $ListenPort, $null, $null)
        if (-not $iar.AsyncWaitHandle.WaitOne(2000, $false)) {
            $client.Close()
            return $false
        }
        try {
            $client.EndConnect($iar)
        } catch {
            $client.Close()
            return $false
        }
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

function Get-ListenerPids {
    param([int] $ListenPort)
    $found = [System.Collections.Generic.HashSet[int]]::new()

    Get-NetTCPConnection -LocalPort $ListenPort -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object {
            $op = [int]$_.OwningProcess
            if ($op -gt 0) { [void]$found.Add($op) }
        }

    $ns = netstat -ano 2>$null
    if ($ns) {
        $pattern = ":$ListenPort\s+.*LISTENING\s+(\d+)\s*$"
        foreach ($line in $ns) {
            if ($line -match $pattern) {
                $pidVal = [int]$Matches[1]
                if ($pidVal -gt 0) { [void]$found.Add($pidVal) }
            }
        }
    }

    return @($found | Sort-Object -Unique)
}

function Write-PortStillBusyDiagnosis {
    param([int] $ListenPort)

    Write-Host ""
    Write-Host "--- Likely causes (TCP works but taskkill cannot see a process) ---" -ForegroundColor Cyan

    Write-Host "1) Cursor / VS Code forwarded this port (very common)." -ForegroundColor White
    Write-Host "   In Cursor: open the Ports view (bottom status bar or Command Palette -> 'Ports: Focus on Ports View')." -ForegroundColor Gray
    Write-Host "   If $ListenPort appears, click the trash icon to STOP forwarding (that frees localhost:$ListenPort )." -ForegroundColor Gray
    Write-Host "   Same in VS Code: Ports panel -> delete the row for $ListenPort ." -ForegroundColor Gray

    Write-Host "2) Windows port proxy (rare):" -ForegroundColor White
    try {
        $px = netsh interface portproxy show all 2>$null
        if ($px -and ($px -match '\d')) {
            Write-Host $px -ForegroundColor Gray
        } else {
            Write-Host "   (none from: netsh interface portproxy show all)" -ForegroundColor DarkGray
        }
    } catch { }

    Write-Host "3) WSL / Docker sometimes hide the owner PID from netstat." -ForegroundColor White
    Write-Host "   Try: wsl --shutdown   (then retry your app / this script)" -ForegroundColor Gray

    Write-Host "4) See the real listener (run PowerShell as Administrator for best results):" -ForegroundColor White
    Write-Host "   resmon.exe  -> Network tab -> Listening Ports -> filter $ListenPort" -ForegroundColor Gray
    Write-Host "   or:  netstat -abno | findstr :$ListenPort" -ForegroundColor Gray

    Write-Host ""
    Write-Host "Current TCP rows for this port (State / Address / PID):" -ForegroundColor Cyan
    Get-NetTCPConnection -LocalPort $ListenPort -ErrorAction SilentlyContinue |
        Select-Object State, LocalAddress, LocalPort, OwningProcess |
        Format-Table -AutoSize
}

function Write-ListenerDetails {
    param([int[]] $Pids)
    foreach ($procId in $Pids) {
        $gp = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($gp) {
            $path = ''
            try { $path = $gp.Path } catch { }
            Write-Host ("PID {0,-8} {1,-20} {2}" -f $procId, $gp.ProcessName, $path)
            continue
        }
        $p = Get-CimInstance Win32_Process -Filter "ProcessId = $procId" -ErrorAction SilentlyContinue
        if ($p) {
            $cmd = if ($p.CommandLine) { ($p.CommandLine -replace '\s+', ' ') } else { '' }
            if ($cmd.Length -gt 160) { $cmd = $cmd.Substring(0, 160) + '...' }
            Write-Host ("PID {0,-8} {1,-16} {2}" -f $procId, $p.Name, $cmd)
        } else {
            Write-Host "PID $procId  (no process with this ID - likely stale TCP table row)" -ForegroundColor DarkYellow
        }
    }
}

Write-Host "=== TCP listeners on port $Port ===" -ForegroundColor Cyan
$listenPids = Get-ListenerPids -ListenPort $Port
Write-ListenerDetails -Pids $listenPids

$ghostCount = 0
foreach ($x in $listenPids) {
    if (-not (Test-ProcessRecordExists $x)) { $ghostCount++ }
}
if ($ghostCount -gt 0) {
    Write-Host ""
    Write-Host "$ghostCount PID(s) in the TCP table do not match any running process (Windows can leave stale rows)." -ForegroundColor DarkGray
}

if (-not $listenPids -or $listenPids.Count -eq 0) {
    if (-not (Test-LocalTcpPortOpen -ListenPort $Port)) {
        Write-Host "Nothing is listening on port $Port (connect probe OK)." -ForegroundColor Green
    } else {
        Write-Host "Port $Port accepts TCP but no LISTENING row matched; try again or run as Administrator." -ForegroundColor Yellow
    }
    exit 0
}

Write-Host ""
$resolvable = @($listenPids | Where-Object { Test-ProcessRecordExists $_ })
if ($resolvable.Count -gt 0) {
    Write-Host "PIDs to stop (running processes): $($resolvable -join ', ')" -ForegroundColor Yellow
} else {
    Write-Host "No running process matches these PIDs; only a TCP connect probe can tell if the port is busy." -ForegroundColor Yellow
}

if ($WhatIf) {
    Write-Host "WhatIf: no processes were stopped."
    exit 0
}

foreach ($procId in $listenPids) {
    Write-Host "Stopping PID $procId (tree)..." -ForegroundColor DarkYellow
    & taskkill.exe /F /T /PID $procId 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { continue }

    if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
        } catch {
            Write-Host "Could not stop PID $procId : $_" -ForegroundColor Red
            Write-Host "  Try: PowerShell as Administrator." -ForegroundColor Yellow
        }
    } else {
        Write-Host "  taskkill: PID $procId not found (skipped)." -ForegroundColor DarkGray
    }
}

Start-Sleep -Milliseconds 800

if (-not (Test-LocalTcpPortOpen -ListenPort $Port)) {
    Write-Host ""
    Write-Host "Port $Port is free (nothing accepts TCP on 127.0.0.1:$Port )." -ForegroundColor Green
    $left = Get-ListenerPids -ListenPort $Port
    if ($left -and $left.Count -gt 0) {
        Write-Host "netstat may still show LISTENING with ghost PIDs; safe to start uvicorn." -ForegroundColor DarkGray
    }
    exit 0
}

Write-Host ""
Write-Host "Port $Port still accepts TCP connections on 127.0.0.1 (something is really there)." -ForegroundColor Red
Write-ListenerDetails -Pids (Get-ListenerPids -ListenPort $Port)
Write-PortStillBusyDiagnosis -ListenPort $Port
exit 1
