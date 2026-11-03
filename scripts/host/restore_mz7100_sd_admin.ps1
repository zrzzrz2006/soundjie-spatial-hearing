$ErrorActionPreference = 'Stop'

$expectedSerial = '121220160204'
$expectedSize = 15728640000
$linuxScript = '/mnt/f/MLK/DPU_ZYNQ7000/usb_mic_bringup_v1/scripts/host/restore_mz7100_sd_in_wsl.sh'
$logRoot = 'F:\MLK\DPU_ZYNQ7000\logs'
$logPath = Join-Path $logRoot ("restore_mz7100_sd_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
Start-Transcript -LiteralPath $logPath -Force | Out-Null

try {
    if (-not (Test-Administrator)) {
        throw 'This script must be run as Administrator.'
    }

    $matches = @(Get-Disk | Where-Object {
        ([string]$_.SerialNumber).Trim() -eq $expectedSerial -and [int64]$_.Size -eq $expectedSize
    })
    if ($matches.Count -ne 1) {
        throw "Expected exactly one SD disk with serial $expectedSerial and size $expectedSize; found $($matches.Count)."
    }
    $disk = $matches[0]
    $physicalPath = "\\.\PHYSICALDRIVE$($disk.Number)"
    Write-Host "Exact SD target: Disk $($disk.Number), serial $expectedSerial, $expectedSize bytes"

    if ($disk.IsOffline) {
        Write-Host "Returning Disk $($disk.Number) online before a clean WSL attach"
        Set-Disk -Number $disk.Number -IsOffline $false
        Start-Sleep -Seconds 2
    }

    foreach ($partition in Get-Partition -DiskNumber $disk.Number -ErrorAction SilentlyContinue) {
        foreach ($accessPath in @($partition.AccessPaths)) {
            if ($accessPath -match '^[A-Z]:\\$') {
                Write-Host "Dismounting $accessPath before WSL takes the disk"
                & mountvol.exe $accessPath /p
            }
        }
    }

    $attached = $false
    try {
        Write-Host "Attaching $physicalPath to WSL in bare mode"
        & wsl.exe --mount $physicalPath --bare
        if ($LASTEXITCODE -ne 0) { throw "wsl --mount failed with exit code $LASTEXITCODE" }
        $attached = $true

        & wsl.exe -d Ubuntu -u root -- bash $linuxScript
        if ($LASTEXITCODE -ne 0) { throw "Linux restore script failed with exit code $LASTEXITCODE" }
    }
    finally {
        if ($attached) {
            Write-Host "Detaching $physicalPath from WSL"
            & wsl.exe --unmount $physicalPath
        }
    }

    Write-Host ''
    Write-Host 'RESTORE_COMPLETE'
    Write-Host 'The SD card now contains the restored Debian 9 baseline, DPUV1, and DPUY1 bundles.'
    Write-Host 'Close this window, then tell Codex: 恢复完成'
}
catch {
    Write-Host ''
    Write-Host 'RESTORE_FAILED' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Full log: $logPath" -ForegroundColor Yellow
}
finally {
    Stop-Transcript | Out-Null
    Read-Host 'Press Enter to close'
}

