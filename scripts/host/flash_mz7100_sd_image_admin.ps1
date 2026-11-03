param(
    [string]$ImagePath = 'F:\MLK\DPU_ZYNQ7000\artifacts\mz7100_debian9_restored_sd\mz7100_debian9_dpuv1_dpuy1_15gb.img',
    [string]$ExpectedImageHash = 'b9ce8994fda9506205c910c8ca67a23a8098a58b790f50c58a775eb920e553ae',
    [switch]$NoPause
)

$ErrorActionPreference = 'Stop'

$expectedSerial = '121220160204'
$expectedSize = [int64]15728640000
$imagePath = $ImagePath
$expectedImageHash = $ExpectedImageHash.ToLowerInvariant()
$logRoot = 'F:\MLK\DPU_ZYNQ7000\logs'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $logRoot "flash_mz7100_sd_$stamp.log"

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

public static class Mz7100RawVolume {
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    public static extern SafeFileHandle CreateFile(
        string name,
        uint desiredAccess,
        uint shareMode,
        IntPtr securityAttributes,
        uint creationDisposition,
        uint flagsAndAttributes,
        IntPtr templateFile);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool DeviceIoControl(
        SafeFileHandle device,
        uint controlCode,
        IntPtr inputBuffer,
        uint inputBufferSize,
        IntPtr outputBuffer,
        uint outputBufferSize,
        out uint bytesReturned,
        IntPtr overlapped);
}
'@

function Lock-DiskVolumes {
    param([Parameter(Mandatory=$true)][int]$DiskNumber)

    $volumePaths = New-Object System.Collections.Generic.List[string]
    foreach ($partition in Get-Partition -DiskNumber $DiskNumber -ErrorAction SilentlyContinue) {
        foreach ($accessPath in @($partition.AccessPaths)) {
            if ($accessPath -match '^[A-Z]:\\$') {
                Write-Host "Removing mounted access path $accessPath"
                & mountvol.exe $accessPath /p
                if ($LASTEXITCODE -ne 0) { throw "mountvol failed for $accessPath with exit code $LASTEXITCODE" }
            } elseif ($accessPath -match '^\\\\\?\\Volume\{[^}]+\}\\$') {
                [void]$volumePaths.Add($accessPath)
            }
        }
    }

    $handles = New-Object System.Collections.Generic.List[Microsoft.Win32.SafeHandles.SafeFileHandle]
    try {
        foreach ($volumePath in $volumePaths) {
            $rawPath = $volumePath.TrimEnd('\').Replace('\\?\', '\\.\')
            Write-Host "Locking and dismounting $rawPath"
            $handle = [Mz7100RawVolume]::CreateFile(
                $rawPath,
                [uint32]3221225472,
                [uint32]0x00000003,
                [IntPtr]::Zero,
                [uint32]3,
                [uint32]0,
                [IntPtr]::Zero)
            if ($handle.IsInvalid) {
                $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
                throw "CreateFile failed for $rawPath with Win32 error $errorCode"
            }

            $bytesReturned = [uint32]0
            $locked = [Mz7100RawVolume]::DeviceIoControl(
                $handle, [uint32]0x00090018,
                [IntPtr]::Zero, 0, [IntPtr]::Zero, 0,
                [ref]$bytesReturned, [IntPtr]::Zero)
            if (-not $locked) {
                $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
                $handle.Dispose()
                throw "FSCTL_LOCK_VOLUME failed for $rawPath with Win32 error $errorCode"
            }

            $dismounted = [Mz7100RawVolume]::DeviceIoControl(
                $handle, [uint32]0x00090020,
                [IntPtr]::Zero, 0, [IntPtr]::Zero, 0,
                [ref]$bytesReturned, [IntPtr]::Zero)
            if (-not $dismounted) {
                $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
                $handle.Dispose()
                throw "FSCTL_DISMOUNT_VOLUME failed for $rawPath with Win32 error $errorCode"
            }
            [void]$handles.Add($handle)
        }
        return @($handles)
    }
    catch {
        foreach ($handle in $handles) { $handle.Dispose() }
        throw
    }
}

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Copy-ExactWithHash {
    param(
        [Parameter(Mandatory=$true)][string]$Source,
        [Parameter(Mandatory=$true)][string]$Destination,
        [Parameter(Mandatory=$true)][int64]$Length,
        [Parameter(Mandatory=$true)][bool]$DestinationMustBeNew,
        [Parameter(Mandatory=$true)][string]$Label
    )

    $sourceStream = [System.IO.File]::Open($Source, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    if ($DestinationMustBeNew) {
        $targetStream = [System.IO.File]::Open($Destination, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    } else {
        $targetStream = [System.IO.File]::Open($Destination, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
    }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $buffer = New-Object byte[] (8MB)
    $copied = [int64]0
    $nextProgress = 0
    $resultHash = $null
    try {
        while ($copied -lt $Length) {
            $remaining = $Length - $copied
            $wanted = if ($remaining -gt [int64]$buffer.Length) { $buffer.Length } else { [int]$remaining }
            $read = $sourceStream.Read($buffer, 0, $wanted)
            if ($read -le 0) { throw "$Label ended early at $copied of $Length bytes." }
            $targetStream.Write($buffer, 0, $read)
            [void]$sha.TransformBlock($buffer, 0, $read, $null, 0)
            $copied += $read
            $progress = [int](($copied * 100) / $Length)
            if ($progress -ge $nextProgress) {
                Write-Host "$Label progress: $progress%"
                $nextProgress += 5
            }
        }
        [void]$sha.TransformFinalBlock([byte[]]::new(0), 0, 0)
        $resultHash = ([BitConverter]::ToString($sha.Hash)).Replace('-', '').ToLowerInvariant()
        $targetStream.Flush($true)
    }
    finally {
        $targetStream.Dispose()
        $sourceStream.Dispose()
        $sha.Dispose()
    }
    return $resultHash
}

function Get-RawHash {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][int64]$Length,
        [Parameter(Mandatory=$true)][string]$Label
    )
    $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $buffer = New-Object byte[] (8MB)
    $readTotal = [int64]0
    $nextProgress = 0
    try {
        while ($readTotal -lt $Length) {
            $remaining = $Length - $readTotal
            $wanted = if ($remaining -gt [int64]$buffer.Length) { $buffer.Length } else { [int]$remaining }
            $read = $stream.Read($buffer, 0, $wanted)
            if ($read -le 0) { throw "$Label ended early at $readTotal of $Length bytes." }
            [void]$sha.TransformBlock($buffer, 0, $read, $null, 0)
            $readTotal += $read
            $progress = [int](($readTotal * 100) / $Length)
            if ($progress -ge $nextProgress) {
                Write-Host "$Label progress: $progress%"
                $nextProgress += 5
            }
        }
        [void]$sha.TransformFinalBlock([byte[]]::new(0), 0, 0)
        return ([BitConverter]::ToString($sha.Hash)).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $stream.Dispose()
        $sha.Dispose()
    }
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
Start-Transcript -LiteralPath $logPath -Force | Out-Null
$lockedVolumeHandles = @()

try {
    if (-not (Test-Administrator)) { throw 'This script must be run as Administrator.' }
    if (-not (Test-Path -LiteralPath $imagePath)) { throw "Restore image is missing: $imagePath" }
    $image = Get-Item -LiteralPath $imagePath
    if ([int64]$image.Length -ne $expectedSize) { throw "Restore image size is $($image.Length), expected $expectedSize." }

    Write-Host 'Verifying the complete restore image before touching the SD card.'
    $imageHash = (Get-FileHash -LiteralPath $imagePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($imageHash -ne $expectedImageHash) { throw "Restore image hash mismatch: $imageHash" }
    Write-Host "Restore image verified: $imageHash"

    $matches = @(Get-Disk | Where-Object {
        ([string]$_.SerialNumber).Trim() -eq $expectedSerial -and [int64]$_.Size -eq $expectedSize
    })
    if ($matches.Count -ne 1) {
        throw "Expected exactly one SD disk with serial $expectedSerial and size $expectedSize; found $($matches.Count)."
    }
    $disk = $matches[0]
    $diskNumber = $disk.Number
    $physicalPath = "\\.\PhysicalDrive$diskNumber"
    Write-Host "Exact SD target: Disk $diskNumber, serial $expectedSerial, $expectedSize bytes"

    if ($disk.IsOffline) {
        Set-Disk -Number $diskNumber -IsOffline $false
        Start-Sleep -Seconds 2
    }
    $lockedVolumeHandles = @(Lock-DiskVolumes -DiskNumber $diskNumber)
    Write-Host "Locked volume handles: $($lockedVolumeHandles.Count)"
    Start-Sleep -Seconds 2

    Write-Host 'Current-card backup skipped by explicit user request.'
    Write-Host 'Writing the verified Debian 9 + DPUV1 + DPUY1 image to the SD card.'
    $writeHash = Copy-ExactWithHash -Source $imagePath -Destination $physicalPath -Length $expectedSize -DestinationMustBeNew $false -Label 'SD write'
    if ($writeHash -ne $expectedImageHash) { throw "Source hash changed during write: $writeHash" }

    Write-Host 'Reading the complete SD card back for byte-for-byte verification.'
    $readbackHash = Get-RawHash -Path $physicalPath -Length $expectedSize -Label 'SD readback'
    if ($readbackHash -ne $expectedImageHash) { throw "SD readback hash mismatch: $readbackHash" }

    Write-Host ''
    Write-Host 'RESTORE_COMPLETE' -ForegroundColor Green
    Write-Host "SD readback SHA256: $readbackHash"
    Write-Host 'All mounted volumes were removed and every raw write was flushed; it is safe to remove after closing this window.'
}
catch {
    Write-Host ''
    Write-Host 'RESTORE_FAILED' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Full log: $logPath" -ForegroundColor Yellow
}
finally {
    foreach ($handle in $lockedVolumeHandles) {
        if ($null -ne $handle -and -not $handle.IsClosed) { $handle.Dispose() }
    }
    Stop-Transcript | Out-Null
    if (-not $NoPause) { Read-Host 'Press Enter to close' }
}

