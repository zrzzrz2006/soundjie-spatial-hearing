$ErrorActionPreference = 'Stop'

$diskNumber = 2
$expectedSerial = '121220160204'
$expectedSize = 15728640000
$sourcePath = '\\.\PhysicalDrive2'
$backupRoot = 'F:\MLK\DPU_ZYNQ7000\backups\sd_full_before_debian_restore'

$disk = Get-Disk -Number $diskNumber
$actualSerial = [string]$disk.SerialNumber
if ($actualSerial.Trim() -ne $expectedSerial) {
    throw "Refusing to read Disk ${diskNumber}: serial '$actualSerial' does not match '$expectedSerial'."
}
if ([int64]$disk.Size -ne $expectedSize) {
    throw "Refusing to read Disk ${diskNumber}: size '$($disk.Size)' does not match '$expectedSize'."
}

New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$imagePath = Join-Path $backupRoot "physicaldrive2_petalinux5_before_restore_$stamp.img"
$hashPath = "$imagePath.sha256.txt"
$manifestPath = "$imagePath.manifest.txt"
if (Test-Path -LiteralPath $imagePath) {
    throw "Refusing to overwrite existing image: $imagePath"
}

$source = [System.IO.File]::Open($sourcePath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
$target = [System.IO.File]::Open($imagePath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
$sha = [System.Security.Cryptography.SHA256]::Create()
$buffer = New-Object byte[] (8MB)
$copied = [int64]0
$lastProgress = -1

try {
    while (($read = $source.Read($buffer, 0, $buffer.Length)) -gt 0) {
        $target.Write($buffer, 0, $read)
        [void]$sha.TransformBlock($buffer, 0, $read, $null, 0)
        $copied += $read
        $progress = [int](($copied * 100) / $expectedSize)
        if ($progress -ge ($lastProgress + 5)) {
            Write-Host "Backup progress: $progress% ($copied / $expectedSize bytes)"
            $lastProgress = $progress
        }
    }
    [void]$sha.TransformFinalBlock([byte[]]::new(0), 0, 0)
}
finally {
    $target.Flush()
    $target.Dispose()
    $source.Dispose()
}

if ($copied -ne $expectedSize) {
    throw "Backup size mismatch: copied $copied bytes, expected $expectedSize."
}

$hash = [Convert]::ToHexString($sha.Hash).ToLowerInvariant()
Set-Content -LiteralPath $hashPath -Value "$hash  $([System.IO.Path]::GetFileName($imagePath))" -Encoding ascii
@(
    "source=$sourcePath"
    "disk_number=$diskNumber"
    "disk_serial=$actualSerial"
    "disk_size=$expectedSize"
    "image=$imagePath"
    "image_size=$copied"
    "sha256=$hash"
    "completed=$(Get-Date -Format o)"
) | Set-Content -LiteralPath $manifestPath -Encoding utf8

Write-Host "BACKUP_COMPLETE"
Write-Host "IMAGE=$imagePath"
Write-Host "SHA256=$hash"
Write-Host 'You may close this window.'
Read-Host 'Press Enter to close'
