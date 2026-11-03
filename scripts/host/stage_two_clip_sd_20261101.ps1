# One-session migration, pinned to the already authorized SD card.
# Archive only this project's obsolete D3 test pair before freeing space.
# Does not format, repartition, touch raw sectors, or change boot configuration.
$ErrorActionPreference = 'Stop'
$sdRoot = '\\?\Volume{9b6c8f5a-bf04-11f1-9fa3-005056c00008}\'
$artifact = 'F:\MLK\DPU_ZYNQ7000\artifacts\soundjie_two_clips\20261101a'
$source = Join-Path $artifact 'release-v4-live'
$output = Join-Path $artifact 'sd-persistence-20261101'
$backup = Join-Path $output 'archived-d3'
$python = 'F:\MLK\DPU_ZYNQ7000\envs\soundjie_d4_host\Scripts\python.exe'
$repo = 'E:\Soundjie_Work\d4'
$expectedNew = '0e89305ba1822fe5b8f7a41fb0b6919b09b1212eca0e458b9e51046bb762d7af'
$expectedOld = '6e2a2a300b037cb6187cdc1dd718f28be3ff3e4ed3695f9890d552e333a4d95c'
$names = @('SJD3L1.TGZ', 'SJD3L1.SHA')

function Save-NewJson($path, $value) {
    $bytes = [Text.Encoding]::UTF8.GetBytes(($value | ConvertTo-Json -Depth 12) + "`n")
    $stream = [IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try { $stream.Write($bytes,0,$bytes.Length); $stream.Flush($true) }
    finally { $stream.Dispose() }
}
function Assert-SdIdentity {
    $disk = Get-Disk -Number 2
    if ($disk.SerialNumber -ne '121220160204' -or $disk.Size -ne 15728640000 -or $disk.BusType -ne 'USB' -or $disk.IsBoot -or $disk.IsSystem) { throw 'SD identity changed.' }
    $part = Get-Partition -DiskNumber 2 -PartitionNumber 1
    if ($part.Offset -ne 1048576 -or $part.Size -ne 103809024 -or $sdRoot -notin $part.AccessPaths) { throw 'SD partition differs.' }
    $vol = Get-Volume -UniqueId $sdRoot
    if ($vol.FileSystem -ne 'FAT32' -or $vol.FileSystemLabel -ne 'boot' -or $vol.HealthStatus -ne 'Healthy' -or [string]$vol.OperationalStatus -ne 'OK') { throw 'SD filesystem differs.' }
}

Assert-SdIdentity
if (Test-Path -LiteralPath $backup) { throw 'Archive destination exists; inspect partial state before retry.' }
foreach ($name in @('SJ6V1.TGZ','SJ6V1.SHA')) {
    if (Test-Path -LiteralPath ($sdRoot+$name)) { throw 'New SD destination already exists.' }
}
if ((Get-FileHash -LiteralPath (Join-Path $source 'SJ6V1.TGZ') -Algorithm SHA256).Hash -ine $expectedNew) { throw 'New release changed.' }
if ((Get-FileHash -LiteralPath ($sdRoot+'SJD3L1.TGZ') -Algorithm SHA256).Hash -ine $expectedOld) { throw 'Old D3 artifact differs.' }
$oldManifest = [IO.File]::ReadAllText($sdRoot+'SJD3L1.SHA').Trim()
if ($oldManifest -cne "$expectedOld  SJD3L1.TGZ") { throw 'Old D3 manifest differs.' }
$newManifest = [IO.File]::ReadAllText((Join-Path $source 'SJ6V1.SHA')).Trim()
if ($newManifest -cne "$expectedNew  SJ6V1.TGZ") { throw 'New manifest differs.' }
$before = Get-Content -LiteralPath (Join-Path $output 'preflight-root-inventory.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$current = @(Get-ChildItem -LiteralPath $sdRoot -File -Force)
if ($current.Count -ne $before.file_count) { throw 'SD root file inventory changed.' }
foreach ($file in $current) {
    $old = $before.files.PSObject.Properties[$file.Name].Value
    if ($null -eq $old -or $file.Length -ne $old.size -or (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash -ine $old.sha256) { throw "SD source changed: $($file.Name)" }
}
$bytesRequired = (Get-Item -LiteralPath (Join-Path $source 'SJ6V1.TGZ')).Length + (Get-Item -LiteralPath (Join-Path $source 'SJ6V1.SHA')).Length + 1048576
$afterArchiveFree = (Get-Volume -UniqueId $sdRoot).SizeRemaining
foreach ($name in $names) { $afterArchiveFree += (Get-Item -LiteralPath ($sdRoot+$name)).Length }
if ($afterArchiveFree -lt $bytesRequired) { throw 'Planned archive does not provide sufficient space.' }

[void](New-Item -ItemType Directory -Path $backup)
$records = @()
foreach ($name in $names) {
    $from = $sdRoot+$name
    $to = Join-Path $backup $name
    $inputStream = [IO.File]::Open($from,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
    try {
        $outputStream = [IO.File]::Open($to,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
        try { $inputStream.CopyTo($outputStream); $outputStream.Flush($true) }
        finally { $outputStream.Dispose() }
    } finally { $inputStream.Dispose() }
    $sourceHash = (Get-FileHash -LiteralPath $from -Algorithm SHA256).Hash.ToLowerInvariant()
    $copyHash = (Get-FileHash -LiteralPath $to -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sourceHash -cne $copyHash -or (Get-Item -LiteralPath $from).Length -ne (Get-Item -LiteralPath $to).Length) { throw 'Archive copy readback failed.' }
    $records += [ordered]@{ Name=$name; Source=$from; Backup=$to; Bytes=(Get-Item -LiteralPath $to).Length; SHA256=$copyHash }
}
Save-NewJson (Join-Path $output 'archive-copy-verified.json') ([ordered]@{ Status='both_obsolete_generated_artifacts_backed_up_before_removal'; DiskSerial='121220160204'; Files=$records })

Assert-SdIdentity
foreach ($record in $records) {
    # Exact non-recursive files only, with source and backup rechecked immediately.
    if ($record.Source -cne ($sdRoot+$record.Name) -or $record.Name -notin $names) { throw 'Removal path differs.' }
    if ((Get-FileHash -LiteralPath $record.Source -Algorithm SHA256).Hash -ine $record.SHA256 -or (Get-FileHash -LiteralPath $record.Backup -Algorithm SHA256).Hash -ine $record.SHA256) { throw 'Archive changed before removal.' }
    Remove-Item -LiteralPath $record.Source
    if (Test-Path -LiteralPath $record.Source) { throw 'Exact obsolete file was not removed.' }
}
Save-NewJson (Join-Path $output 'archive-removal-receipt.json') ([ordered]@{ RemovedFromSD=$names; VerifiedArchive=$backup; FreeBytesAfterArchive=(Get-Volume -UniqueId $sdRoot).SizeRemaining; OtherFilesRemoved=0 })

Assert-SdIdentity
& $python -B (Join-Path $repo 'scripts\host\stage_additive_inference_sd.py') --source-dir $source --source-manifest (Join-Path $source 'SJ6V1.SHA') --destination SJ6V1.TGZ --destination SJ6V1.SHA --sd-root $sdRoot --expected-volume-serial 55DE-86B8 --report (Join-Path $output 'copy-readback.json')
if ($LASTEXITCODE -ne 0) { throw 'New package staging failed; inspect receipts; do not eject blindly.' }

$expectedNames = @($before.files.PSObject.Properties.Name | Where-Object { $_ -notin $names }) + @('SJ6V1.TGZ','SJ6V1.SHA')
$after = @(Get-ChildItem -LiteralPath $sdRoot -File -Force)
if (@(Compare-Object ($expectedNames | Sort-Object) ($after.Name | Sort-Object)).Count -ne 0) { throw 'Unexpected SD files after stage.' }
$unchanged = 0
foreach ($file in $after) {
    if ($file.Name -in @('SJ6V1.TGZ','SJ6V1.SHA')) { continue }
    $old = $before.files.PSObject.Properties[$file.Name].Value
    if ($file.Length -ne $old.size -or (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash -ine $old.sha256) { throw "Original file changed: $($file.Name)" }
    $unchanged++
}
Assert-SdIdentity
$report = [ordered]@{ Result='SD_APPLICATION_STAGE_READBACK_OK'; NewFiles=@('SJ6V1.TGZ','SJ6V1.SHA'); ArchiveOnlyRemoved=$names; UnchangedOriginalFiles=$unchanged; BootFilesUnchanged=$true; PreviousSJ4R1Retained=$true; FreeBytesAfter=(Get-Volume -UniqueId $sdRoot).SizeRemaining; RawSectorWrites=$false; FormatOrPartitionChanges=$false; StartupChanged=$false; SafeEjectPending=$true; BoardReloadVerified=$false }
Save-NewJson (Join-Path $output 'post-stage-verification.json') $report
$report | ConvertTo-Json -Depth 4
