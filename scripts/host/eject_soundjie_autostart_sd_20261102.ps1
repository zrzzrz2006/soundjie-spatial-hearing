# Normal removal of the explicitly verified SD reader; never force dismount.
$ErrorActionPreference='Stop'
$sdRoot='\\?\Volume{9b6c8f5a-bf04-11f1-9fa3-005056c00008}\'
$run='F:\MLK\DPU_ZYNQ7000\artifacts\soundjie_autostart\20261101a\sd-install-20261102a'
$output=Join-Path $run 'safe-eject.json'
$diskId='USBSTOR\DISK&VEN_MASS&PROD_STORAGE_DEVICE&REV_1.00\121220160204&0'
$usbId='USB\VID_14CD&PID_1212\121220160204'
if (Test-Path -LiteralPath $output) { throw 'Removal receipt already exists; inspect before retry.' }
$stage=Get-Content -LiteralPath (Join-Path $run 'installed.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($stage.result -cne 'AUTOSTART_FILES_STAGED_NOT_YET_BOOT_TESTED' -or $stage.unchanged_original_files -ne 45) { throw 'Install result is not verified.' }
$targets=@(Get-Disk | Where-Object { $_.SerialNumber.Trim() -eq '121220160204' })
if ($targets.Count -ne 1) { throw 'Target card is missing or ambiguous.' }
$card=$targets[0]
if ($card.Size -ne 15728640000 -or $card.BusType -ne 'USB' -or $card.IsBoot -or $card.IsSystem -or $card.IsOffline) { throw 'Target identity differs.' }
$partition=Get-Partition -DiskNumber $card.Number -PartitionNumber 1
if ($partition.Offset -ne 1048576 -or $partition.Size -ne 103809024 -or $sdRoot -notin $partition.AccessPaths) { throw 'Partition differs.' }
$volume=Get-Volume -UniqueId $sdRoot
if ($volume.FileSystem -ne 'FAT32' -or $volume.FileSystemLabel -ne 'boot' -or $volume.HealthStatus -ne 'Healthy') { throw 'Volume differs.' }
foreach ($entry in $stage.files.PSObject.Properties) {
    $name=$entry.Name
    if ($name.Contains('\') -or $name.Contains('/') -or $name -in @('.','..')) { throw 'Invalid receipt name.' }
    $path=$sdRoot+$name
    if ((Get-Item -LiteralPath $path).Length -ne $entry.Value.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ine $entry.Value.sha256) { throw "Readback differs: $name" }
}
if (@(Get-ChildItem -LiteralPath $sdRoot -File -Force).Count -ne @($stage.files.PSObject.Properties).Count) { throw 'Root file inventory differs.' }
$parent=(Get-PnpDeviceProperty -InstanceId $diskId -KeyName DEVPKEY_Device_Parent).Data
$children=@((Get-PnpDeviceProperty -InstanceId $usbId -KeyName DEVPKEY_Device_Children).Data)
if ($parent -ine $usbId -or $children.Count -ne 1 -or $children[0] -ine $diskId) { throw 'Reader topology differs.' }
Add-Type -TypeDefinition @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class SoundjieAutoSdRemove20261102 {
 [DllImport("cfgmgr32.dll", CharSet=CharSet.Unicode, ExactSpelling=true)] public static extern uint CM_Locate_DevNodeW(out uint node, string id, uint flags);
 [DllImport("cfgmgr32.dll", CharSet=CharSet.Unicode, ExactSpelling=true)] public static extern uint CM_Request_Device_EjectW(uint node, out uint veto, StringBuilder name, uint length, uint flags);
}
'@
[uint32]$node=0
if ([SoundjieAutoSdRemove20261102]::CM_Locate_DevNodeW([ref]$node,$usbId,0) -ne 0) { throw 'Reader no longer present.' }
[uint32]$veto=0
$vetoName=New-Object System.Text.StringBuilder 1024
$result=[SoundjieAutoSdRemove20261102]::CM_Request_Device_EjectW($node,[ref]$veto,$vetoName,1024,0)
$deadline=(Get-Date).AddSeconds(15)
if ($result -eq 0) {
    while ((Test-Path -LiteralPath $sdRoot) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 250 }
}
$remaining=@(Get-Disk | Where-Object { $_.SerialNumber.Trim() -eq '121220160204' })
$present=Test-Path -LiteralPath $sdRoot
$report=[ordered]@{
    Time=(Get-Date -Format o); Method='CM_Request_Device_EjectW'; Flags=0;
    Reader=$usbId; Serial='121220160204'; ObservedDiskNumber=$card.Number;
    Result=$result; VetoType=$veto; VetoName=$vetoName.ToString();
    VolumePresent=$present; DiskPresent=($remaining.Count -gt 0);
    SafeToUnplug=($result -eq 0 -and -not $present -and $remaining.Count -eq 0);
    FPresent=(Test-Path -LiteralPath 'F:\'); ForcedDismount=$false
}
$bytes=[Text.Encoding]::UTF8.GetBytes(($report | ConvertTo-Json -Depth 4)+"`n")
$stream=[IO.File]::Open($output,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try { $stream.Write($bytes,0,$bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
$report | ConvertTo-Json -Depth 4
if (-not $report.SafeToUnplug) { throw 'Normal removal not confirmed; do not unplug.' }
