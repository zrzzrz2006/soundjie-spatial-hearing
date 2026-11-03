# Exact authorized reader/card, normal Windows PnP removal; no forced fallback.
$ErrorActionPreference='Stop'
$root='\\?\Volume{9b6c8f5a-bf04-11f1-9fa3-005056c00008}\'
$diskId='USBSTOR\DISK&VEN_MASS&PROD_STORAGE_DEVICE&REV_1.00\121220160204&0'
$usbId='USB\VID_14CD&PID_1212\121220160204'
$output='F:\MLK\DPU_ZYNQ7000\artifacts\soundjie_two_clips\20261101a\sd-persistence-20261101\safe-eject.json'
if (Test-Path -LiteralPath $output) { throw 'Evidence exists; inspect before repeating.' }
$stage=Get-Content -LiteralPath 'F:\MLK\DPU_ZYNQ7000\artifacts\soundjie_two_clips\20261101a\sd-persistence-20261101\post-stage-verification.json' -Raw -Encoding UTF8 | ConvertFrom-Json
if ($stage.Result -ne 'SD_APPLICATION_STAGE_READBACK_OK' -or -not $stage.BootFilesUnchanged -or $stage.UnchangedOriginalFiles -ne 44) { throw 'Staging is not verified.' }
$disk=Get-Disk -Number 2
if ($disk.SerialNumber -ne '121220160204' -or $disk.Size -ne 15728640000 -or $disk.BusType -ne 'USB' -or $disk.IsBoot -or $disk.IsSystem) { throw 'SD identity changed.' }
$partition=Get-Partition -DiskNumber 2 -PartitionNumber 1
if ($partition.Offset -ne 1048576 -or $partition.Size -ne 103809024 -or $root -notin $partition.AccessPaths) { throw 'SD partition changed.' }
$volume=Get-Volume -UniqueId $root
if ($volume.FileSystem -ne 'FAT32' -or $volume.FileSystemLabel -ne 'boot' -or $volume.HealthStatus -ne 'Healthy') { throw 'Unexpected volume.' }
$hashes=@{
 'SJ6V1.TGZ'='0e89305ba1822fe5b8f7a41fb0b6919b09b1212eca0e458b9e51046bb762d7af'
 'SJ6V1.SHA'='dd08825e30595b48b6d1322832e5b72bbc5b5f6f6a50cb11211d30ab48574e26'
 'SJ4R1.TGZ'='28a34d2c936adb708e5541e4eb21ab6c76491445911bc98cff2e43b9c1c1f640'
}
foreach ($name in $hashes.Keys) {
 if ((Get-FileHash -LiteralPath ($root+$name) -Algorithm SHA256).Hash -ine $hashes[$name]) { throw "Readback changed: $name" }
}
$parent=(Get-PnpDeviceProperty -InstanceId $diskId -KeyName DEVPKEY_Device_Parent).Data
$children=@((Get-PnpDeviceProperty -InstanceId $usbId -KeyName DEVPKEY_Device_Children).Data)
if ($parent -ine $usbId -or $children.Count -ne 1 -or $children[0] -ine $diskId) { throw 'Unexpected reader topology.' }
Add-Type -TypeDefinition @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class SoundjieSdRemove20261101 {
 [DllImport("cfgmgr32.dll", CharSet=CharSet.Unicode, ExactSpelling=true)] public static extern uint CM_Locate_DevNodeW(out uint node, string id, uint flags);
 [DllImport("cfgmgr32.dll", CharSet=CharSet.Unicode, ExactSpelling=true)] public static extern uint CM_Request_Device_EjectW(uint node, out uint veto, StringBuilder name, uint length, uint flags);
}
'@
[uint32]$node=0
if ([SoundjieSdRemove20261101]::CM_Locate_DevNodeW([ref]$node,$usbId,0) -ne 0) { throw 'Reader no longer present.' }
[uint32]$veto=0
$vetoName=New-Object System.Text.StringBuilder 1024
$result=[SoundjieSdRemove20261101]::CM_Request_Device_EjectW($node,[ref]$veto,$vetoName,1024,0)
$deadline=(Get-Date).AddSeconds(15)
if ($result -eq 0) {
 while ((Test-Path -LiteralPath $root) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 250 }
}
$present=Test-Path -LiteralPath $root
$remaining=@(Get-Disk | Where-Object { $_.SerialNumber -eq '121220160204' })
$report=[ordered]@{
 Time=(Get-Date -Format o); Method='CM_Request_Device_EjectW'; Flags=0;
 Reader=$usbId; Disk=$diskId; ExpectedDiskSerial='121220160204';
 Result=$result; VetoType=$veto; VetoName=$vetoName.ToString();
 VolumePresent=$present; DiskPresent=($remaining.Count -gt 0);
 SafeToUnplug=($result -eq 0 -and -not $present -and $remaining.Count -eq 0);
 FPresent=(Test-Path -LiteralPath 'F:\'); ForcedDismount=$false
}
$bytes=[Text.Encoding]::UTF8.GetBytes(($report | ConvertTo-Json -Depth 4)+"`n")
$stream=[IO.File]::Open($output,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try { $stream.Write($bytes,0,$bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
$report | ConvertTo-Json -Depth 4
if (-not $report.SafeToUnplug) { throw 'Normal safe removal not confirmed; do not unplug.' }
