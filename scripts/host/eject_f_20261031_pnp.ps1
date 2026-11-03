# One-session fallback after Shell Eject did not remove this verified USB disk.
# Uses normal Windows PnP safe-removal (veto respected), never force dismount.
param([string]$ClearScanSha256)
$ErrorActionPreference = 'Stop'
$targetRoot = 'F:\'
$diskId = 'SCSI\DISK&VEN_GENERIC&PROD_EXTERNAL\7&F55AA2C&0&000000'
$usbId = 'USB\VID_13FD&PID_5912\MSFT300004P70D0000E319621X'
$evidence = 'E:\Soundjie_Work\d4\evidence\soundjie-d4'
$handleExe = 'C:\Users\32450\AppData\Local\Temp\CodexSafeEjectHandle\handle64.exe'
if ($targetRoot.StartsWith($env:SystemDrive, [StringComparison]::OrdinalIgnoreCase)) { throw 'System drive refused.' }
if (-not (Test-Path -LiteralPath $targetRoot)) { throw 'Target already absent; no removal witnessed.' }
$volumeInfo = & fsutil.exe fsinfo volumeinfo F: 2>&1 | Out-String
if ($LASTEXITCODE -ne 0 -or $volumeInfo -notmatch '0x24c72563') { throw 'Volume serial changed.' }
if ([System.IO.DriveInfo]::new($targetRoot).TotalSize -ne 2000382066688) { throw 'Volume size changed.' }
if (Get-Process -Name vmware-vmx -ErrorAction SilentlyContinue) { throw 'A VM is still running.' }
$signature = Get-AuthenticodeSignature -LiteralPath $handleExe
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') { throw 'Handle signature invalid.' }
Add-Type -TypeDefinition @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class SoundjieSafeRemove {
 [DllImport("cfgmgr32.dll", CharSet=CharSet.Unicode, ExactSpelling=true)] public static extern uint CM_Locate_DevNodeW(out uint node, string id, uint flags);
 [DllImport("cfgmgr32.dll", ExactSpelling=true)] public static extern uint CM_Get_Parent(out uint parent, uint node, uint flags);
 [DllImport("cfgmgr32.dll", ExactSpelling=true)] public static extern uint CM_Get_Child(out uint child, uint node, uint flags);
 [DllImport("cfgmgr32.dll", ExactSpelling=true)] public static extern uint CM_Get_Sibling(out uint sibling, uint node, uint flags);
 [DllImport("cfgmgr32.dll", CharSet=CharSet.Unicode, ExactSpelling=true)] public static extern uint CM_Get_Device_IDW(uint node, StringBuilder id, uint length, uint flags);
 [DllImport("cfgmgr32.dll", CharSet=CharSet.Unicode, ExactSpelling=true)] public static extern uint CM_Request_Device_EjectW(uint node, out uint veto, StringBuilder name, uint length, uint flags);
}
'@
function Get-NodeId([uint32]$Node) {
  $buffer = New-Object System.Text.StringBuilder 1024
  if ([SoundjieSafeRemove]::CM_Get_Device_IDW($Node,$buffer,1024,0) -ne 0) { throw 'Device identity unavailable.' }
  return $buffer.ToString()
}
[uint32]$diskNode=0
if ([SoundjieSafeRemove]::CM_Locate_DevNodeW([ref]$diskNode,$diskId,0) -ne 0) { throw 'Exact disk not present.' }
[uint32]$usbNode=0
if ([SoundjieSafeRemove]::CM_Get_Parent([ref]$usbNode,$diskNode,0) -ne 0 -or (Get-NodeId $usbNode) -ine $usbId) { throw 'Unexpected parent; hub removal refused.' }
[uint32]$childNode=0
if ([SoundjieSafeRemove]::CM_Get_Child([ref]$childNode,$usbNode,0) -ne 0 -or (Get-NodeId $childNode) -ine $diskId) { throw 'USB child changed.' }
[uint32]$siblingNode=0
if ([SoundjieSafeRemove]::CM_Get_Sibling([ref]$siblingNode,$childNode,0) -eq 0) { throw 'USB parent contains other devices; refused.' }
# Storage enumeration was already completed: disk 1 contains only F.
# Run the required final handle scan after all such enumeration.
$scanPath=Join-Path $evidence 'safe-eject-F-20261031-final-handles.txt'
if ($ClearScanSha256) {
  # A scan just completed in this same operation. Its literal output and SHA
  # are verified; do not repeat a long scan solely because Handle returned 1
  # for no matches. Windows PnP still performs its own live veto checks.
  $age=((Get-Date).ToUniversalTime()-(Get-Item -LiteralPath $scanPath).LastWriteTimeUtc).TotalSeconds
  if ($age -lt 0 -or $age -gt 120 -or (Get-FileHash -LiteralPath $scanPath -Algorithm SHA256).Hash -ine $ClearScanSha256) { throw 'Clear scan is stale or changed.' }
  $handles=Get-Content -Raw -LiteralPath $scanPath
  $scanExit=1
} else {
  Write-Output 'Exact USB disk verified. Final read-only handle scan started.'
  $handles = & $handleExe -accepteula -nobanner $targetRoot 2>&1 | Out-String
  $scanExit = $LASTEXITCODE
  $handles | Set-Content -LiteralPath $scanPath -Encoding UTF8
}
if ($scanExit -notin @(0,1) -or $handles.Trim() -cne 'No matching handles found.') { throw "Handle scan not clear: $handles" }
if ((Get-NodeId $usbNode) -ine $usbId -or (Get-NodeId $diskNode) -ine $diskId) { throw 'Device changed during scan.' }
[uint32]$vetoType=0
$vetoName=New-Object System.Text.StringBuilder 1024
$result=[SoundjieSafeRemove]::CM_Request_Device_EjectW($usbNode,[ref]$vetoType,$vetoName,1024,0)
$deadline=(Get-Date).AddSeconds(20)
if ($result -eq 0) {
  while ((Test-Path -LiteralPath $targetRoot) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 250 }
}
$present=Test-Path -LiteralPath $targetRoot
$report=[ordered]@{
  Time=(Get-Date -Format o); Method='CM_Request_Device_EjectW'; Flags=0;
  ExpectedVolumeSerial='0x24c72563'; ExpectedDiskNumber=1; RelatedDriveLetters=@('F');
  DeviceInstance=$usbId; ReturnCode=$result; VetoType=$vetoType; VetoName=$vetoName.ToString();
  ClearHandleScanSha256=(Get-FileHash -LiteralPath $scanPath -Algorithm SHA256).Hash;
  FPresent=$present; SafeToUnplug=($result -eq 0 -and -not $present);
  OriginalUserFilesDeleted=0; ForcedDismount=$false
}
$report | ConvertTo-Json -Depth 4 | Tee-Object -FilePath (Join-Path $evidence 'safe-eject-F-20261031-pnp.json')
if (-not $report.SafeToUnplug) { throw 'Safe removal not confirmed; do not unplug.' }
