param(
    [ValidateSet('ReadOnlyVmdk', 'Vmware', 'Auto')]
    [string]$AuditMode = 'ReadOnlyVmdk',
    [string]$Vmrun = 'F:\MLK\VMware\Workstation26H1u1\vmrun.exe',
    [string]$Vmx = 'F:\MLK\VMware\VMs\osrc_debian\osrc_Debian.codex.vmx',
    [string]$Vmdk = 'F:\MLK\VMware\VMs\osrc_debian\osrc_Debian.vmdk',
    [string]$GuestUser = 'root',
    [string]$GuestPassword = 'root'
)

$ErrorActionPreference = 'Stop'
$hostReport = 'F:\MLK\DPU_ZYNQ7000\usb_mic_bringup_v1\release\build-report.txt'

function Invoke-VmwareAudit {
    if (-not (Test-Path -LiteralPath $Vmrun)) { throw "vmrun not found: $Vmrun" }
    if (-not (Test-Path -LiteralPath $Vmx)) { throw "VMX not found: $Vmx" }

    $projectGuest = '/mnt/hgfs/DPU_ZYNQ7000/usb_mic_bringup_v1'
    $auditScript = "$projectGuest/scripts/guest/audit_usb_audio_support.sh"
    $report = "$projectGuest/release/build-report.txt"
    $kernelSource = '/mnt/workspace/mz7100_linux_extract/osrc-lab-7100/sources/kernel'
    $rootfs = '/mnt/workspace/mz7100_linux_extract/osrc-lab-7100/boards/MZ7X/buildroot'
    $dtb = '/mnt/hgfs/DPU_ZYNQ7000/artifacts/mz7100_dpu_2017_probe_v1/sd_boot_ready/devicetree.dtb'

    $runningBefore = (& $Vmrun -T ws list) -contains $Vmx
    if (-not $runningBefore) {
        & $Vmrun -T ws start $Vmx nogui | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'Failed to start VMware guest.' }
    }

    try {
        & $Vmrun -T ws -gu $GuestUser -gp $GuestPassword runProgramInGuest $Vmx `
            /bin/bash $auditScript `
            --expected-kernel 4.14.0-xilinx `
            --kernel-source $kernelSource `
            --rootfs $rootfs `
            --dtb $dtb `
            --output $report
        if ($LASTEXITCODE -ne 0) { throw "Guest audit failed with exit code $LASTEXITCODE" }
        Add-Content -LiteralPath $hostReport -Value 'AUDIT_TRANSPORT=vmware-guest'
    }
    finally {
        if (-not $runningBefore) {
            & $Vmrun -T ws stop $Vmx soft | Out-Null
        }
    }
}

function Invoke-ReadOnlyVmdkAudit {
    if (-not (Test-Path -LiteralPath $Vmdk)) { throw "VMDK not found: $Vmdk" }

    $bash = @'
set -eu
vmdk=/mnt/f/MLK/VMware/VMs/osrc_debian/osrc_Debian.vmdk
mount_dir=/mnt/vmdk_ro
attached_here=0
mounted_here=0

command -v qemu-nbd >/dev/null 2>&1 || {
  echo "qemu-nbd is required for the read-only VMDK audit" >&2
  exit 69
}
modprobe nbd max_part=16

if [ ! -s /sys/block/nbd0/pid ]; then
  qemu-nbd --read-only --format=vmdk --connect=/dev/nbd0 "$vmdk"
  attached_here=1
fi

mkdir -p "$mount_dir"
if ! mountpoint -q "$mount_dir"; then
  mount -o ro,noload /dev/nbd0p1 "$mount_dir"
  mounted_here=1
fi

cleanup() {
  if [ "$mounted_here" -eq 1 ]; then umount "$mount_dir"; fi
  if [ "$attached_here" -eq 1 ]; then qemu-nbd --disconnect /dev/nbd0 >/dev/null; fi
}
trap cleanup EXIT

project=/mnt/f/MLK/DPU_ZYNQ7000/usb_mic_bringup_v1
bash "$project/scripts/guest/audit_usb_audio_support.sh" \
  --expected-kernel 4.14.0-xilinx \
  --kernel-source "$mount_dir/mnt/workspace/mz7100_linux_extract/osrc-lab-7100/sources/kernel" \
  --rootfs "$mount_dir/mnt/workspace/mz7100_linux_extract/osrc-lab-7100/boards/MZ7X/buildroot" \
  --dtb /mnt/f/MLK/DPU_ZYNQ7000/artifacts/mz7100_dpu_2017_probe_v1/sd_boot_ready/devicetree.dtb \
  --output "$project/release/build-report.txt"
printf '%s\n' 'AUDIT_TRANSPORT=read-only-vmdk' >> "$project/release/build-report.txt"
'@

    & wsl.exe -u root -e bash -lc $bash
    if ($LASTEXITCODE -ne 0) { throw "Read-only VMDK audit failed with exit code $LASTEXITCODE" }
}

switch ($AuditMode) {
    'Vmware' { Invoke-VmwareAudit }
    'ReadOnlyVmdk' { Invoke-ReadOnlyVmdkAudit }
    'Auto' {
        try { Invoke-VmwareAudit }
        catch {
            Write-Warning "VMware audit failed; using read-only VMDK fallback: $($_.Exception.Message)"
            Invoke-ReadOnlyVmdkAudit
        }
    }
}

if (-not (Test-Path -LiteralPath $hostReport)) { throw "Audit report missing: $hostReport" }
Get-Content -LiteralPath $hostReport
