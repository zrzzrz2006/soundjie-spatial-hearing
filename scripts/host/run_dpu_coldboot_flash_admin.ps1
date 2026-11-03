$ErrorActionPreference = 'Stop'

$flasher = Join-Path $PSScriptRoot 'flash_mz7100_sd_image_admin.ps1'
$image = 'F:\MLK\DPU_ZYNQ7000\artifacts\mz7100_debian9_restored_sd\mz7100_debian9_originalboot_regpatch_dpuy1v2_15gb.img'
$hash = '8fa1fb0e9cf6114c581f037b64e850a8c200a3d50a24781e8b39a4f7d1b50522'

& $flasher -ImagePath $image -ExpectedImageHash $hash -NoPause
