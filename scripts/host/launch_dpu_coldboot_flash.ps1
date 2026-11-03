$ErrorActionPreference = 'Stop'

$adminScript = Join-Path $PSScriptRoot 'run_dpu_coldboot_flash_admin.ps1'
$arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $adminScript)

$process = Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList $arguments -Wait -PassThru
exit $process.ExitCode
