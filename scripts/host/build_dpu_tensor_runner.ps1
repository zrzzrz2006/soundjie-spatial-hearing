param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$ToolchainRoot = 'F:\MLK\DPU_ZYNQ7000\tools\gcc-linaro-6.3.1-2017.02-win32-armhf-fresh',
    [string]$DnndkRoot = 'F:\MLK\DPU_ZYNQ7000\references\Zynq-7000-DPU-TRD_guest\reference-files\sdcard\zynq7020_dnndk_v3.0\pkgs',
    [string]$ModelElf = 'F:\MLK\DPU_ZYNQ7000\references\Zynq-7000-DPU-TRD_guest\reference-files\sdcard\yolo_prune\model\dpu_yolo.elf',
    [string]$Output = 'F:\MLK\DPU_ZYNQ7000\artifacts\dpu_real_inference_v1\level-a\dpu_yolo_probe'
)

$ErrorActionPreference = 'Stop'
$cxx = Join-Path $ToolchainRoot 'bin\arm-linux-gnueabihf-g++.exe'
$readelf = Join-Path $ToolchainRoot 'bin\arm-linux-gnueabihf-readelf.exe'
foreach ($required in @(
    $cxx,
    $readelf,
    (Join-Path $DnndkRoot 'include\n2cube.h'),
    (Join-Path $DnndkRoot 'lib\libn2cube.so'),
    (Join-Path $DnndkRoot 'lib\libhineon.so'),
    $ModelElf,
    (Join-Path $RepoRoot 'src\dpu_tensor_runner.cpp'),
    (Join-Path $RepoRoot 'src\tensor_stats.cpp')
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Missing required build input: $required"
    }
}

$outputDir = Split-Path -Parent $Output
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$temporary = "$Output.new"
$arguments = @(
    '-std=c++11', '-O2', '-Wall', '-Wextra',
    '-static-libgcc', '-static-libstdc++',
    ('-I' + (Join-Path $DnndkRoot 'include')),
    ('-I' + (Join-Path $RepoRoot 'src')),
    (Join-Path $RepoRoot 'src\dpu_tensor_runner.cpp'),
    (Join-Path $RepoRoot 'src\tensor_stats.cpp'),
    $ModelElf,
    ('-L' + (Join-Path $DnndkRoot 'lib')),
    '-Wl,-rpath,/usr/lib',
    '-ln2cube', '-lhineon', '-lpthread', '-ldl',
    '-o', $temporary
)
& $cxx @arguments
if ($LASTEXITCODE -ne 0) { throw "Cross-build failed with exit code $LASTEXITCODE" }

$header = (& $readelf -h $temporary | Out-String)
$attributes = (& $readelf -A $temporary | Out-String)
$versions = (& $readelf --version-info $temporary | Out-String)
if ($header -notmatch 'Machine:\s+ARM' -or $header -notmatch 'Version5 EABI' -or
    $attributes -notmatch 'Tag_ABI_VFP_args: VFP registers') {
    throw 'Output is not ARM EABI5 hard-float.'
}
$glibcVersions = [regex]::Matches($versions, 'GLIBC_(\d+)\.(\d+)') | ForEach-Object {
    [version]("{0}.{1}" -f $_.Groups[1].Value, $_.Groups[2].Value)
} | Sort-Object -Unique
$maxGlibc = $glibcVersions | Select-Object -Last 1
if ($null -eq $maxGlibc -or $maxGlibc -gt [version]'2.24') {
    throw "Output requires unsupported glibc version: $maxGlibc"
}

# The destination is a generated artifact owned by this project.  Replace it
# only after the new binary has passed all static compatibility gates.
Move-Item -LiteralPath $temporary -Destination $Output -Force
Write-Output "OUTPUT=$Output"
Write-Output "MAX_GLIBC=$maxGlibc"
Write-Output "SHA256=$((Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash)"
