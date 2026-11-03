param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path,
    [string]$ToolchainRoot = 'F:\MLK\DPU_ZYNQ7000\tools\gcc-linaro-6.3.1-2017.02-win32-armhf-fresh',
    [string]$DnndkRoot = 'F:\MLK\DPU_ZYNQ7000\references\Zynq-7000-DPU-TRD_guest\reference-files\sdcard\zynq7020_dnndk_v3.0\pkgs',
    [Parameter(Mandatory=$true)][string]$ModelElf,
    [string]$Output = 'F:\MLK\DPU_ZYNQ7000\artifacts\soundjie_d0\runtime\sound_shape_probe'
)

$ErrorActionPreference = 'Stop'
$cxx = Join-Path $ToolchainRoot 'bin\arm-linux-gnueabihf-g++.exe'
$readelf = Join-Path $ToolchainRoot 'bin\arm-linux-gnueabihf-readelf.exe'
$required = @(
    $cxx, $readelf, (Join-Path $DnndkRoot 'include\n2cube.h'),
    (Join-Path $DnndkRoot 'lib\libn2cube.so'), (Join-Path $DnndkRoot 'lib\libhineon.so'),
    $ModelElf, (Join-Path $RepoRoot 'src\sound_shape_runner.cpp'),
    (Join-Path $RepoRoot 'src\tensor_stats.cpp'),
    (Join-Path $RepoRoot 'src\sha256.cpp'), (Join-Path $RepoRoot 'src\sha256.h')
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing build input: $path" }
}

New-Item -ItemType Directory -Path (Split-Path -Parent $Output) -Force | Out-Null
if (Test-Path -LiteralPath $Output) { throw "Output already exists: $Output" }
$temporary = "$Output.new.$PID"
if (Test-Path -LiteralPath $temporary) { throw "Temporary output already exists: $temporary" }
$modelSnapshotDirectory = "$Output.model.$([Guid]::NewGuid().ToString('N'))"
$modelSnapshot = Join-Path $modelSnapshotDirectory 'model.elf'
try {
    [System.IO.Directory]::CreateDirectory($modelSnapshotDirectory) | Out-Null
    [System.IO.File]::Copy($ModelElf, $modelSnapshot, $false)
    $modelSha = (Get-FileHash -LiteralPath $modelSnapshot -Algorithm SHA256).Hash.ToLowerInvariant()
    $modelDefine = '-DSOUNDJIE_LINKED_MODEL_SHA256="{0}"' -f $modelSha
    & $cxx '-std=c++11' '-O2' '-Wall' '-Wextra' '-Werror' `
        '-march=armv7-a' '-mfloat-abi=hard' '-mfpu=vfpv3-d16' '-static-libgcc' '-static-libstdc++' `
        $modelDefine ('-I' + (Join-Path $DnndkRoot 'include')) ('-I' + (Join-Path $RepoRoot 'src')) `
        (Join-Path $RepoRoot 'src\sound_shape_runner.cpp') (Join-Path $RepoRoot 'src\tensor_stats.cpp') `
        (Join-Path $RepoRoot 'src\sha256.cpp') `
        $modelSnapshot ('-L' + (Join-Path $DnndkRoot 'lib')) '-Wl,-rpath,/usr/lib' `
        '-ln2cube' '-lhineon' '-lpthread' '-ldl' '-o' $temporary
    if ($LASTEXITCODE -ne 0) { throw "Cross-build failed: $LASTEXITCODE" }

    $header = (& $readelf -h $temporary | Out-String)
    $attributes = (& $readelf -A $temporary | Out-String)
    $versions = (& $readelf --version-info $temporary | Out-String)
    if ($header -notmatch 'Machine:\s+ARM' -or $header -notmatch 'Version5 EABI' -or
        $attributes -notmatch 'Tag_CPU_arch: v7' -or
        $attributes -notmatch 'Tag_ABI_VFP_args: VFP registers') { throw 'Not ARMv7 EABI5 hard-float' }
    $glibc = [regex]::Matches($versions, 'GLIBC_(\d+)\.(\d+)') | ForEach-Object {
        [version]("{0}.{1}" -f $_.Groups[1].Value, $_.Groups[2].Value)
    } | Sort-Object -Unique | Select-Object -Last 1
    if ($null -eq $glibc -or $glibc -gt [version]'2.17') { throw "Unsupported glibc: $glibc" }
    $modelShaAfter = (Get-FileHash -LiteralPath $modelSnapshot -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($modelShaAfter -ne $modelSha) { throw 'Model snapshot changed during build' }
    [System.IO.File]::Move($temporary, $Output)
    "OUTPUT=$Output"
    "MAX_GLIBC=$glibc"
    "SHA256=$((Get-FileHash -LiteralPath $Output -Algorithm SHA256).Hash)"
}
finally {
    if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
    if (Test-Path -LiteralPath $modelSnapshotDirectory) {
        Remove-Item -LiteralPath $modelSnapshotDirectory -Recurse -Force
    }
}
