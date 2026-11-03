param(
    [string]$ConfigPath = "",
    [string]$WorkDir = "",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $repoRoot "config\soundjie_d4_pl_sources.json"
}
$configFull = (Resolve-Path -LiteralPath $ConfigPath).Path
$config = Get-Content -LiteralPath $configFull -Raw | ConvertFrom-Json

if ([string]::IsNullOrWhiteSpace($WorkDir)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $WorkDir = "$($config.staging.work_dir_prefix)$stamp"
}

$candidate = [System.IO.Path]::GetFullPath($WorkDir)
$prefix = [System.IO.Path]::GetFullPath([string]$config.staging.work_dir_prefix)
$vendorRoot = [System.IO.Path]::GetFullPath([string]$config.staging.vendor_source_root)
if ([System.IO.Path]::GetPathRoot($candidate) -ne 'E:\') {
    throw "REFUSE: PL rebuild staging must stay on the approved E drive"
}
if (-not $candidate.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "REFUSE: work directory must use the approved create-only prefix $prefix"
}
if ($candidate.StartsWith($vendorRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "REFUSE: vendor source tree is read-only"
}
if (Test-Path -LiteralPath $candidate) {
    throw "REFUSE: work directory already exists: $candidate"
}

foreach ($name in @("ch24_archive", "dpu_hdf")) {
    $entry = $config.sources.$name
    if (-not (Test-Path -LiteralPath $entry.path -PathType Leaf)) {
        throw "REFUSE: missing source $name at $($entry.path)"
    }
    $actual = (Get-FileHash -LiteralPath $entry.path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne ([string]$entry.sha256).ToLowerInvariant()) {
        throw "REFUSE: SHA-256 mismatch for $name"
    }
}
if (-not (Test-Path -LiteralPath $config.sources.dpu_ip_repo.path -PathType Container)) {
    throw "REFUSE: missing DPU IP repository"
}
if ($ValidateOnly) {
    Write-Output "D4_PL_STAGE_VALIDATION=pass"
    Write-Output "WORK_DIR=$candidate"
    exit 0
}

$unrar = "C:\Program Files\WinRAR\UnRAR.exe"
$archiveFormat = [string]$config.sources.ch24_archive.format
if ([string]::IsNullOrWhiteSpace($archiveFormat)) { $archiveFormat = 'rar' }
if ($archiveFormat -notin @('zip', 'rar')) { throw 'Unsupported CH24 archive format' }
if ($archiveFormat -eq 'rar' -and -not (Test-Path -LiteralPath $unrar -PathType Leaf)) {
    throw "REFUSE: WinRAR UnRAR.exe is unavailable"
}

New-Item -ItemType Directory -Path $candidate | Out-Null
$sourcesDir = New-Item -ItemType Directory -Path (Join-Path $candidate "sources")
$ch24Dir = New-Item -ItemType Directory -Path (Join-Path $sourcesDir "ch24")
$dpuDir = New-Item -ItemType Directory -Path (Join-Path $sourcesDir "dpu")
$ipDir = Join-Path $dpuDir "ip_repo"

if ($archiveFormat -eq 'zip') {
    Expand-Archive -LiteralPath ([string]$config.sources.ch24_archive.path) -DestinationPath $ch24Dir.FullName
} else {
    & $unrar x -idq -o- ([string]$config.sources.ch24_archive.path) "$($ch24Dir.FullName)\"
    if ($LASTEXITCODE -ne 0) { throw "CH24 extraction failed with exit code $LASTEXITCODE" }
}

Copy-Item -LiteralPath $config.sources.dpu_hdf.path -Destination (Join-Path $dpuDir "mz7100_dpu_system_wrapper.hdf")
Copy-Item -LiteralPath $config.sources.dpu_ip_repo.path -Destination $ipDir -Recurse

$hdfExtract = New-Item -ItemType Directory -Path (Join-Path $dpuDir "hdf")
& tar -xf (Join-Path $dpuDir "mz7100_dpu_system_wrapper.hdf") -C $hdfExtract.FullName
$hdfTarExit = $LASTEXITCODE
$requiredHdfMembers = @("sysdef.xml", "system.hwh", "system_bd.tcl", "system_wrapper.bit")
$missingHdfMembers = @(
    $requiredHdfMembers | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $hdfExtract.FullName $_) -PathType Leaf)
    }
)
if ($missingHdfMembers.Count -ne 0) {
    throw "DPU HDF extraction is incomplete: $($missingHdfMembers -join ', ')"
}
if ($hdfTarExit -ne 0) {
    Write-Warning "tar returned $hdfTarExit after extracting all required HDF members; Windows timestamp warnings are accepted"
}
Write-Output "HDF_PAYLOAD_VALIDATED=pass"

$receipt = [ordered]@{
    schema = "soundjie-d4-pl-stage-v1"
    created_utc = [DateTime]::UtcNow.ToString("o")
    work_dir = $candidate
    config_sha256 = (Get-FileHash -LiteralPath $configFull -Algorithm SHA256).Hash.ToLowerInvariant()
    ch24_archive_sha256 = ([string]$config.sources.ch24_archive.sha256).ToLowerInvariant()
    dpu_hdf_sha256 = ([string]$config.sources.dpu_hdf.sha256).ToLowerInvariant()
    source_policy = "read-only; create-only destination"
}
$receipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $candidate "stage-receipt.json") -Encoding UTF8
Write-Output "D4_PL_STAGE=pass"
Write-Output "WORK_DIR=$candidate"
