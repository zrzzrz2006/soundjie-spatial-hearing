[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$OutputManifest
)

$ErrorActionPreference = 'Stop'
$sourceRoot = [System.IO.Path]::GetFullPath($Root)
$outputPath = [System.IO.Path]::GetFullPath($OutputManifest)
if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
    throw "Source root does not exist: $sourceRoot"
}
$sourcePrefix = $sourceRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
if ($outputPath.StartsWith($sourcePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputManifest must be outside the vendor source root'
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
function Get-Sha256Lower([string]$Path) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        return ([System.BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    } finally {
        $stream.Dispose()
        $sha.Dispose()
    }
}
$extensions = @('.xpr', '.bd', '.hdf', '.xsa', '.zip', '.rar', '.7z')
$candidates = @()
Get-ChildItem -LiteralPath $sourceRoot -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $extensions -contains $_.Extension.ToLowerInvariant() } |
    Sort-Object FullName |
    ForEach-Object {
        $members = @()
        $memberStatus = 'not_applicable'
        if ($_.Extension -ieq '.zip') {
            try {
                $zip = [System.IO.Compression.ZipFile]::OpenRead($_.FullName)
                try {
                    $members = @($zip.Entries | ForEach-Object { $_.FullName })
                    $memberStatus = 'listed'
                } finally {
                    $zip.Dispose()
                }
            } catch {
                $memberStatus = 'unreadable'
            }
        } elseif ($_.Extension -in @('.rar', '.7z')) {
            $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
            if ($tar) {
                $listed = @(& $tar.Source -tf $_.FullName 2>$null)
                if ($LASTEXITCODE -eq 0) {
                    $members = $listed
                    $memberStatus = 'listed_with_tar'
                } else {
                    $memberStatus = 'unreadable_with_tar'
                }
            } else {
                $memberStatus = 'external_lister_required'
            }
        }
        $relative = $_.FullName.Substring($sourceRoot.Length).TrimStart('\', '/')
        $hintText = ($relative + ' ' + ($members -join ' ')).ToLowerInvariant()
        $score = 0
        foreach ($term in @('hdmi', 'vdma', 'video', 'lcd', 'display', 'framebuffer')) {
            if ($hintText.Contains($term)) { $score++ }
        }
        $candidates += [ordered]@{
            path = $_.FullName
            relative_path = $relative
            extension = $_.Extension.ToLowerInvariant()
            size = $_.Length
            sha256 = Get-Sha256Lower $_.FullName
            display_hint_score = $score
            member_status = $memberStatus
            members = @($members)
        }
    }

$manifest = [ordered]@{
    schema = 1
    source_root = $sourceRoot
    read_only = $true
    candidate_count = $candidates.Count
    candidates = @($candidates)
    result = 'pass'
}
$parent = Split-Path -Parent $outputPath
if ($parent -and -not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $outputPath -Encoding UTF8
Write-Output "INDEXED=$($candidates.Count)"
Write-Output "RESULT=pass"
