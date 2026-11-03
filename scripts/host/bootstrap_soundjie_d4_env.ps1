[CmdletBinding()]
param(
    [string]$VenvPath = 'F:\MLK\DPU_ZYNQ7000\envs\soundjie_d4_host',
    [string]$BasePython = 'D:\Python3.8\python\python.exe',
    [string]$RepoRoot = '',
    [switch]$ValidateOnly,
    [switch]$RunBaseline
)

$ErrorActionPreference = 'Stop'

if (-not $RepoRoot) {
    $RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
} else {
    $RepoRoot = [System.IO.Path]::GetFullPath($RepoRoot)
}
$fullVenv = [System.IO.Path]::GetFullPath($VenvPath)
$venvRoot = [System.IO.Path]::GetPathRoot($fullVenv)
if (-not [string]::Equals($venvRoot, 'F:\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "D4 environment must be placed on the F drive: $fullVenv"
}
if (-not (Test-Path -LiteralPath $BasePython -PathType Leaf)) {
    throw "Pinned CPython is missing: $BasePython"
}

$requirements = Join-Path $RepoRoot 'config\soundjie_d4_host_requirements.txt'
if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
    throw "Requirements file is missing: $requirements"
}

$abiProbe = @'
import json
import sys
import sysconfig
print(json.dumps({
    'executable': sys.executable,
    'version': sys.version,
    'Py_GIL_DISABLED': int(sysconfig.get_config_var('Py_GIL_DISABLED') or 0),
}))
'@
$abiJson = & $BasePython -c $abiProbe
if ($LASTEXITCODE -ne 0) {
    throw 'Pinned CPython ABI probe failed'
}
$abi = $abiJson | ConvertFrom-Json
if ($abi.Py_GIL_DISABLED -ne 0) {
    throw "Free-threaded CPython is not supported: $($abi.executable)"
}

Write-Output "D4_ENV_PATH=$fullVenv"
Write-Output "BASE_PYTHON=$($abi.executable)"
Write-Output "PYTHON_VERSION=$($abi.version)"
Write-Output "PY_GIL_DISABLED=$($abi.Py_GIL_DISABLED)"
if ($ValidateOnly) {
    Write-Output 'D4_ENV_VALIDATION=pass'
    exit 0
}

$venvPython = Join-Path $fullVenv 'Scripts\python.exe'
if (Test-Path -LiteralPath $fullVenv) {
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        throw "Existing destination is not a usable venv; it will not be deleted: $fullVenv"
    }
} else {
    $parent = Split-Path -Parent $fullVenv
    if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
    & $BasePython -m venv $fullVenv
    if ($LASTEXITCODE -ne 0) {
        throw 'Creating the D4 venv failed'
    }
}

$pipCache = 'F:\MLK\DPU_ZYNQ7000\cache\pip-soundjie-d4'
if (-not (Test-Path -LiteralPath $pipCache -PathType Container)) {
    New-Item -ItemType Directory -Path $pipCache -Force | Out-Null
}
$env:PIP_CACHE_DIR = $pipCache
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
& $venvPython -m pip install --requirement $requirements
if ($LASTEXITCODE -ne 0) {
    throw 'Installing the pinned D4 dependencies failed'
}

$dependencyProbe = @'
import json
import numpy
import scipy
import serial
import soundfile
import torch
print(json.dumps({
    'numpy': numpy.__version__,
    'scipy': scipy.__version__,
    'pyserial': serial.VERSION,
    'soundfile': soundfile.__version__,
    'torch': torch.__version__,
}, sort_keys=True))
'@
$versions = & $venvPython -c $dependencyProbe
if ($LASTEXITCODE -ne 0) {
    throw 'D4 dependency import probe failed'
}
Write-Output "PIP_CACHE=$pipCache"
Write-Output "DEPENDENCIES=$versions"

if ($RunBaseline) {
    $safeDirectory = $RepoRoot.Replace('\', '/')
    $env:GIT_CONFIG_COUNT = '1'
    $env:GIT_CONFIG_KEY_0 = 'safe.directory'
    $env:GIT_CONFIG_VALUE_0 = $safeDirectory
    Push-Location $RepoRoot
    try {
        & $venvPython -m unittest discover -s tests -p 'test_*.py'
        if ($LASTEXITCODE -ne 0) {
            throw 'D4 baseline tests failed'
        }
    } finally {
        Pop-Location
    }
}

Write-Output 'D4_ENV_STATUS=pass'
