#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/conda.sh
conda activate vitis-ai-caffe

echo "CONDA_PREFIX=${CONDA_PREFIX}"
find "${CONDA_PREFIX}" /opt/vitis_ai -type f \
  \( -iname '*.dcf' -o -iname '*.json' -o -iname '*.prototxt' \) \
  2>/dev/null \
  | grep -Ei '/(arch|target|dpu)|dcf' \
  | sort
