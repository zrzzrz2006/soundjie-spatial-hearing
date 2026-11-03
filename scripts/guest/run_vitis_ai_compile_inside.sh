#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <absolute-repo-directory> <absolute-run-directory>" >&2
  exit 64
fi

repo_dir=$1
run_dir=$2
source /etc/profile.d/conda.sh
conda activate vitis-ai-caffe

bash "${repo_dir}/scripts/guest/build_sound_shape_vitis_ai.sh" \
  --model-root "${repo_dir}/models/sound_shape_probe" \
  --hwh "${run_dir}/mz7100-system.hwh" \
  --platform-dcf "${run_dir}/b1152-target-1.4.0-candidate.dcf" \
  --output-dir "${run_dir}/output" \
  --report "${run_dir}/sound-shape-compile-report.txt"
