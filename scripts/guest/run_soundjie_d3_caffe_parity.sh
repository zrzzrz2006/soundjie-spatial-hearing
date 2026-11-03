#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <absolute-input-directory> <absolute-output-directory>" >&2
  exit 64
fi

input_dir=$1
output_dir=$2

source /etc/profile.d/conda.sh
conda activate vitis-ai-caffe

python "${input_dir}/export_soundjie_d3_caffe.py" caffe \
  --weights "${input_dir}/fused-weights.npz" \
  --deploy "${input_dir}/deploy.prototxt" \
  --inputs "${input_dir}/parity-inputs.npy" \
  --reference-logits "${input_dir}/pytorch-logits.npy" \
  --output-dir "${output_dir}"
