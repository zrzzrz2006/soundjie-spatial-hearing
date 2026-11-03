#!/usr/bin/env bash
set -u

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <hardware.hwh>" >&2
  exit 64
fi

source /etc/profile.d/conda.sh
conda activate vitis-ai-caffe

input_hwh=$1
set +e
dlet -f "${input_hwh}" > dlet.log 2>&1
status=$?
set -e

printf '%s\n' "${status}" > dlet.exit_code
find . -maxdepth 2 -type f -printf '%P\n' | sort > dlet.outputs.txt
exit "${status}"
