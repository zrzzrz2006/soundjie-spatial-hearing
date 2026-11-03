#!/usr/bin/env bash
set -u

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <absolute-work-directory>" >&2
  exit 64
fi

work_dir=$1
export PATH=/mnt/workspace/docker-20.10.24:${PATH}
export DOCKER_HOST=unix:///mnt/workspace/docker-run/docker.sock

set +e
docker run --rm \
  --user root \
  -v "${work_dir}:/work" \
  -w /work \
  xilinx/vitis-ai-cpu:1.1.56 \
  bash ./probe_dlet_hwh.sh design_1.hwh \
  > "${work_dir}/container.log" 2>&1
status=$?
set -e

printf '%s\n' "${status}" > "${work_dir}/container.exit_code"
exit 0
