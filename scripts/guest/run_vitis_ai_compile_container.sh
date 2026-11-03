#!/usr/bin/env bash
set -u

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <absolute-repo-directory> <absolute-run-directory>" >&2
  exit 64
fi

repo_dir=$1
run_dir=$2
export PATH=/mnt/workspace/docker-20.10.24:${PATH}
export DOCKER_HOST=unix:///mnt/workspace/docker-run/docker.sock

set +e
docker run --rm \
  -v "${repo_dir}:${repo_dir}:ro" \
  -v "${run_dir}:${run_dir}" \
  xilinx/vitis-ai-cpu:1.1.56 \
  bash "${run_dir}/run_vitis_ai_compile_inside.sh" "${repo_dir}" "${run_dir}" \
  > "${run_dir}/container-compile.log" 2>&1
status=$?
set -e

report="${run_dir}/sound-shape-compile-report.txt"
if [[ ! -s "$report" ]] || ! grep -q 'RESULT=pass' "${run_dir}/container-compile.log"; then
  status=70
fi

printf '%s\n' "${status}" > "${run_dir}/container-compile.exit_code"
exit 0
