#!/bin/sh
set -eu

root=
report=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --root) root=$2; shift 2 ;;
    --report) report=$2; shift 2 ;;
    *) echo "unknown_argument:$1" >&2; exit 64 ;;
  esac
done

case "$root:$report" in
  /*:/*) ;;
  *) echo "absolute_paths_required" >&2; exit 64 ;;
esac
[ -d "$root" ] || { echo "install_root_missing:$root" >&2; exit 66; }
mkdir -p "$(dirname -- "$report")"

tool_status() {
  tool=$1
  path=$(find "$root" -type f -name "$tool" -perm -111 2>/dev/null | head -n 1)
  if [ -n "$path" ] && "$path" --help >/dev/null 2>&1; then
    echo pass
  else
    echo fail
  fi
}

decent_q_status=$(tool_status decent_q)
dnnc_status=$(tool_status dnnc)
dlet_status=$(tool_status dlet)
ddump_status=$(tool_status ddump)

result=pass
for status in "$decent_q_status" "$dnnc_status" "$dlet_status" "$ddump_status"; do
  [ "$status" = pass ] || result=fail
done

caffe_status=fail
caffe_init=$(find "$root" -type f -path '*/python/caffe/__init__.py' 2>/dev/null | head -n 1)
if [ -n "$caffe_init" ]; then
  python_root=$(dirname -- "$(dirname -- "$caffe_init")")
  for interpreter in python3 python python2.7; do
    if command -v "$interpreter" >/dev/null 2>&1 && \
       PYTHONPATH="$python_root" "$interpreter" -c 'import caffe' >/dev/null 2>&1; then
      caffe_status=pass
      break
    fi
  done
fi
[ "$caffe_status" = pass ] || result=fail

cat > "$report" <<EOF
DNNDK_ROOT=$root
DECENT_Q_STATUS=$decent_q_status
DNNC_STATUS=$dnnc_status
DLET_STATUS=$dlet_status
DDUMP_STATUS=$ddump_status
CAFFE_PYTHON_STATUS=$caffe_status
RESULT=$result
EOF
cat "$report"
[ "$result" = pass ]
