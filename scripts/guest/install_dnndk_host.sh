#!/bin/sh
set -eu

archive=
contract=
install_root=
report=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --archive) archive=$2; shift 2 ;;
    --contract) contract=$2; shift 2 ;;
    --install-root) install_root=$2; shift 2 ;;
    --report) report=$2; shift 2 ;;
    *) echo "unknown_argument:$1" >&2; exit 64 ;;
  esac
done

case "$archive:$contract:$install_root:$report" in
  /*:/*:/*:/*) ;;
  *) echo "absolute_paths_required" >&2; exit 64 ;;
esac

if [ -e "$install_root" ]; then
  echo "install_root_exists:$install_root" >&2
  exit 73
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
auditor="$script_dir/../host/audit_dnndk_archive.py"
audit_report="${report}.audit"
python3 "$auditor" "$archive" --contract "$contract" --report "$audit_report"

parent=$(dirname -- "$install_root")
mkdir -p "$parent" "$(dirname -- "$report")"
temp=$(mktemp -d "${install_root}.tmp.XXXXXX")
cleanup() {
  if [ -n "${temp:-}" ] && [ -d "$temp" ]; then
    rm -rf -- "$temp"
  fi
}
trap cleanup EXIT HUP INT TERM

tar -xzf "$archive" -C "$temp" --no-same-owner --no-same-permissions
mv -- "$temp" "$install_root"
temp=

cat > "$report" <<EOF
INSTALL_ROOT=$install_root
ARCHIVE=$archive
AUDIT_REPORT=$audit_report
RESULT=pass
EOF
echo "RESULT=pass"
