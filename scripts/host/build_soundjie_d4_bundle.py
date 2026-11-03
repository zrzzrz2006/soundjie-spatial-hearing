#!/usr/bin/env python3
"""Create a development package or host-audited LOCAL board-test candidate.

Neither mode certifies physical acceptance or third-party redistribution rights.
"""
import argparse
import gzip
import hashlib
import io
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def deployment_name_allowed(name):
    fixed = {
        'runtime/soundjie_d4_runtime', 'runtime/soundjie_d4_classifier',
        'runtime/soundjie_hdmi_ui', 'runtime/xvf3800_doa', 'runtime/dpu-4.14.0-xilinx.ko',
        'dnndk/pkgs/bin/dexplorer', 'dnndk/pkgs/lib/libn2cube.so', 'dnndk/pkgs/lib/libhineon.so',
        'audio/bin/arecord', 'audio/lib/libasound.so.2', 'audio/load-usb-audio.sh',
        'audio/modules/snd-hwdep.ko', 'audio/modules/snd-rawmidi.ko',
        'audio/modules/snd-usbmidi-lib.ko', 'audio/modules/snd-usb-audio.ko',
        'model/dpu_soundjie_d3.elf', 'model/labels.txt', 'model/display_labels_zh.txt',
        'config/logmel_frontend_v1.json', 'config/soundjie_d4_runtime.tsv',
        'config/soundjie_d3_classes.json', 'assets/hdmi/soundjie_glyph_atlas.bin',
    }
    path = PurePosixPath(name)
    return (name in fixed or
            (name.startswith(('probe/', 'audio/board-tools/')) and path.suffix == '.sh') or
            (name.startswith('audio/share/alsa/') and path.suffix in ('', '.conf', '.alisp')) or
            (name.startswith('licenses/') and path.suffix.lower() in ('', '.md', '.txt')))


def checked_entry(item, hardware=False):
    name = item['name']
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or path.as_posix() != name or
            any(p in ('..', '.') for p in path.parts) or
            not re.fullmatch(r'[A-Za-z0-9_./-]+', name)):
        raise ValueError('unsafe member name: ' + name)
    if hardware:
        if name not in {'SJD4PL.BIT', 'SJD4PL.DTB', 'hardware/soundjie_d4_combined.hdf'}:
            raise ValueError('unexpected hardware member')
    elif not deployment_name_allowed(name):
        raise ValueError('member outside deployment allowlist: ' + name)
    source = Path(item['source'])
    if source.is_symlink() or not source.is_file():
        raise ValueError('source must be a regular non-symlink file: ' + name)
    payload = source.read_bytes()
    if len(payload) != item['size'] or sha(payload) != item['sha256']:
        raise ValueError('source size or SHA256 mismatch: ' + name)
    mode = item['mode']
    if mode not in ('0644', '0755'):
        raise ValueError('unsupported file mode')
    source_sha = sha(payload)
    # Keep pinned JSON/model bytes unchanged; only shell scripts need conversion.
    if name.endswith('.sh'):
        payload = payload.replace(b'\r\n', b'\n')
        if b'\r' in payload:
            raise ValueError('bare CR in shell script')
    elif item.get('normalize_lf'):
        raise ValueError('normalization is restricted to shell scripts')
    metadata = {'name': name, 'size': len(payload), 'sha256': sha(payload),
                'source_sha256': source_sha, 'mode': mode}
    return metadata, payload


def add(archive, name, payload, mode=0o644):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = mode
    info.mtime = info.uid = info.gid = 0
    info.uname = info.gname = ''
    archive.addfile(info, io.BytesIO(payload))


def build_development_bundle(config, output, *, local_candidate=False):
    raw_config = Path(config).read_bytes()
    data = json.loads(raw_config)
    if (data.get('schema') != 'soundjie-d4-bundle-inputs-v1' or
            data.get('status') != 'preflight_only_not_packaged' or
            not re.fullmatch('[0-9a-f]{40}', data.get('source_commit', '')) or
            not data.get('pending_gates') or not data.get('archive_members')):
        raise ValueError('invalid development input contract')
    entries = []
    hardware = []
    seen = set()
    for kind, target in (('archive_members', entries), ('external_hardware', hardware)):
        for item in data.get(kind, []):
            metadata, payload = checked_entry(item, hardware=kind == 'external_hardware')
            key = metadata['name'].casefold()
            if key in seen:
                raise ValueError('duplicate member name')
            seen.add(key)
            target.append((metadata, payload))
        target.sort(key=lambda row: row[0]['name'])
    for key in seen:
        if any(str(parent) in seen for parent in PurePosixPath(key).parents):
            raise ValueError('a regular file is an ancestor of another member')
    manifest_name = 'manifest/candidate.json' if local_candidate else 'manifest/development.json'
    prefix = 'SJD4A' if local_candidate else 'SJD4DEV'
    manifest = {'schema': 1, 'status': 'LOCAL_BOARD_TEST_CANDIDATE' if local_candidate else 'DEVELOPMENT_ONLY_NOT_RELEASED',
                'board_tested': False, 'independent_acceptance': False,
                'source_commit': data['source_commit'], 'input_manifest_sha256': sha(raw_config),
                'pending_gates': data['pending_gates'],
                'members': [m for m, _ in entries],
                'external_hardware': [m for m, _ in hardware]}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode('utf-8')
    sums = ''.join(m['sha256'] + '  ' + m['name'] + '\n' for m, _ in entries)
    if local_candidate:
        spec = importlib.util.spec_from_file_location('d4_package_audit', Path(__file__).with_name('audit_soundjie_d4_package.py'))
        auditor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auditor)
        payloads = {m['name']: payload for m, payload in entries}
        payloads[manifest_name] = manifest_bytes
        payloads['manifest/SHA256SUMS'] = (sums + sha(manifest_bytes) + '  ' + manifest_name + '\n').encode('ascii')
        manifest['host_audit'] = auditor.audit_payloads(payloads, {m['name']: p for m, p in hardware})
        if not manifest['host_audit']['hardware']['hdmi_output_verified']:
            raise ValueError('HDMI output is not implemented/verified in the pinned hardware; candidate blocked')
        manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode('utf-8')
    sums += sha(manifest_bytes) + '  ' + manifest_name + '\n'
    # All input validation completes before claiming a new output directory.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / (prefix + '.TGZ')
    with archive_path.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.USTAR_FORMAT) as archive:
                for metadata, payload in entries:
                    add(archive, metadata['name'], payload, int(metadata['mode'], 8))
                add(archive, manifest_name, manifest_bytes)
                add(archive, 'manifest/SHA256SUMS', sums.encode('ascii'))
    outputs = [(prefix + '.TGZ', sha(archive_path.read_bytes()))]
    for metadata, payload in hardware:
        destination = output / metadata['name']
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(payload)
        outputs.append((metadata['name'], metadata['sha256']))
    # The checksum marker is written last. A partial directory must never be used.
    if local_candidate:
        with (output / 'SJD4PL.SHA').open('x', encoding='ascii', newline='\n') as stream:
            stream.write(''.join(digest + '  ' + name + '\n' for name, digest in outputs[1:]))
    with (output / (prefix + '.SHA')).open('x', encoding='ascii', newline='\n') as stream:
        stream.write(''.join(digest + '  ' + name + '\n' for name, digest in outputs))
    return {'status': manifest['status'], 'members': len(entries), 'sha256': outputs[0][1]}


def build_local_candidate(config, output):
    return build_development_bundle(config, output, local_candidate=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--local-candidate', action='store_true', help='host-audited local test only; not a redistribution release')
    args = parser.parse_args()
    print(json.dumps(build_development_bundle(args.config, args.output_dir, local_candidate=args.local_candidate), indent=2))
