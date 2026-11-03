#!/usr/bin/env python3
"""Read-only audit of actual D4 archive bytes. Never a board-pass certificate."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import tarfile
import xml.etree.ElementTree as ET
import zipfile


HARDWARE_SHA = {
    'SJD4PL.BIT': 'b7b2b8f41f641a1dc8f684313531c8aa95ad95cde42e86e7366fc0e78dfff907',
    'SJD4PL.DTB': 'f32fdba24e81f859166d871f6732b06469f85063993255e78f37d2eeb32c066a',
    'hardware/soundjie_d4_combined.hdf': '338a00120b72eda6da4a4dbf774052f94b46d6cd36c58ad3a32a3b6ee74a02e1',
}
# Registered only after actual routed-driver positive/negative tests and
# report/export exit 0. Evidence: hdmi-driver-audit-20261023.md. Never accept
# each file independently from a union of revisions: the whole tuple must match.
HDMI_HARDWARE_SHA = {
    'SJD4PL.BIT': 'd4c94f01ecc63e8c30dd964a1a83fe7e35aa442bd7afccf09cfe8ebc6a0e1173',
    'SJD4PL.DTB': 'f32fdba24e81f859166d871f6732b06469f85063993255e78f37d2eeb32c066a',
    'hardware/soundjie_d4_combined.hdf': 'bc2c1a87dfd48b973669f95e222ab385755529a05009df5b494b8a0e5e48d827',
}
CORE_SHA = {
    'model/dpu_soundjie_d3.elf': '66a317bc2c21e9d155d38e195466219bba0a7e722a01d480dbab95c0d930e7cd',
    'config/logmel_frontend_v1.json': 'bd11b2f97b4b82e34d39a78121df92b9d1c3620de1ab296934b04c1ff995a0b1',
    'config/soundjie_d3_classes.json': 'd2e1045867e463878f7c2fa54246be0ffd99b8912e178596fe0bb932c5c01bc8',
}
EXECUTABLES = ('runtime/soundjie_d4_runtime', 'runtime/soundjie_d4_classifier',
               'runtime/soundjie_hdmi_ui', 'runtime/xvf3800_doa',
               'audio/bin/arecord', 'dnndk/pkgs/bin/dexplorer')
LIBRARIES = ('audio/lib/libasound.so.2', 'dnndk/pkgs/lib/libn2cube.so', 'dnndk/pkgs/lib/libhineon.so')
REQUIRED = set(EXECUTABLES + LIBRARIES) | set(CORE_SHA) | {
    'runtime/dpu-4.14.0-xilinx.ko', 'audio/load-usb-audio.sh',
    'audio/modules/snd-hwdep.ko', 'audio/modules/snd-rawmidi.ko',
    'audio/modules/snd-usbmidi-lib.ko', 'audio/modules/snd-usb-audio.ko',
    'audio/share/alsa/alsa.conf', 'probe/run_soundjie_d4_session.sh',
    'probe/run_soundjie_d4_runtime.sh', 'config/soundjie_d4_runtime.tsv',
    'model/labels.txt', 'model/display_labels_zh.txt', 'assets/hdmi/soundjie_glyph_atlas.bin',
    'licenses/ESC-50-LICENSE', 'licenses/LibriSpeech-LICENSE.TXT',
    'licenses/GNU-Unifont-COPYING', 'licenses/GNU-Unifont-OFL-1.1.txt',
    'licenses/gnu-unifont-17.0.04.md', 'licenses/respeaker-xvf3800-source.md',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def hardware_profile(hardware):
    digests = {name: sha(hardware[name]) for name in HARDWARE_SHA if name in hardware}
    for profile, expected in (('legacy_lcd', HARDWARE_SHA), ('hdmi_20261023', HDMI_HARDWARE_SHA)):
        if digests == expected:
            return profile
    raise ValueError('hardware SHA mismatch: unregistered or mixed BIT/HDF/DTB pair')


def elf_sections(data):
    if len(data) < 52 or data[:7] != b'\x7fELF\x01\x01\x01':
        raise ValueError('expected ELF32 little-endian')
    header = struct.unpack_from('<HHIIIIIHHHHHH', data, 16)
    if header[1] != 40:
        raise ValueError('expected ARM machine')
    offset, size, count, strings_index = header[5], header[10], header[11], header[12]
    if size != 40 or not count or strings_index >= count or offset + size * count > len(data):
        raise ValueError('invalid ELF section table')
    rows = [struct.unpack_from('<10I', data, offset + size * i) for i in range(count)]
    strings_row = rows[strings_index]
    strings = data[strings_row[4]:strings_row[4] + strings_row[5]]
    result = {}
    for row in rows:
        if row[0] >= len(strings) or b'\0' not in strings[row[0]:]:
            raise ValueError('invalid ELF section name')
        name = strings[row[0]:].split(b'\0', 1)[0].decode('ascii')
        if row[1] != 8 and row[4] + row[5] > len(data):
            raise ValueError('ELF section out of bounds')
        if name in result:
            raise ValueError('duplicate ELF section')
        result[name] = {'offset': row[4], 'type': row[1],
                        'data': b'' if row[1] == 8 else data[row[4]:row[4] + row[5]]}
    return result


def elf_contract(data, executable=False):
    sections = elf_sections(data)
    flags = struct.unpack_from('<I', data, 36)[0]
    if flags >> 24 != 5 or not flags & 0x400 or flags & 0x200:
        raise ValueError('expected EABI5 hard float')
    if executable and sections.get('.interp', {}).get('data') != b'/lib/ld-linux-armhf.so.3\0':
        raise ValueError('wrong ARM interpreter')
    # Dynamic string table, not debug strings or arbitrary embedded text.
    strings = sections.get('.dynstr', {}).get('data', b'')
    versions = [tuple(map(int, match.split(b'.'))) for match in
                re.findall(rb'\bGLIBC_([0-9]+(?:\.[0-9]+)+)\x00', strings)]
    if not versions or max(versions) > (2, 17):
        raise ValueError('missing or too-new GLIBC dependency')
    return {'machine': 'ARM', 'abi': 'EABI5-hardfloat',
            'max_glibc': '.'.join(map(str, max(versions)))}


def compare_model_sections(model, classifier):
    left, right = elf_sections(model), elf_sections(classifier)
    names = {'.deephi.' + kind + '.soundjie_d3' for kind in
             ('strtab', 'metadata', 'tensor', 'node', 'parameter', 'code')}
    for table in (left, right):
        if {n for n in table if n.startswith('.deephi.')} != names:
            raise ValueError('DPU section set mismatch')
    for name in names:
        if not left[name]['data'] or left[name]['data'] != right[name]['data']:
            raise ValueError('linked DPU section mismatch: ' + name)
    return len(names)


def read_dtb(data):
    if len(data) < 40:
        raise ValueError('short DTB')
    magic, total, structure, strings, _, version, _, _, string_size, structure_size = struct.unpack_from('>10I', data)
    if magic != 0xd00dfeed or total != len(data) or version != 17:
        raise ValueError('invalid DTB header')
    if structure + structure_size > total or strings + string_size > total:
        raise ValueError('invalid DTB bounds')
    names = data[strings:strings + string_size]
    block = data[structure:structure + structure_size]
    pos, stack, tree = 0, [], {}
    while pos + 4 <= len(block):
        token = struct.unpack_from('>I', block, pos)[0]
        pos += 4
        if token == 1:
            end = block.index(b'\0', pos)
            stack.append(block[pos:end].decode('ascii'))
            key = '/' + '/'.join(stack[1:])
            if key in tree:
                raise ValueError('duplicate DTB node')
            tree[key] = {}
            pos = (end + 4) & ~3
        elif token == 2:
            if not stack:
                raise ValueError('unbalanced DTB')
            stack.pop()
        elif token == 3:
            length, name_offset = struct.unpack_from('>II', block, pos)
            pos += 8
            if not stack or name_offset >= len(names) or pos + length > len(block):
                raise ValueError('invalid DTB property')
            name = names[name_offset:names.index(b'\0', name_offset)].decode('ascii')
            props = tree['/' + '/'.join(stack[1:])]
            if name in props:
                raise ValueError('duplicate DTB property')
            props[name] = block[pos:pos + length]
            pos = (pos + length + 3) & ~3
        elif token == 9:
            if stack:
                raise ValueError('unclosed DTB node')
            return tree
        elif token != 4:
            raise ValueError('unknown DTB token')
    raise ValueError('missing DTB end')


def hardware_semantics(tree, hdf):
    with zipfile.ZipFile(io.BytesIO(hdf)) as archive:
        hardware = ET.fromstring(archive.read('system.hwh'))
    vdma = next(m for m in hardware.iter('MODULE') if m.get('INSTANCE') == 'axi_vdma_0')
    params = {p.get('NAME'): p.get('VALUE') for p in vdma.findall('./PARAMETERS/PARAMETER')}
    node = tree['/amba_pl/dma@43000000']
    stores = struct.unpack('>I', node['xlnx,num-fstores'])[0]
    if stores != int(params['C_NUM_FSTORES']):
        raise ValueError('VDMA frame stores mismatch')
    channel = tree['/amba_pl/dma@43000000/dma-channel@43000000']
    if ('xlnx,genlock-mode' in channel) != bool(int(params['C_MM2S_GENLOCK_MODE'])):
        raise ValueError('VDMA genlock mismatch')
    dpu = tree['/amba_pl/dpu_eu@4f000000']
    checks = [(node['reg'], (0x43000000, 0x10000)),
              (node['interrupts'], (0, 30, 4)),
              (dpu['reg'], (0x4f000000, 0x1000000)),
              (dpu['interrupts'], (0, 31, 4))]
    if any(raw != struct.pack('>' + 'I' * len(values), *values) for raw, values in checks):
        raise ValueError('hardware address/IRQ mismatch')
    if tree['/axi_vdma_lcd']['compatible'] != b'topic,vdma-fb\0':
        raise ValueError('framebuffer driver binding mismatch')
    ports = {p.get('NAME') for p in hardware.findall('./EXTERNALPORTS/PORT')}
    # Retained LCD ports do not disqualify the new implementation. Only the
    # precisely registered HDF with physically audited transmitter is qualified;
    # audit_payloads also requires its exact matching BIT and DTB tuple.
    lcd_only = {'lcd_clk', 'lcd_data', 'lcd_de', 'lcd_hs', 'lcd_vs'} <= ports
    hdmi_ports = {'HDMI_CLK_P', 'HDMI_CLK_N', 'HDMI_D0_P', 'HDMI_D0_N',
                  'HDMI_D1_P', 'HDMI_D1_N', 'HDMI_D2_P', 'HDMI_D2_N'}
    transmitter = any(m.get('INSTANCE') == 'soundjie_hdmi_tx' and
                      m.get('VLNV') == 'xilinx.com:user:HDMI_FPGA_ML:1.0' for m in hardware.iter('MODULE'))
    hdmi_verified = (sha(hdf) == HDMI_HARDWARE_SHA['hardware/soundjie_d4_combined.hdf']
                     and hdmi_ports <= ports and transmitter)
    return {'vdma_frame_stores': stores, 'vdma_spi': 30, 'dpu_spi': 31,
            'display_output': 'HDMI_TMDS_VIDEO' if hdmi_verified else ('LCD_RGB_NO_HDMI' if lcd_only else 'UNVERIFIED'),
            'hdmi_output_verified': hdmi_verified,
            'pair_basis': 'pinned implementation BIT/HDF/DTB plus parsed DT/VDMA semantics'}


def audit_payloads(members, hardware):
    missing = REQUIRED - members.keys()
    if missing:
        raise ValueError('required members missing: ' + ', '.join(sorted(missing)))
    profile = hardware_profile(hardware)
    manifest_name = 'manifest/candidate.json' if 'manifest/candidate.json' in members else 'manifest/development.json'
    manifest = json.loads(members[manifest_name])
    status = 'LOCAL_BOARD_TEST_CANDIDATE' if manifest_name.endswith('candidate.json') else 'DEVELOPMENT_ONLY_NOT_RELEASED'
    if manifest['board_tested'] is not False or manifest['status'] != status:
        raise ValueError('unexpected manifest status')
    expected = {m['name']: m for m in manifest['members']}
    if len(expected) != len(manifest['members']) or members.keys() != expected.keys() | {manifest_name, 'manifest/SHA256SUMS'}:
        raise ValueError('manifest member set mismatch')
    for name, item in expected.items():
        if sha(members[name]) != item['sha256'] or len(members[name]) != item['size']:
            raise ValueError('manifest checksum mismatch: ' + name)
        if (name in EXECUTABLES or name.endswith('.sh')) and item['mode'] != '0755':
            raise ValueError('missing executable mode: ' + name)
    sums = {}
    for line in members['manifest/SHA256SUMS'].decode('ascii').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if not match or match[2] in sums:
            raise ValueError('invalid checksum list')
        sums[match[2]] = match[1]
    if sums.keys() != expected.keys() | {manifest_name}:
        raise ValueError('checksum member set mismatch')
    if any(sha(members[name]) != digest for name, digest in sums.items()):
        raise ValueError('inner checksum mismatch')
    for name, digest in CORE_SHA.items():
        if sha(members[name]) != digest:
            raise ValueError('core SHA mismatch: ' + name)
    elf = {name: elf_contract(members[name], name in EXECUTABLES) for name in EXECUTABLES + LIBRARIES}
    count = compare_model_sections(members['model/dpu_soundjie_d3.elf'], members['runtime/soundjie_d4_classifier'])
    classifier = members['runtime/soundjie_d4_classifier']
    for name in ('model/dpu_soundjie_d3.elf', 'config/logmel_frontend_v1.json'):
        if CORE_SHA[name].encode('ascii') not in classifier:
            raise ValueError('classifier missing compiled SHA binding')
    for name in members:
        if name.endswith('.sh') and b'\r' in members[name]:
            raise ValueError('shell script must use LF')
    return {'status': 'HOST_BYTES_CHECKED_BOARD_PENDING', 'board_tested': False,
            'hardware_profile': profile,
            'distribution_license_clearance': False, 'elf': elf,
            'model_sections_matched': count, 'model_sha256': CORE_SHA['model/dpu_soundjie_d3.elf'],
            'hardware': hardware_semantics(read_dtb(hardware['SJD4PL.DTB']), hardware['hardware/soundjie_d4_combined.hdf'])}


def audit_directory(folder):
    folder = Path(folder)
    archive_path = folder / ('SJD4A.TGZ' if (folder / 'SJD4A.TGZ').exists() else 'SJD4DEV.TGZ')
    prefix = archive_path.stem
    expected_outer = {prefix + '.TGZ'} | set(HARDWARE_SHA)
    outer = {}
    for line in (folder / (prefix + '.SHA')).read_text(encoding='ascii').splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if not match or match[2] in outer or match[2] not in expected_outer:
            raise ValueError('invalid outer checksum entry')
        outer[match[2]] = match[1]
    if outer.keys() != expected_outer:
        raise ValueError('outer checksum member set mismatch')
    # No paths from an unchecked checksum file are opened.
    for name, digest in outer.items():
        if sha((folder / name).read_bytes()) != digest:
            raise ValueError('outer checksum mismatch: ' + name)
    with tarfile.open(archive_path) as archive:
        members = {}
        modes = {}
        for item in archive:
            if not item.isfile() or item.name in members:
                raise ValueError('nonregular or duplicate tar member')
            members[item.name] = archive.extractfile(item).read()
            modes[item.name] = item.mode
    hardware = {name: (folder / name).read_bytes() for name in HARDWARE_SHA}
    report = audit_payloads(members, hardware)
    manifest_name = 'manifest/candidate.json' if prefix == 'SJD4A' else 'manifest/development.json'
    for item in json.loads(members[manifest_name])['members']:
        if modes[item['name']] != int(item['mode'], 8):
            raise ValueError('archive mode differs from manifest')
    report['archive_sha256'] = sha(archive_path.read_bytes())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_directory(args.bundle_dir), indent=2, sort_keys=True))
