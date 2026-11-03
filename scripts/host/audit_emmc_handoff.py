"""Read-only SDIO1/EMIO configuration gate; NOT a wiring or board acceptance test."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


def audit(path):
    raw = path.read_bytes()
    if path.suffix.lower() == '.hdf':
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith('.hwh')]
            if len(names) != 1:
                raise ValueError('HDF must contain exactly one HWH')
            xml = archive.read(names[0])
    elif path.suffix.lower() == '.hwh':
        xml = raw
    else:
        raise ValueError('expected .hdf or .hwh')
    root = ET.fromstring(xml)
    ps = [m for m in root.iter('MODULE') if m.get('MODTYPE') == 'processing_system7']
    if len(ps) != 1:
        raise ValueError('expected exactly one processing_system7 module')
    parameters = {p.get('NAME'): p.get('VALUE') for p in ps[0].findall('./PARAMETERS/PARAMETER')}
    ports = {p.get('NAME'): p.get('DIR') for p in ps[0].findall('./PORTS/PORT')}
    expected = {
        'PCW_SD1_PERIPHERAL_ENABLE': '1', 'PCW_EN_SDIO1': '1',
        'PCW_EN_EMIO_SDIO1': '1', 'PCW_SD1_SD1_IO': 'EMIO',
        'PCW_SD1_GRP_CD_ENABLE': '1', 'PCW_SD1_GRP_CD_IO': 'EMIO',
        'PCW_SD1_GRP_WP_ENABLE': '1', 'PCW_SD1_GRP_WP_IO': 'EMIO',
    }
    errors = [f'{k}: expected {v}, got {parameters.get(k)!r}'
              for k, v in expected.items() if parameters.get(k) != v]
    required_ports = {'CLK': 'O', 'CLK_FB': 'I', 'CMD_I': 'I', 'CMD_O': 'O',
                      'CMD_T': 'O', 'DATA_I': 'I', 'DATA_O': 'O', 'DATA_T': 'O',
                      'CDN': 'I', 'WP': 'I'}
    for suffix, direction in required_ports.items():
        name = 'SDIO1_' + suffix
        if ports.get(name) != direction:
            errors.append(f'{name}: expected direction {direction}, got {ports.get(name)!r}')
    return {
        'input': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
        'status': 'rejected' if errors else 'configuration_present_not_board_validated',
        'board_ready': False, 'errors': errors,
        'not_checked': ['clock feedback wiring', 'IOBUF and pin constraints',
                        'bank voltage', 'timing', 'device tree', 'board operation'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('handoff', type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.handoff)
    except (OSError, ValueError, ET.ParseError, zipfile.BadZipFile, KeyError) as error:
        report = {'status': 'rejected', 'board_ready': False, 'errors': [str(error)]}
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 2 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
