"""Compile the actual DT and compare its VDMA settings with exported hardware."""
import hashlib
import re
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from scripts.host.audit_soundjie_d4_package import read_dtb

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('F:/MLK/DPU_ZYNQ7000')
HDF = BASE / 'work/soundjie_d4_pl_20261020_184100/final_20261020_224500/soundjie_d4_combined.hdf'
VENDOR = BASE / 'work/soundjie_d4_pl_restaged_20261021a/sources/ch24/CH24_LCD_Tourch/02_fpga/fpga/system/system.sdk/device_tree_bsp_0'
DTC = '/mnt/e/Xilinx/Vivado/2024.2/bin/dtc'


class DtbHardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert hashlib.sha256(HDF.read_bytes()).hexdigest() == '338a00120b72eda6da4a4dbf774052f94b46d6cd36c58ad3a32a3b6ee74a02e1'
        with zipfile.ZipFile(HDF) as archive:
            hardware = ET.fromstring(archive.read('system.hwh'))
        vdma = next(m for m in hardware.iter('MODULE') if m.get('INSTANCE') == 'axi_vdma_0')
        cls.params = {p.get('NAME'): p.get('VALUE') for p in vdma.findall('./PARAMETERS/PARAMETER')}
        cls.temp = tempfile.TemporaryDirectory(prefix='d4-dtb-test-', dir=BASE / 'work')
        cls.addClassCleanup(cls.temp.cleanup)
        folder = Path(cls.temp.name)
        for name in ('system-top.dts', 'zynq-7000.dtsi', 'skeleton.dtsi', 'pcw.dtsi'):
            shutil.copyfile(VENDOR / name, folder / name)
        shutil.copyfile(ROOT / 'config/soundjie_d4_combined_pl.dtsi', folder / 'pl.dtsi')
        linux = '/mnt/f/' + folder.as_posix()[3:]
        subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', DTC, '-I', 'dts', '-O', 'dtb', '-o', linux + '/test.dtb', linux + '/system-top.dts'], check=True, capture_output=True)
        cls.tree = subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', DTC, '-I', 'dtb', '-O', 'dts', linux + '/test.dtb'], check=True, capture_output=True).stdout.decode('utf-8')
        cls.nodes = read_dtb((folder / 'test.dtb').read_bytes())
        cls.vdma = cls.tree.split('dma@43000000 {', 1)[1].split('dpu_eu@', 1)[0]

    def test_compiled_frame_store_count_matches_hardware(self):
        value = re.search(r'xlnx,num-fstores = <(0x[0-9a-f]+|[0-9]+)>', self.vdma).group(1)
        self.assertEqual(int(value, 0), int(self.params['C_NUM_FSTORES']))

    def test_compiled_genlock_flag_matches_mm2s_mode(self):
        self.assertEqual('xlnx,genlock-mode;' in self.vdma, int(self.params['C_MM2S_GENLOCK_MODE']) != 0)

    def test_usb_controller_retains_board_host_mode(self):
        usb = self.nodes['/amba/usb@e0002000']
        self.assertEqual(usb.get('dr_mode'), b'host\0')

    def test_usb_phy_reference_resolves_to_board_ulpi(self):
        usb = self.nodes['/amba/usb@e0002000']
        self.assertIn('usb-phy', usb)
        targets = [p for p in self.nodes.values() if p.get('phandle') == usb['usb-phy']]
        self.assertEqual(len(targets), 1)
        phy = targets[0]
        self.assertEqual(phy.get('compatible'), b'ulpi-phy\0')
        self.assertEqual(phy.get('reg'), bytes.fromhex('e000200000001000'))
        self.assertEqual(phy.get('view-port'), bytes.fromhex('00000170'))
        self.assertEqual(phy.get('#phy-cells'), bytes(4))
        self.assertIn('drv-vbus', phy)


if __name__ == '__main__':
    unittest.main()
