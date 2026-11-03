"""Exercise the compiled layout gate with ARM ELF bytes, not audit mocks."""
import importlib.util
import struct
import unittest
from pathlib import Path


HELPER = Path(__file__).resolve().parents[1] / 'scripts/host/audit_alert4_dpu_layout.py'


def fixture():
    """Small ELF with the observed DNNC v2.05 tensor/node ABI and FC layout."""
    strings = bytearray(b'\0')

    def string(value):
        offset = len(strings)
        strings.extend(value.encode('ascii') + b'\0')
        return offset

    compiler = string('dnnc version v2.05\nDPU Target : v1.4.0\n')
    target = string('DPU_1152FA')
    tensors = [
        (2, 64, 96, 1, 0, 6144, 8, 7, 1, 0, 0, 0, 0),
        (1, 32, 48, 16, 6144, 24576, 8, 4, 16, 0, 0, 0, 0),
        (1, 16, 24, 32, 30720, 12288, 8, 4, 32, 0, 0, 0, 0),
        (1, 8, 12, 64, 0, 6144, 8, 4, 64, 0, 0, 0, 0),
        (1, 4, 6, 96, 6144, 2304, 8, 4, 96, 0, 0, 0, 0),
        (4, 1, 1, 6, 0, 6, 8, 4, 6, 0, 0, 0, 0),
    ]
    params = bytearray()
    nodes = []
    for i, name in enumerate(['conv1', 'conv2', 'conv3', 'conv4', 'fc8']):
        channels = [16, 32, 64, 96, 6][i]
        weight_dims = [(16, 1, 3, 3), (32, 16, 3, 3), (64, 32, 3, 3),
                       (96, 64, 3, 3), (6, 96, 4, 6)][i]
        weight_size = 1
        for dimension in weight_dims:
            weight_size *= dimension
        bias_offset = len(params)
        params.extend(bytes([1]) * channels)
        weight_offset = len(params)
        params.extend(bytes([2]) * weight_size)
        if i == 4:
            params[bias_offset + 5] = 0
            params[-2304:] = bytes(2304)
        node = [1, string(name), 27648, 0, 3, 2, 65536, 131075,
                1, i, 1, i + 1, 1, i * 4, 4, string(name + '_code'),
                4, 2, bias_offset, channels, 8, 13 if i == 4 else 6,
                string(name + '_bias'), 2, channels, 1, 1, 1,
                weight_offset, weight_size, 8, 10, string(name + '_weights'),
                2, *weight_dims, 0, 0]
        nodes.extend(node)
    metadata = (target, compiler, 0, 5, 52, 43008, 0xffffffff, 0xffffffff,
                0xffffffff, 65543, 131080, 0, 0, 0, 0, 0)
    parts = [
        ('.shstrtab', 3, b''),
        ('.deephi.strtab.soundjie_d3', 1, bytes(strings)),
        ('.deephi.metadata.soundjie_d3', 1, struct.pack('<16I', *metadata)),
        ('.deephi.tensor.soundjie_d3', 1, b''.join(struct.pack('<13I', *t) for t in tensors)),
        ('.deephi.node.soundjie_d3', 1, struct.pack('<200I', *nodes)),
        ('.deephi.parameter.soundjie_d3', 1, bytes(params)),
        ('.deephi.code.soundjie_d3', 1, bytes(20)),
    ]
    section_names = b'\0' + b''.join(name.encode() + b'\0' for name, _, _ in parts)
    parts[0] = ('.shstrtab', 3, section_names)
    raw = bytearray(52)
    headers = [bytes(40)]
    offsets = {}
    for index, (name, kind, data) in enumerate(parts, 1):
        offset = len(raw)
        raw.extend(data)
        offsets[name] = (offset, len(data), index)
        headers.append(struct.pack('<10I', section_names.index(name.encode() + b'\0'),
                                   kind, 0, 0, offset, len(data), 0, 0, 1, 0))
    table = len(raw)
    raw.extend(b''.join(headers))
    ident = b'\x7fELF\x01\x01\x01' + bytes(9)
    struct.pack_into('<16sHHIIIIIHHHHHH', raw, 0, ident, 1, 40, 1, 0, 0,
                     table, 0x05000000, 52, 0, 0, 40, len(headers), 1)
    return raw, offsets


class Alert4DpuLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HELPER.is_file():
            cls.audit = None
            return
        spec = importlib.util.spec_from_file_location('alert4_layout_test_module', HELPER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.audit = staticmethod(module.audit)

    def setUp(self):
        self.assertIsNotNone(self.audit, 'compiled ELF layout audit is missing')
        self.raw, self.offsets = fixture()

    def word(self, section, offset, value):
        base = self.offsets['.deephi.' + section + '.soundjie_d3'][0]
        struct.pack_into('<I', self.raw, base + offset, value)

    def test_accepts_only_explicit_five_semantic_six_storage_layout(self):
        result = self.audit(bytes(self.raw))
        self.assertEqual(result['logical_output_count'], 5)
        self.assertEqual(result['storage_output_count'], 6)
        self.assertEqual(result['output']['shape_hwc'], [1, 1, 6])
        self.assertEqual(result['output']['size'], 6)
        self.assertEqual(result['output']['fix_pos'], 4)
        self.assertEqual(result['padding']['index'], 5)
        self.assertEqual(result['padding']['weight_bytes'], 2304)
        self.assertTrue(result['padding']['weights_zero'])
        self.assertTrue(result['padding']['bias_zero'])

    def test_rejects_changed_output_fields_and_legacy_eight_shape(self):
        for field, value in [(0, 1), (1, 2), (2, 2), (3, 5), (3, 8), (5, 5),
                             (5, 8), (6, 16), (7, 3), (8, 8), (9, 1)]:
            with self.subTest(field=field, value=value):
                self.raw, self.offsets = fixture()
                self.word('tensor', 5 * 52 + field * 4, value)
                with self.assertRaises(ValueError):
                    self.audit(bytes(self.raw))

    def test_rejects_input_or_fc_input_shape_drift(self):
        for record, field, value in [(0, 5, 6143), (0, 7, 6), (4, 3, 95), (4, 5, 2303)]:
            with self.subTest(record=record, field=field):
                self.raw, self.offsets = fixture()
                self.word('tensor', record * 52 + field * 4, value)
                with self.assertRaises(ValueError):
                    self.audit(bytes(self.raw))

    def test_rejects_real_sixth_bias_and_any_nonzero_padding_weight(self):
        base, size, _ = self.offsets['.deephi.parameter.soundjie_d3']
        for offset in [base + size - 13824 - 1, base + size - 2304,
                       base + size - 1001, base + size - 1]:
            with self.subTest(offset=offset):
                changed = self.raw[:]
                changed[offset] = 1
                with self.assertRaises(ValueError):
                    self.audit(bytes(changed))

    def test_does_not_reject_zero_semantic_bias_or_zero_individual_weights(self):
        base, size, _ = self.offsets['.deephi.parameter.soundjie_d3']
        self.raw[base + size - 13824 - 6] = 0
        self.raw[base + size - 13824] = 0
        self.audit(bytes(self.raw))

    def test_rejects_wrong_fc_binding_sizes_quantization_and_param_offsets(self):
        for field, value in [(9, 3), (11, 4), (17, 1), (18, 0xffffffff), (19, 5),
                             (20, 16), (21, 12), (24, 5), (28, 0xffffffff),
                             (29, 11520), (30, 16), (31, 9), (34, 5), (35, 95)]:
            with self.subTest(field=field):
                self.raw, self.offsets = fixture()
                self.word('node', 4 * 160 + field * 4, value)
                with self.assertRaises(ValueError):
                    self.audit(bytes(self.raw))

    def test_rejects_truncated_wrong_arch_and_unsupported_tensor_abi(self):
        for length in [0, 16, 51, len(self.raw) - 1]:
            with self.subTest(length=length), self.assertRaises(ValueError):
                self.audit(bytes(self.raw[:length]))
        for offset, value in [(4, 2), (5, 2), (18, 62)]:
            changed = self.raw[:]
            changed[offset] = value
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.audit(bytes(changed))
        for field, value in [(3, 4), (4, 48)]:
            self.raw, self.offsets = fixture()
            self.word('metadata', field * 4, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit(bytes(self.raw))

    def test_rejects_missing_duplicate_or_out_of_bounds_sections(self):
        table = struct.unpack_from('<I', self.raw, 32)[0]
        tensor_index = self.offsets['.deephi.tensor.soundjie_d3'][2]
        node_index = self.offsets['.deephi.node.soundjie_d3'][2]
        changes = [(table + tensor_index * 40, 0),
                   (table + node_index * 40, struct.unpack_from('<I', self.raw, table + tensor_index * 40)[0]),
                   (table + tensor_index * 40 + 16, len(self.raw)),
                   (table + tensor_index * 40 + 20, 0xffffffff)]
        for offset, value in changes:
            changed = self.raw[:]
            struct.pack_into('<I', changed, offset, value)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                self.audit(bytes(changed))


if __name__ == '__main__':
    unittest.main()
