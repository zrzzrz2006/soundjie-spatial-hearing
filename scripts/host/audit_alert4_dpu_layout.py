#!/usr/bin/env python3
"""Audit the observed alert4bg DNNC v2.05 ARM32 ELF storage layout.

``audit(raw)`` accepts the raw relocatable model ELF, not a linked executable.
It raises ValueError on unsupported or malformed layouts. This is deliberately
specific to the audited B1152F model: five semantic logits occupy six channels.
The caller must separately bind the five-output prototxt, class contract and
model hash. A zero sixth row alone cannot identify a model's class semantics.

Evidence: bfaf1f963a6cc9b6a00a4023de36385a858c38efc1541a5dafb286205927e8e6.
Its first 5*2304 FC weight bytes equal quantized Caffe weights * 2**10 after
per-output CHW -> HWC conversion; its first five biases equal Caffe * 2**13.
The sixth bias and final 2304 weight bytes are zero. No runtime/board is used.
"""
import hashlib
import struct


_SUFFIX = '.soundjie_d3'
_TENSORS = (
    (2, 64, 96, 1, 0, 6144, 8, 7, 1, 0, 0, 0, 0),
    (1, 32, 48, 16, 6144, 24576, 8, 4, 16, 0, 0, 0, 0),
    (1, 16, 24, 32, 30720, 12288, 8, 4, 32, 0, 0, 0, 0),
    (1, 8, 12, 64, 0, 6144, 8, 4, 64, 0, 0, 0, 0),
    (1, 4, 6, 96, 6144, 2304, 8, 4, 96, 0, 0, 0, 0),
    (4, 1, 1, 6, 0, 6, 8, 4, 6, 0, 0, 0, 0),
)


def _require(condition, message):
    if not condition:
        raise ValueError('alert4 DPU layout: ' + message)


def _slice(raw, offset, size, label):
    _require(0 <= offset <= len(raw) and 0 <= size <= len(raw) - offset,
             label + ' outside section/file')
    return raw[offset:offset + size]


def _string(raw, offset):
    _require(0 <= offset < len(raw), 'string offset outside table')
    _require(offset == 0 or raw[offset - 1] == 0, 'string offset is not a boundary')
    end = raw.find(b'\0', offset)
    _require(end >= 0, 'unterminated string')
    try:
        return raw[offset:end].decode('ascii')
    except UnicodeDecodeError as error:
        raise ValueError('alert4 DPU layout: non-ASCII string') from error


def _disjoint(ranges, label):
    ordered = sorted((start, start + size) for start, size in ranges if size)
    _require(all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:])),
             'overlapping ' + label)


def _sections(raw):
    _require(isinstance(raw, bytes), 'raw must be ELF bytes')
    header = struct.unpack('<16sHHIIIIIHHHHHH', _slice(raw, 0, 52, 'ELF header'))
    ident, kind, machine, version, entry, phoff, shoff, flags, ehsize, phsize, phnum, shsize, shnum, names_index = header
    _require(ident[:7] == b'\x7fELF\x01\x01\x01' and
             kind == 1 and machine == 40 and version == 1,
             'expected little-endian ELF32 ARM relocatable version 1')
    _require(entry == 0 and phoff == 0 and phnum == 0 and ehsize == 52 and
             shsize == 40 and 0 < names_index < shnum <= 4096,
             'unsupported ELF header/section table')
    table = _slice(raw, shoff, shnum * 40, 'section table')
    headers = list(struct.iter_unpack('<10I', table))
    _require(headers[0] == (0,) * 10, 'invalid null section')
    names = headers[names_index]
    _require(names[1] == 3, 'section names must use a string table')
    names = _slice(raw, names[4], names[5], 'section names')
    result = {}
    ranges = [(0, 52), (shoff, shnum * 40)]
    for section in headers[1:]:
        name = _string(names, section[0])
        _require(name and name not in result, 'missing/duplicate section name')
        _require(section[1] != 8, 'unexpected NOBITS section in raw model ELF')
        data = _slice(raw, section[4], section[5], name)
        result[name] = (section[1], data)
        ranges.append((section[4], section[5]))
    _disjoint(ranges, 'ELF sections')
    expected = {'.deephi.' + name + _SUFFIX
                for name in ('strtab', 'metadata', 'tensor', 'node', 'parameter', 'code')}
    _require({name for name in result if name.startswith('.deephi.')} == expected,
             'missing, extra or differently named DPU sections')
    for name in expected:
        _require(result[name][0] == 1, name + ' must be PROGBITS')
    return {name: data for name, (_, data) in result.items()}


def _tensor_details(record):
    return {'shape_hwc': list(record[1:4]), 'size': record[5],
            'bits': record[6], 'fix_pos': record[7], 'channel_stride': record[8]}


def audit(raw):
    """Return JSON-serializable layout evidence, or raise ValueError.

    This verifies storage/padding, not five-class identity or numerical fidelity.
    Keep the existing prototxt/hash gates and perform fixed-reference DPU parity.
    """
    sections = _sections(raw)
    data = {name: sections['.deephi.' + name + _SUFFIX]
            for name in ('strtab', 'metadata', 'tensor', 'node', 'parameter', 'code')}
    _require(len(data['metadata']) == 64, 'unsupported metadata record')
    metadata = struct.unpack('<16I', data['metadata'])
    _require(metadata[2:5] == (0, 5, 52), 'unsupported node count/tensor ABI')
    _require(_string(data['strtab'], metadata[0]) == 'DPU_1152FA', 'wrong DPU target')
    compiler = _string(data['strtab'], metadata[1])
    _require(compiler.startswith('dnnc version v2.05\nDPU Target : v1.4.0\n'),
             'unsupported compiler/target version')
    _require(len(data['tensor']) == 6 * 52, 'expected six tensor records')
    tensors = tuple(struct.iter_unpack('<13I', data['tensor']))
    _require(tensors == _TENSORS, 'tensor shape/stride/quantization differs from audited model')
    _require(len(data['node']) == 5 * 160, 'unsupported node record layout')
    nodes = list(struct.iter_unpack('<40I', data['node']))
    parameter_ranges, code_ranges = [], []
    for index, (name, node) in enumerate(zip(('conv1', 'conv2', 'conv3', 'conv4', 'fc8'), nodes)):
        _require(node[0] == 1 and _string(data['strtab'], node[1]) == name,
                 'wrong node identity/order')
        _require(node[4:13] == (3, 2, 65536, 131075, 1, index, 1, index + 1, 1) and
                 node[16:18] == (4, 2), 'unsupported node tensor/parameter binding')
        for pointer, suffix in ((15, '_code'), (22, '_bias'), (32, '_weights')):
            _require(_string(data['strtab'], node[pointer]) == name + suffix,
                     'wrong node segment name')
        for offset, size in ((node[18], node[19]), (node[28], node[29])):
            _require(size > 0, 'empty parameter segment')
            _slice(data['parameter'], offset, size, name + ' parameters')
            parameter_ranges.append((offset, size))
        _require(node[14] > 0, 'empty node code')
        _slice(data['code'], node[13], node[14], name + ' code')
        code_ranges.append((node[13], node[14]))
    _disjoint(parameter_ranges, 'parameter segments')
    _disjoint(code_ranges, 'code segments')

    fc = nodes[-1]
    _require(fc[19:22] == (6, 8, 13) and fc[23:28] == (2, 6, 1, 1, 1),
             'wrong FC bias shape/quantization')
    _require(fc[29:32] == (6 * 2304, 8, 10) and
             fc[33:40] == (2, 6, 96, 4, 6, 0, 0),
             'wrong FC weight shape/quantization')
    _require(fc[28] == fc[18] + 6 and fc[28] + fc[29] == len(data['parameter']),
             'FC parameters must be the contiguous final bias/weight segments')
    bias = _slice(data['parameter'], fc[18], fc[19], 'FC bias')
    weights = _slice(data['parameter'], fc[28], fc[29], 'FC weights')
    _require(bias[5] == 0, 'sixth bias is not zero padding')
    _require(not any(weights[5 * 2304:]), 'sixth weight row is not zero padding')
    return {
        'schema': 1,
        'contract': 'soundjie-alert4bg-v2',
        'elf_sha256': hashlib.sha256(raw).hexdigest(),
        'logical_output_count': 5,
        'storage_output_count': 6,
        'input': _tensor_details(tensors[0]),
        'output': _tensor_details(tensors[-1]),
        'padding': {'index': 5, 'bias_zero': True, 'weights_zero': True, 'weight_bytes': 2304},
        'fc8': {'bias_offset': fc[18], 'bias_bytes': fc[19], 'bias_fix_pos': fc[21],
                'weights_offset': fc[28], 'weights_bytes': fc[29], 'weights_fix_pos': fc[31],
                'semantic_bias_sha256': hashlib.sha256(bias[:5]).hexdigest(),
                'semantic_weights_sha256': hashlib.sha256(weights[:5 * 2304]).hexdigest()},
    }
