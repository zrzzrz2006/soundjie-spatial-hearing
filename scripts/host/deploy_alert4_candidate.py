"""Create-only physical-five alert4bg conversion; no board, audio or D5 access.

stage/assess run in the Windows host environment; build/fixed-eval run in
the pinned legacy Caffe image. Completion manifests are always written last.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess

import numpy as np
from scripts.host import audit_soundjie_d3_model as audit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = 'soundjie-alert4bg-v2'
LABELS = ['siren', 'car_horn', 'doorbell', 'dog_bark', 'background_unknown']
ARTIFACT_ROOT = Path('F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_alert4bg/20261031a')
IMAGE = 'sha256:798f6eaea3896e2ad08aa7651a589ffaff70a052a677dfe956ecb0abdc1a55fd'
IDENTITY_FIELDS = ('recording_id', 'sampling_identity', 'source_group',
                   'uploader_group', 'parent_sha256', 'pcm_sha256_16k')
EXPORT_FILES = {'fused-weights.npz', 'parity-inputs.npy', 'pytorch-logits.npy',
                'calibration-train.npy', 'deploy.prototxt', 'labels.txt', 'thresholds.json'}
STAGED_FILES = EXPORT_FILES | {'export-manifest.json', 'training-result.json',
    'cache-audit.json', 'class-map.json', 'frontend.json', 'training-config.json',
    'source-manifest.jsonl', 'validation-inputs.npy', 'validation-targets.npy',
    'validation-float.npy', 'validation-sources.json'}
SHAPES = {}
for _layer, _inc, _outc in [('conv1', 1, 16), ('conv2', 16, 32),
                           ('conv3', 32, 64), ('conv4', 64, 96)]:
    SHAPES[_layer + '.weight'] = (_outc, _inc, 3, 3)
    SHAPES[_layer + '.bias'] = (_outc,)
SHAPES.update({'fc8.weight': (5, 2304), 'fc8.bias': (5,)})
sha = audit.sha256_file


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def save(path, value):
    with Path(path).open('xb') as stream:
        np.save(stream, value, allow_pickle=False)


def text_file(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(value)


def copy_file(source, dest):
    with Path(source).open('rb') as src, Path(dest).open('xb') as dst:
        shutil.copyfileobj(src, dst)


def require_hash(path, expected):
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('missing/invalid SHA256: ' + str(path))
    if sha(path) != expected:
        raise ValueError('hash drift: ' + str(path))


def check_files(root, pins, required=()):
    root = Path(root).resolve()
    if not isinstance(pins, dict) or not pins or not set(required) <= set(pins):
        raise ValueError('incomplete artifact pins')
    for name, expected in pins.items():
        rel = PurePosixPath(name)
        if (not name or '\\' in name or ':' in name or rel.is_absolute()
                or '..' in rel.parts or str(rel) != name):
            raise ValueError('unsafe artifact name: ' + name)
        path = root / name
        try:
            path.resolve().relative_to(root)
        except ValueError:
            raise ValueError('artifact escapes root: ' + name)
        if not path.is_file():
            raise ValueError('missing artifact: ' + name)
        require_hash(path, expected)


def file_pins(root):
    return {p.relative_to(root).as_posix(): sha(p)
            for p in sorted(root.rglob('*')) if p.is_file()}


def check_weights(arrays):
    if set(arrays) != set(SHAPES):
        raise ValueError('weight names differ')
    for name, shape in SHAPES.items():
        w = arrays[name]
        if w.shape != shape or w.dtype != np.float32 or not np.isfinite(w).all():
            raise ValueError('weight shape/type/value differs: ' + name)


def check_samples(x, rows, purpose):
    if (x.shape != (len(rows), 1, 64, 96) or not len(rows)
            or x.dtype != np.float32 or not np.isfinite(x).all()):
        raise ValueError('feature shape/type/value differs')
    for tensor, row in zip(x, rows):
        if (row['purpose'] != purpose or row['derivation'] is not None
                or type(row['target_index']) is not int or not 0 <= row['target_index'] < 5
                or sha_bytes(tensor.tobytes()) != row['feature_sha256']):
            raise ValueError('feature provenance differs')
        for field in IDENTITY_FIELDS:
            if field != 'uploader_group' and not row.get(field):
                raise ValueError('missing identity: ' + field)
        for field in ('parent_sha256', 'pcm_sha256_16k', 'feature_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', row[field]):
                raise ValueError('invalid provenance hash: ' + field)


def check_calibration(x, rows):
    check_samples(x, rows, 'train')
    if (len(rows) != 160 or len({r['sampling_identity'] for r in rows}) != 160
            or Counter(r['target_index'] for r in rows) != {i: 32 for i in range(5)}):
        raise ValueError('calibration requires 32 distinct train originals per class')


def check_validation(x, y, rows):
    check_samples(x, rows, 'validation')
    if (y.shape != (len(rows),) or not np.issubdtype(y.dtype, np.integer)
            or not np.array_equal(y, [r['target_index'] for r in rows])
            or set(y.tolist()) != set(range(5))
            or len({r['recording_id'] for r in rows}) != len(rows)):
        raise ValueError('validation targets/sources differ')


def check_disjoint(train, validation):
    for field in IDENTITY_FIELDS:
        left = {r[field] for r in train if r.get(field)}
        right = {r[field] for r in validation if r.get(field)}
        if left & right:
            raise ValueError('train/validation identity overlap: ' + field)


def check_source_binding(rows, sources):
    # Metadata only. Never open source audio paths, historical D5, or live replay.
    from scripts.host.soundjie_alert4bg import classify_fsd, map_existing
    by_id = {r['recording_id']: r for r in sources}
    if len(by_id) != len(sources):
        raise ValueError('duplicate source recording')
    for source in sources:
        if source['purpose'] not in ('train', 'validation') or source['class_contract'] != CONTRACT:
            raise ValueError('forbidden source purpose/contract')
        if source['source_id'] == 'fsd50k':
            choice = classify_fsd(source['original_labels'])
        else:
            mapped = map_existing(source)
            choice = None if mapped is None else (mapped['target_index'], mapped['negative_subtype'])
        if (choice != (source['target_index'], source['negative_subtype'])
                or source['target_id'] != LABELS[source['target_index']]):
            raise ValueError('source semantic class mismatch')
    for row in rows:
        source = by_id.get(row['recording_id'])
        if source is None:
            raise ValueError('missing source recording')
        fields = ('purpose', 'target_index', 'negative_subtype', 'source_group',
                  'sampling_identity', 'uploader_group')
        if (any(row.get(k) != source.get(k) for k in fields)
                or row['parent_sha256'] != source['sha256']):
            raise ValueError('source/feature provenance mismatch')


def check_export(meta):
    fold = meta.get('folding', {})
    if (meta.get('schema') != 2 or meta.get('contract') != CONTRACT
            or meta.get('status') != 'verified_fp32_export_ready_for_separate_DNNDK_ARM_stage'
            or meta.get('output_shape') != [1, 5] or meta.get('output_tensor_name') != 'fc8'
            or meta.get('checkpoint_reload_parity') is not True
            or fold.get('top1_identical') is not True
            or not 0 <= fold.get('max_absolute_error', float('inf')) <= 1e-4
            or meta.get('calibration_train_only') is not True
            or meta.get('calibration_counts') != [32] * 5
            or set(meta.get('files', {})) != EXPORT_FILES):
        raise ValueError('incomplete/wrong physical-five export')
    expected = dict(contract=CONTRACT, physical_output_count=5, logical_output_count=5,
                    physical_labels=LABELS, display_labels=LABELS, background_index=4,
                    probability_merge_groups=[[i] for i in range(5)], output_tensor_name='fc8')
    if meta.get('output_contract') != expected:
        raise ValueError('legacy or merged class contract is forbidden')


def probabilities(logits):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 5 or not len(z) or not np.isfinite(z).all():
        raise ValueError('finite physical-five logits required')
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def parity(reference, candidate, tolerance=1e-4):
    probabilities(reference)
    probabilities(candidate)
    if reference.shape != candidate.shape:
        raise ValueError('parity shape differs')
    error = float(np.max(np.abs(reference - candidate)))
    matches = int(np.sum(reference.argmax(1) == candidate.argmax(1)))
    if error > tolerance or matches != len(reference):
        raise ValueError('Caffe parity failed: error=%g top1=%d/%d' % (error, matches, len(reference)))
    return dict(max_absolute_error=error, top1_matches=matches, samples=len(reference), tolerance=tolerance)


def staged_inputs(directory, manifest=None):
    m = read(directory / 'input-manifest.json') if manifest is None else manifest
    if m.get('contract') != CONTRACT or m.get('status') != 'verified_inputs_complete':
        raise ValueError('incomplete staged manifest')
    check_files(directory, m['artifacts'], STAGED_FILES)
    meta = read(directory / 'export-manifest.json')
    check_export(meta)
    require_hash(directory / 'export-manifest.json', m['export_manifest_sha256'])
    if m['checkpoint_sha256'] != meta['checkpoint_sha256']:
        raise ValueError('checkpoint chain drift')
    check_files(directory, meta['files'], EXPORT_FILES)
    if (directory / 'labels.txt').read_text(encoding='utf-8').splitlines() != LABELS:
        raise ValueError('label order differs')
    for filename, key in [('class-map.json', 'class_map_sha256'), ('frontend.json', 'frontend_sha256'),
                          ('source-manifest.jsonl', 'source_manifest_sha256')]:
        require_hash(directory / filename, meta[key])
    class_map = read(directory / 'class-map.json')
    if (class_map['contract'] != CONTRACT or class_map['labels'] != LABELS
            or class_map['physical_output_count'] != 5 or class_map['background_index'] != 4):
        raise ValueError('class map differs')
    with np.load(directory / 'fused-weights.npz', allow_pickle=False) as data:
        check_weights({k: data[k] for k in data.files})
    cal = np.load(directory / 'calibration-train.npy', allow_pickle=False)
    check_calibration(cal, meta['calibration_sources'])
    vx = np.load(directory / 'validation-inputs.npy', allow_pickle=False)
    y = np.load(directory / 'validation-targets.npy', allow_pickle=False)
    rows = read(directory / 'validation-sources.json')
    check_validation(vx, y, rows)
    check_disjoint(meta['calibration_sources'], rows)
    px = np.load(directory / 'parity-inputs.npy', allow_pickle=False)
    check_samples(px, meta['parity_sources'], 'validation')
    if len(px) != 10 or Counter(r['target_index'] for r in meta['parity_sources']) != {i: 2 for i in range(5)}:
        raise ValueError('parity requires two validation sources per class')
    for c in range(5):
        if len({r['source_group'] for r in meta['parity_sources'] if r['target_index'] == c}) != 2:
            raise ValueError('parity source groups repeat')
    for row in meta['parity_sources']:
        if row not in rows:
            raise ValueError('parity source absent from validation')
    for filename, count in [('pytorch-logits.npy', len(px)), ('validation-float.npy', len(vx))]:
        z = np.load(directory / filename, allow_pickle=False)
        probabilities(z)
        if z.shape != (count, 5):
            raise ValueError('reference logit length differs')
    if audit._network_output_count((directory / 'deploy.prototxt').read_text(), 'fc8') != 5:
        raise ValueError('Caffe output count differs')
    return m, meta


def stage(exported, training, output, export_sha256, training_sha256):
    """Require completed, externally pinned manifests before reading model/cache."""
    from scripts.host.soundjie_alert4bg import CLASS_FILE, CONFIG_FILE, DATA_ROOT, make_model, validate_rows
    from scripts.host.train_soundjie_alert4bg import load_cache
    from scripts.host.train_soundjie_alert6 import predict
    import torch
    if output.exists():
        raise FileExistsError(output)
    for directory in (exported, training):
        directory.resolve().relative_to(ARTIFACT_ROOT.resolve())
    require_hash(exported / 'export-manifest.json', export_sha256)
    require_hash(training / 'result.json', training_sha256)
    meta = read(exported / 'export-manifest.json')
    check_export(meta)
    check_files(exported, meta['files'], EXPORT_FILES)
    result = read(training / 'result.json')
    if result['contract'] != CONTRACT or result['d5_audio_read'] or result['d5_features_read'] or result['selection_used_replay']:
        raise ValueError('training purpose/contract differs')
    selection = result['artifacts']['selected']
    selected = Path(selection['directory'])
    selected.resolve().relative_to(training.resolve())
    checkpoint = selected / 'model.pt'
    if (selection['checkpoint_sha256'] != meta['checkpoint_sha256']
            or Path(meta['checkpoint_path']).resolve() != checkpoint.resolve()):
        raise ValueError('selected/export checkpoint differs')
    require_hash(checkpoint, meta['checkpoint_sha256'])
    require_hash(selected / 'validation-logits.npz', selection['logits_sha256'])
    require_hash(CLASS_FILE, meta['class_map_sha256'])
    require_hash(CONFIG_FILE, result['config_sha256'])
    require_hash(ROOT / 'scripts/host/export_soundjie_alert4bg.py', meta['export_script_sha256'])
    require_hash(ROOT / 'scripts/host/export_soundjie_siren_merge.py', meta['fold_helper_sha256'])
    features = Path(result['features_directory'])
    x, y, purpose, rows, cache = load_cache(features)
    if (cache['cache_sha256'] != meta['cache_sha256'] or result['cache_sha256'] != meta['cache_sha256']
            or cache['frontend_sha256'] != meta['frontend_sha256']
            or cache['manifest_sha256'] != meta['source_manifest_sha256']
            or cache['config'] != result['config']):
        raise ValueError('export/training/cache chain differs')
    for name, expected in result['implementation_sha256'].items():
        check_files(ROOT, {name.replace('\\', '/'): expected})
    for name, expected in cache['implementation_sha256'].items():
        check_files(ROOT, {name.replace('\\', '/'): expected})
    source_path = DATA_ROOT / 'manifest.jsonl'
    if Path(cache['manifest_path']).resolve() != source_path.resolve():
        raise ValueError('unexpected source manifest path')
    require_hash(source_path, meta['source_manifest_sha256'])
    sources = [json.loads(line) for line in source_path.read_text(encoding='utf-8').splitlines() if line]
    validate_rows(sources)
    check_source_binding(rows, sources)
    val_rows = [r for r in rows if r['purpose'] == 'validation']
    check_disjoint([r for r in rows if r['purpose'] == 'train'], val_rows)
    for row in meta['calibration_sources'] + meta['parity_sources']:
        if row not in rows:
            raise ValueError('export source absent from verified feature cache')
    va = purpose == 'validation'
    vx, vy = x[va], y[va]
    check_validation(vx, vy, val_rows)
    with np.load(selected / 'validation-logits.npz', allow_pickle=False) as data:
        reference, targets = data['logits'], data['y']
    if not np.array_equal(targets, vy):
        raise ValueError('training/export validation order differs')
    torch.set_num_threads(4)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (ckpt['contract'] != CONTRACT or ckpt['labels'] != LABELS
            or ckpt['class_map_sha256'] != meta['class_map_sha256'] or ckpt['cache_sha256'] != meta['cache_sha256']):
        raise ValueError('checkpoint metadata differs')
    model = make_model()
    model.load_state_dict(ckpt['state_dict'], strict=True)
    reloaded = predict(model, vx)
    reload_parity = parity(reference, reloaded, tolerance=1e-6)
    from scripts.host.soundjie_alert4bg import metrics, apply_thresholds
    thresholds = read(exported / 'thresholds.json')
    if (thresholds['contract'] != CONTRACT or thresholds['labels'] != LABELS
            or thresholds['checkpoint_sha256'] != meta['checkpoint_sha256']
            or thresholds['alert_thresholds'] != selection['metrics']['thresholds']
            or meta['validation_metrics'] != selection['metrics']
            or metrics(vy, apply_thresholds(probabilities(reference), thresholds['alert_thresholds']), val_rows)
               != selection['metrics']['thresholded']):
        raise ValueError('validation metric/threshold provenance differs')
    # Inputs are immutable snapshots: a failed stage never publishes a completion marker.
    output.mkdir(parents=True, exist_ok=False)
    for name in sorted(EXPORT_FILES | {'export-manifest.json'}):
        copy_file(exported / name, output / name)
    for source, name in [(training / 'result.json', 'training-result.json'),
                         (features / 'audit.json', 'cache-audit.json'), (CLASS_FILE, 'class-map.json'),
                         (CONFIG_FILE, 'training-config.json'), (source_path, 'source-manifest.jsonl'),
                         (ROOT / 'config/logmel_frontend_v1.json', 'frontend.json')]:
        copy_file(source, output / name)
    save(output / 'validation-inputs.npy', vx)
    save(output / 'validation-targets.npy', vy)
    save(output / 'validation-float.npy', reference)
    write(output / 'validation-sources.json', val_rows)
    # Recheck producer manifests after reading to catch concurrent producer changes.
    require_hash(exported / 'export-manifest.json', export_sha256)
    require_hash(training / 'result.json', training_sha256)
    pending = dict(schema=2, contract=CONTRACT,
          status='verified_inputs_complete', export_manifest_sha256=export_sha256,
          training_result_sha256=training_sha256, checkpoint_sha256=meta['checkpoint_sha256'],
          source_manifest_sha256=meta['source_manifest_sha256'], cache_sha256=meta['cache_sha256'],
          checkpoint_reload_parity=reload_parity, validation_count=len(vy),
          validation_counts=np.bincount(vy, minlength=5).tolist(), source_disjoint=True,
          d5_accessed=False, live_replay_used=False, independent_acceptance=False,
          implementation_sha256=sha(Path(__file__)), artifacts=file_pins(output))
    staged_inputs(output, pending)
    write(output / 'input-manifest.json', pending)


def forward(net, inputs):
    out = []
    for feature in inputs:
        net.blobs['data'].data[...] = feature[None, ...]
        net.forward()
        out.append(np.asarray(net.blobs['fc8'].data[0]).reshape(-1).copy())
    z = np.stack(out).astype(np.float32)
    probabilities(z)
    if z.shape != (len(inputs), 5):
        raise ValueError('Caffe physical-five output differs')
    return z


def compile_contract():
    contract = audit.load_compiler_contract(ROOT / 'config/soundjie_d3_compile.json')
    contract.update(accepted_dnnc_versions=['v2.05'], output_count=5)
    return contract


def build(input_dir, output, compiler):
    import caffe
    import h5py
    from scripts.host.quantize_soundjie_d4 import require_arm32
    manifest, meta = staged_inputs(input_dir)
    output.mkdir(parents=True, exist_ok=False)
    commands = []
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(compiler.parent.parent / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')

    def run(argv, name):
        commands.append(argv)
        with (output / name).open('xb') as stream:
            subprocess.run(argv, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    run([str(compiler), '--version'], 'compiler-version.txt')
    version = (output / 'compiler-version.txt').read_text()
    if 'dnnc version v2.05' not in version or 'DPU Target : v1.4.0' not in version:
        raise ValueError('compiler mismatch')
    caffe.set_mode_cpu()
    net = caffe.Net(str(input_dir / 'deploy.prototxt'), caffe.TEST)
    with np.load(input_dir / 'fused-weights.npz', allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    check_weights(arrays)
    if set(net.params) != {k.split('.')[0] for k in SHAPES}:
        raise ValueError('Caffe layers differ')
    for layer in net.params:
        if len(net.params[layer]) != 2:
            raise ValueError('Caffe parameter blobs differ')
        for blob, suffix in zip(net.params[layer], ('weight', 'bias')):
            w = arrays[layer + '.' + suffix]
            if blob.data.shape != w.shape:
                raise ValueError('Caffe weight shape differs')
            blob.data[...] = w
    checks = {}
    for name, ref_name in [('parity', 'pytorch-logits.npy'), ('validation', 'validation-float.npy')]:
        z = forward(net, np.load(input_dir / (name + '-inputs.npy'), allow_pickle=False))
        checks[name] = parity(np.load(input_dir / ref_name, allow_pickle=False), z)
        save(output / (name + '-caffe-float.npy'), z)
    write(output / 'caffe-parity.json', checks)
    model = output / 'float.caffemodel'
    net.save(str(model))
    cal = np.load(input_dir / 'calibration-train.npy', allow_pickle=False)
    check_calibration(cal, meta['calibration_sources'])
    hdf = output / 'calibration.h5'
    with h5py.File(str(hdf), 'w-') as stream:
        stream.create_dataset('data', data=cal)
        stream.create_dataset('label', data=np.array([r['target_index'] for r in meta['calibration_sources']], np.float32).reshape(-1, 1))
    listing = output / 'hdf5-list.txt'
    text_file(listing, str(hdf) + '\n')
    quantproto = output / 'quantize.prototxt'
    text_file(quantproto, audit.make_quantizer_prototxt((input_dir / 'deploy.prototxt').read_text(), listing))
    run(['vai_q_caffe', 'quantize', '-model', str(quantproto), '-weights', str(model),
         '-output_dir', str(output / 'quantized'), '-calib_iter', '160'], 'quantize.log')
    runtime = output / 'quantized/runtime-deploy.prototxt'
    text_file(runtime, audit.make_runtime_prototxt((output / 'quantized/deploy.prototxt').read_text()))
    quant = caffe.Net(str(runtime), str(output / 'quantized/deploy.caffemodel'), caffe.TEST)
    for name in ('parity', 'validation'):
        save(output / (name + '-quantized-weights.npy'), forward(quant, np.load(input_dir / (name + '-inputs.npy'), allow_pickle=False)))
    (output / 'compiled').mkdir()
    run([str(compiler), '--parser', 'caffe', '--prototxt', str(runtime),
         '--caffemodel', str(output / 'quantized/deploy.caffemodel'), '--dpu', '1152FA',
         '--cpu_arch', 'arm32', '--abi', '1', '--mode', 'normal',
         '--output_dir', str(output / 'compiled'), '--net_name', 'soundjie_d3'], 'compile.log')
    elfs = list((output / 'compiled').glob('dpu_*.elf'))
    if len(elfs) != 1 or elfs[0].name != 'dpu_soundjie_d3.elf':
        raise ValueError('expected one soundjie_d3 ELF')
    raw = elfs[0].read_bytes()
    require_arm32(raw)
    log = (output / 'compile.log').read_text()
    contract = compile_contract()
    evidence = audit.extract_compile_evidence(contract, log, raw, log, runtime.read_text())
    audit.validate_compile_evidence(contract, evidence)
    write(output / 'commands.json', commands)
    write(output / 'compiled-manifest.json', dict(schema=2, contract=CONTRACT, status='compiled_complete',
          checkpoint_sha256=manifest['checkpoint_sha256'], input_manifest_sha256=sha(input_dir / 'input-manifest.json'),
          compiler_sha256=sha(compiler), image=IMAGE, compile_evidence=evidence, caffe_parity=checks,
          calibration_samples=160, calibration_train_only=True, elf_sha256=sha(elfs[0]),
          board_tested=False, independent_acceptance=False, implementation_sha256=sha(Path(__file__)),
          artifacts=file_pins(output)))


def check_compiled(input_dir, compiled):
    from scripts.host.quantize_soundjie_d4 import require_arm32
    cm = read(compiled / 'compiled-manifest.json')
    if cm.get('contract') != CONTRACT or cm.get('status') != 'compiled_complete':
        raise ValueError('incomplete compiled manifest')
    require_hash(input_dir / 'input-manifest.json', cm['input_manifest_sha256'])
    check_files(compiled, cm['artifacts'], {'compiled/dpu_soundjie_d3.elf',
        'quantized/quantize_train_test.prototxt', 'quantized/quantize_train_test.caffemodel',
        'quantized/deploy.caffemodel', 'quantized/runtime-deploy.prototxt', 'compile.log'})
    require_hash(compiled / 'compiled/dpu_soundjie_d3.elf', cm['elf_sha256'])
    raw = (compiled / 'compiled/dpu_soundjie_d3.elf').read_bytes()
    require_arm32(raw)
    log = (compiled / 'compile.log').read_text()
    contract = compile_contract()
    evidence = audit.extract_compile_evidence(contract, log, raw, log,
                 (compiled / 'quantized/runtime-deploy.prototxt').read_text())
    audit.validate_compile_evidence(contract, evidence)
    if evidence != cm['compile_evidence']:
        raise ValueError('compiled evidence differs from ELF/log')
    return cm


def check_fixed_graph(text):
    if (len(re.findall(r'type\s*:\s*"FixedNeuron"', text)) != 6
            or 'type: "ConvolutionFixed"' not in text
            or audit._network_output_count(text, 'fc8') != 5):
        raise ValueError('physical-five FixedNeuron activation graph missing')


def fixed_eval(input_dir, compiled, output):
    """TEST uses learned fixed positions; evaluating validation never calibrates."""
    import caffe
    staged_inputs(input_dir)
    check_compiled(input_dir, compiled)
    text = (compiled / 'quantized/quantize_train_test.prototxt').read_text()
    check_fixed_graph(text)
    output.mkdir(parents=True, exist_ok=False)
    proto = output / 'fixed-evaluation.prototxt'
    text_file(proto, audit.make_runtime_prototxt(text))
    caffe.set_mode_cpu()
    net = caffe.Net(str(proto), str(compiled / 'quantized/quantize_train_test.caffemodel'), caffe.TEST)

    def params():
        return {name: [b.data.copy() for b in blobs] for name, blobs in net.params.items()}

    before = params()
    for name in ('parity', 'validation'):
        save(output / (name + '-fixed.npy'), forward(net, np.load(input_dir / (name + '-inputs.npy'), allow_pickle=False)))
    after = params()
    if (set(before) != set(after) or any(len(before[k]) != len(after[k]) or
            any(not np.array_equal(a, b) for a, b in zip(before[k], after[k])) for k in before)):
        raise ValueError('TEST changed learned parameters/quantization positions')
    positions = {name: [b.tolist() for b in blobs] for name, blobs in after.items() if 'fixed' in name.lower()}
    write(output / 'fixed-evaluation.json', dict(schema=2, contract=CONTRACT, status='fixed_evaluation_complete',
          scope='Caffe TEST FixedNeuron simulation; not physical DPU',
          input_manifest_sha256=sha(input_dir / 'input-manifest.json'),
          compiled_manifest_sha256=sha(compiled / 'compiled-manifest.json'),
          learned_parameters_unchanged=True, fixed_parameters=positions,
          board_tested=False, independent_acceptance=False, artifacts=file_pins(output)))


def assess_scores(y, rows, fixed_logits, float_thresholds, config):
    from scripts.host.soundjie_alert4bg import metrics, apply_thresholds, calibrate_thresholds
    scores = probabilities(fixed_logits)
    thresholds = calibrate_thresholds(y, scores, config)
    result = dict(raw=metrics(y, scores.argmax(1), rows),
                  original_fp32_thresholds=metrics(y, apply_thresholds(scores, float_thresholds), rows),
                  post_int8=metrics(y, apply_thresholds(scores, thresholds), rows),
                  post_int8_thresholds=thresholds, threshold_search=config,
                  thresholds_fit_on_same_development_validation=True,
                  scope='source-disjoint development validation reused for model selection and threshold tuning',
                  independent_acceptance=False, board_tested=False)
    pred = apply_thresholds(scores, thresholds)
    rates = [float(np.mean(pred[y == 4] == c)) for c in range(4)]
    result['per_alert_background_false_alert_rate'] = rates
    result['per_alert_background_bound_met'] = [r <= config['per_class_background_fpr_limit'] + 1e-12 for r in rates]
    return result


def assess(input_dir, compiled, fixed, output):
    from scripts.host.soundjie_alert4bg import metrics, apply_thresholds
    manifest, meta = staged_inputs(input_dir)
    cm = check_compiled(input_dir, compiled)
    fm = read(fixed / 'fixed-evaluation.json')
    if (fm.get('contract') != CONTRACT or fm.get('status') != 'fixed_evaluation_complete'
            or fm.get('learned_parameters_unchanged') is not True):
        raise ValueError('incomplete FixedNeuron evaluation')
    require_hash(input_dir / 'input-manifest.json', fm['input_manifest_sha256'])
    require_hash(compiled / 'compiled-manifest.json', fm['compiled_manifest_sha256'])
    check_files(fixed, fm['artifacts'], {'validation-fixed.npy', 'parity-fixed.npy', 'fixed-evaluation.prototxt'})
    y = np.load(input_dir / 'validation-targets.npy', allow_pickle=False)
    rows = read(input_dir / 'validation-sources.json')
    thresholds = read(input_dir / 'thresholds.json')['alert_thresholds']
    cfg = read(input_dir / 'training-config.json')['thresholds']
    z = np.load(fixed / 'validation-fixed.npy', allow_pickle=False)
    assessment = assess_scores(y, rows, z, thresholds, cfg)
    comparisons = {}
    for name, path in [('pytorch', input_dir / 'validation-float.npy'),
                       ('caffe_float', compiled / 'validation-caffe-float.npy'),
                       ('caffe_quantized_weights_only', compiled / 'validation-quantized-weights.npy')]:
        p = probabilities(np.load(path, allow_pickle=False))
        comparisons[name] = dict(raw=metrics(y, p.argmax(1), rows),
                                original_fp32_thresholds=metrics(y, apply_thresholds(p, thresholds), rows))
    a = comparisons['pytorch']['original_fp32_thresholds']
    b = assessment['post_int8']
    result = dict(schema=2, contract=CONTRACT, labels=LABELS, status='host_conversion_and_development_assessment_complete',
        checkpoint_sha256=meta['checkpoint_sha256'], export_manifest_sha256=manifest['export_manifest_sha256'],
        input_manifest_sha256=sha(input_dir / 'input-manifest.json'),
        compiled_manifest_sha256=sha(compiled / 'compiled-manifest.json'),
        fixed_manifest_sha256=sha(fixed / 'fixed-evaluation.json'), elf_sha256=cm['elf_sha256'],
        compiled_model_path=str(compiled / 'compiled/dpu_soundjie_d3.elf'),
        compile_evidence=cm['compile_evidence'], caffe_parity=cm['caffe_parity'],
        fp32_thresholds=thresholds, fixed_neuron=assessment, comparisons=comparisons,
        fixed_minus_float=dict(macro_recall=b['macro_recall'] - a['macro_recall'],
                              background_fpr=b['negative_false_alert_rate'] - a['negative_false_alert_rate']),
        board_tested=False, independent_acceptance=False, d5_accessed=False, live_replay_used=False,
        limits=['weak clip labels; energy windows are not manually annotated event boundaries',
                'same development validation used for model selection and thresholds',
                'FixedNeuron simulation is not board execution or acoustic validation'])
    write(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['stage', 'build', 'fixed-eval', 'assess'])
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training', type=Path)
    p.add_argument('--export-sha256')
    p.add_argument('--training-sha256')
    p.add_argument('--compiler', type=Path)
    p.add_argument('--compiled', type=Path)
    p.add_argument('--fixed', type=Path)
    a = p.parse_args()
    required = {'stage': ('training', 'export_sha256', 'training_sha256'),
                'build': ('compiler',), 'fixed-eval': ('compiled',), 'assess': ('compiled', 'fixed')}
    if any(getattr(a, key) is None for key in required[a.command]):
        p.error('missing arguments for ' + a.command)
    if a.command == 'stage':
        stage(a.input, a.training, a.output, a.export_sha256, a.training_sha256)
    elif a.command == 'build':
        build(a.input, a.output, a.compiler)
    elif a.command == 'fixed-eval':
        fixed_eval(a.input, a.compiled, a.output)
    else:
        result = assess(a.input, a.compiled, a.fixed, a.output)
        print(json.dumps(dict(elf_sha256=result['elf_sha256'], thresholds=result['fixed_neuron']['post_int8_thresholds'],
                             metrics=result['fixed_neuron']['post_int8'], board_tested=False)))


if __name__ == '__main__':
    main()
