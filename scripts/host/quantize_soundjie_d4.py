"""D4 train-only calibration staging and isolated legacy compiler execution."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from scripts.host import audit_soundjie_d3_model as audit
from scripts.host.export_soundjie_d4 import select_parity

ROOT = Path(__file__).resolve().parents[2]


def select_calibration(records):
    chosen, groups = [], set()
    for target in range(8):
        indices = []
        for i, row in enumerate(records):
            if (row['purpose'] == 'train' and row['target_index'] == target
                    and row['source_group'] not in groups):
                indices.append(i)
                groups.add(row['source_group'])
                if len(indices) == 8:
                    break
        if len(indices) != 8:
            raise ValueError('eight distinct train sources required per class')
        chosen.extend(indices)
    return chosen


def verify_files(root, bindings):
    root = Path(root).resolve()
    for name, expected in bindings.items():
        path = root / name
        if Path(name).name != name or path.resolve().parent != root:
            raise ValueError('unsafe artifact name')
        if audit.sha256_file(path) != expected:
            raise ValueError('artifact hash drift: ' + name)


def require_arm32(data):
    if len(data) < 52 or data[:6] != b'\x7fELF\x01\x01' or data[18:20] != b'\x28\x00':
        raise ValueError('expected little-endian ARM32 ELF')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stage(features, exported, caffe, output):
    if output.exists():
        raise ValueError('output already exists')
    fused = read_json(exported/'fused-manifest.json')
    verify_files(exported, fused['artifacts'])
    upstream = fused['upstream']
    if read_json(features/'audit.json') != upstream:
        raise ValueError('feature audit drift')
    verify_files(features, {'features.npz': upstream['cache_sha256'],
                           'provenance.json': upstream['provenance_sha256']})
    c = read_json(caffe/'caffe-export-manifest.json')
    verify_files(caffe, {'soundjie_d3_float32.caffemodel': c['caffemodel_sha256'],
                        'caffe-logits.npy': c['caffe_logits_sha256']})
    for key, name in [('weights_sha256','fused-weights.npz'), ('inputs_sha256','parity-inputs.npy'),
                      ('reference_logits_sha256','pytorch-logits.npy'), ('deploy_sha256','deploy.prototxt')]:
        if c[key] != fused['artifacts'][name]:
            raise ValueError('Caffe upstream drift')
    from scripts.guest.export_soundjie_d3_caffe import assess_parity
    assess_parity(np.load(exported/'pytorch-logits.npy'), np.load(caffe/'caffe-logits.npy'))
    with np.load(features/'features.npz', allow_pickle=False) as arrays:
        x, y, purpose = arrays['x'], arrays['y'], arrays['purpose']
    records = read_json(features/'provenance.json')
    parity = select_parity(x, y, purpose, records)
    if parity != fused['parity_indices'] or [records[i] for i in parity] != fused['parity_sources']:
        raise ValueError('validation reference drift')
    if not np.array_equal(x[parity], np.load(exported/'parity-inputs.npy')):
        raise ValueError('validation input drift')
    selected = select_calibration(records)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('deploy.prototxt', 'parity-inputs.npy', 'pytorch-logits.npy', 'fused-manifest.json'):
        shutil.copyfile(str(exported/name), str(output/name))
    for name in ('soundjie_d3_float32.caffemodel', 'caffe-export-manifest.json'):
        shutil.copyfile(str(caffe/name), str(output/name))
    for name, data in [('calibration.npy', x[selected]), ('calibration-targets.npy', y[selected])]:
        with (output/name).open('xb') as stream:
            np.save(stream, data, allow_pickle=False)
    audit.publish_manifest(output/'validation-targets.json',
        dict(parity_target_indices=[int(y[i]) for i in parity], purpose='validation'))
    report = dict(schema=1, checkpoint_sha256=fused['checkpoint_sha256'],
        feature_audit=upstream, calibration_indices=selected,
        calibration_sources=[records[i] for i in selected],
        parity_purpose='validation', independent_acceptance=False,
        artifacts={p.name:audit.sha256_file(p) for p in sorted(output.iterdir())})
    audit.publish_manifest(output/'quantization-input-manifest.json', report)
    return report


def build(input_dir, output, compiler):
    """Run ONLY inside the existing isolated Caffe container, on native ext4."""
    report = read_json(input_dir/'quantization-input-manifest.json')
    verify_files(input_dir, report['artifacts'])
    sources = report['calibration_sources']
    if (len(sources) != 64 or any(r['purpose'] != 'train' for r in sources)
            or len({r['source_group'] for r in sources}) != 64):
        raise ValueError('calibration source contract drift')
    data = np.load(input_dir/'calibration.npy', allow_pickle=False)
    labels = np.load(input_dir/'calibration-targets.npy', allow_pickle=False)
    import hashlib
    if data.shape != (64,1,64,96) or labels.shape != (64,):
        raise ValueError('calibration shape drift')
    for i, row in enumerate(sources):
        if int(labels[i]) != row['target_index'] or hashlib.sha256(data[i].tobytes()).hexdigest() != row['feature_sha256']:
            raise ValueError('calibration tensor/source mismatch')
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = str(compiler.parent.parent/'lib') + ':' + env.get('LD_LIBRARY_PATH','')
    commands = []

    def run(argv, name):
        commands.append(argv)
        with (output/name).open('xb') as log:
            subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    run([str(compiler),'--version'], 'compiler-version.txt')
    version = (output/'compiler-version.txt').read_text()
    if 'dnnc version v2.05' not in version or 'DPU Target : v1.4.0' not in version:
        raise ValueError('unexpected compiler version/target')
    auditor = str(ROOT/'scripts/host/audit_soundjie_d3_model.py')
    run([sys.executable,auditor,'make-assets','--input-root',str(input_dir),
         '--work-dir',str(output)], 'assets.log')
    run(['vai_q_caffe','quantize','-model',str(output/'quantize.prototxt'),
         '-weights',str(input_dir/'soundjie_d3_float32.caffemodel'),
         '-output_dir',str(output/'quantized'),'-calib_iter','64'], 'quantize.log')
    runtime = output/'quantized/runtime-deploy.prototxt'
    runtime.write_text(audit.make_runtime_prototxt((output/'quantized/deploy.prototxt').read_text()))
    run([sys.executable,auditor,'evaluate','--prototxt',str(runtime),
         '--caffemodel',str(output/'quantized/deploy.caffemodel'),
         '--inputs',str(input_dir/'parity-inputs.npy'),
         '--reference-logits',str(input_dir/'pytorch-logits.npy'),
         '--targets-manifest',str(input_dir/'validation-targets.json'),
         '--output',str(output/'quantization-metrics.json')], 'evaluate.log')
    (output/'compiled').mkdir()
    run([str(compiler),'--parser','caffe','--prototxt',str(runtime),
         '--caffemodel',str(output/'quantized/deploy.caffemodel'),'--dpu','1152FA',
         '--cpu_arch','arm32','--abi','1','--mode','normal','--output_dir',str(output/'compiled'),
         '--net_name','soundjie_d3'], 'compile.log')
    elfs = list((output/'compiled').glob('dpu_*.elf'))
    if len(elfs) != 1:
        raise ValueError('expected single DPU ELF')
    data = elfs[0].read_bytes()
    require_arm32(data)
    contract = audit.load_compiler_contract(ROOT/'config/soundjie_d3_compile.json')
    text = (output/'compile.log').read_text()
    evidence = audit.extract_compile_evidence(contract, text, data, text, runtime.read_text())
    audit.validate_compile_evidence(contract, evidence)
    audit.publish_manifest(output/'commands.json', commands)
    result = dict(schema=1, status='host_compiled_not_board_tested', board_tested=False,
        independent_acceptance=False, input_manifest_sha256=audit.sha256_file(input_dir/'quantization-input-manifest.json'),
        checkpoint_sha256=report['checkpoint_sha256'], compile_evidence=evidence,
        compiler_sha256=audit.sha256_file(compiler), quantization_metrics=read_json(output/'quantization-metrics.json'),
        artifacts={str(p.relative_to(output)):audit.sha256_file(p)
                   for p in sorted(output.rglob('*')) if p.is_file()})
    audit.publish_manifest(output/'d4-compiled-manifest.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command')
    s = sub.add_parser('stage')
    for name in ('features','exported','caffe','output'):
        s.add_argument('--'+name, required=True, type=Path)
    b = sub.add_parser('build')
    for name in ('input','output','compiler'):
        b.add_argument('--'+name, required=True, type=Path)
    a = p.parse_args()
    if a.command == 'stage':
        stage(a.features,a.exported,a.caffe,a.output)
    elif a.command == 'build':
        build(a.input,a.output,a.compiler)
    else:
        p.error('command required')


if __name__ == '__main__':
    main()
