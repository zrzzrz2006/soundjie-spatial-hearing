"""Known-clip acoustic enrollment; never a general ambulance/ASR acceptance test."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import torch
from scripts.host.train_soundjie_traffic import feature, perturb, masked_loss, metrics, export_model, demo_label_policy


def read_reference(path):
    b=Path(path).read_bytes()
    if len(b)<16 or b[:8]!=b'SJTVF2\0\0':raise ValueError('reference header')
    n,d=struct.unpack('<II',b[8:16])
    if d!=32 or not 297<=n<=24000 or len(b)!=16+n*d*4:raise ValueError('reference shape')
    f=np.frombuffer(b[16:],'<f4').reshape(n,d).astype(np.float64)
    if not np.isfinite(f).all() or np.max(abs(f))>100:raise ValueError('reference values')
    return f


def blended_reference(digital,acoustic):
    if digital.shape!=acoustic.shape:raise ValueError('unaligned reference shapes')
    x=digital-digital.mean(0);y=acoustic-acoustic.mean(0)
    nx=np.mean(x*x);ny=np.mean(y*y)
    if not np.isfinite([nx,ny]).all() or min(nx,ny)<1e-6:raise ValueError('degenerate reference')
    return (.5*x+.5*y*np.sqrt(nx/ny)).astype('<f4')


def load_network(path):
    b=Path(path).read_bytes()
    if len(b)!=20000 or b[:8]!=b'SJTF1\0\0\0' or struct.unpack('<III',b[8:20])!=(96,48,3):raise ValueError('model header/size')
    p=np.frombuffer(b[20:],'<f4').copy()
    if not np.isfinite(p).all() or np.max(abs(p))>10000 or min(p[96:192])<.00001:raise ValueError('model values')
    model=torch.nn.Sequential(torch.nn.Linear(96,48),torch.nn.ReLU(),torch.nn.Linear(48,3))
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(p[192:4800].reshape(48,96)))
        model[0].bias.copy_(torch.from_numpy(p[4800:4848]))
        model[2].weight.copy_(torch.from_numpy(p[4848:4992].reshape(3,48)))
        model[2].bias.copy_(torch.from_numpy(p[4992:4995]))
    return p[:96],p[96:192],model


def main():
    parser=argparse.ArgumentParser()
    for key in ('digital-reference','acoustic-reference','capture-pcm','base-model-dir','cues','analysis','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report_alignment=json.loads(args.analysis.read_text(encoding='utf-8'))
    raw=args.capture_pcm.read_bytes()
    if len(raw)!=960000 or hashlib.sha256(raw).hexdigest()!=report_alignment['clip_sha256']:raise ValueError('capture identity/count')
    reference=blended_reference(read_reference(args.digital_reference),read_reference(args.acoustic_reference))
    torch.set_num_threads(4);torch.manual_seed(20261031);rng=np.random.default_rng(20261031)
    mean,scale,model=load_network(args.base_model_dir/'traffic.bin')
    cached=np.load(args.base_model_dir/'features.npz',allow_pickle=False)
    X,Y,M,P=[cached[k] for k in ('x','y','mask','purpose')]
    cfg=json.loads(args.cues.read_text(encoding='utf-8'));pcm=np.frombuffer(raw,'<i2')/32768.
    ax=[];ay=[];am=[];plain=[];provenance=[]
    for start in range(0,len(pcm)-16000+1,8000):
        t=start/16000.; overlap=sum(max(0,min(t+1,c['end'])-max(t,c['start'])) for c in cfg['cues'])
        target=[1,1,int(overlap>=.5)];mask=[1,1,int(overlap==0 or overlap>=.5)]
        for variant in range(11):
            z=pcm[start:start+16000];z=z if variant==0 else perturb(z,rng)
            ax.append(feature(z));ay.append(target);am.append(mask);plain.append(variant==0)
            provenance.append(dict(source='board-replay-20261031a',start_sample=start,variant=variant,purpose='acoustic_demo_calibration',label_basis=demo_label_policy()['basis']))
    AX=np.asarray(ax,np.float32);AY=np.asarray(ay,np.float32);AM=np.asarray(am,np.float32);plain=np.asarray(plain,bool)
    def predict(x):
        model.eval()
        with torch.no_grad():return torch.sigmoid(model(torch.from_numpy(((x-mean)/scale).astype(np.float32)))).numpy()
    baseline=metrics(predict(AX[plain]),AY[plain],AM[plain])
    old_demo=P=='demo_calibration';old_train=P!='validation';val=P=='validation'
    # Rehearse original source groups and original clip while enrolling microphone audio.
    TX=np.concatenate([X[old_train],AX,AX]);TY=np.concatenate([Y[old_train],AY,AY]);TM=np.concatenate([M[old_train],AM,AM])
    tensors=[torch.from_numpy(z.astype(np.float32)) for z in ((TX-mean)/scale,TY,TM)]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.002)
    frozen=[p.detach().clone() for p in model.parameters()];history=[];accepted=False
    for epoch in range(1,121):
        model.train()
        for indices in np.array_split(rng.permutation(len(TX)),max(1,(len(TX)+127)//128)):
            optimizer.zero_grad();loss=masked_loss(model(tensors[0][indices]),tensors[1][indices],tensors[2][indices])
            loss=loss+.001*sum((p-q).square().mean() for p,q in zip(model.parameters(),frozen))
            loss.backward();optimizer.step()
        ap=predict(AX[plain]);dp=predict(X[old_demo])
        ametric=metrics(ap,AY[plain],AM[plain]);dmetric=metrics(dp,Y[old_demo],M[old_demo])
        gate=all(ametric[n]['recall']>=.98 and dmetric[n]['recall']>=.98 for n in ('ambulance_siren','traffic_noise'))
        history.append(dict(epoch=epoch,acoustic=ametric,digital=dmetric,calibration_gate=gate))
        if epoch%10==0 or gate:print('CALIBRATION',json.dumps(history[-1]),flush=True)
        if gate:accepted=True;break
    args.output.mkdir(parents=True)
    if not accepted:
        (args.output/'failed-history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        raise RuntimeError('calibration criteria not reached; no deployable data exported')
    with (args.output/'reference.features').open('xb') as f:
        f.write(b'SJTVF2\0\0'+struct.pack('<II',len(reference),32)+reference.tobytes())
    export_model(args.output/'traffic.bin',mean,scale,model)
    report=dict(schema=1,scope='known 30-second clip acoustic enrollment; requires NEW physical replay acceptance',
        independent_physical_acceptance=False,reference_method='equal energy-normalized centered feature mixture of original and first microphone replay',
        matcher_threshold_unchanged=.65,matcher_margin_unchanged=.05,labels=demo_label_policy(),
        baseline_acoustic_calibration=baseline,final_acoustic_calibration=ametric,original_demo_calibration=dmetric,
        public_holdout_diagnostic=metrics(predict(X[val]),Y[val],M[val]),epochs=epoch,
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.digital_reference,args.acoustic_reference,args.capture_pcm,args.base_model_dir/'traffic.bin',args.base_model_dir/'features.npz',args.cues,args.analysis)},
        output_sha256={n:hashlib.sha256((args.output/n).read_bytes()).hexdigest() for n in ('reference.features','traffic.bin')})
    for name,obj in [('report.json',report),('history.json',history),('provenance.json',provenance)]:
        (args.output/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    print('EXPORTED='+json.dumps(report),flush=True)


if __name__=='__main__':main()
