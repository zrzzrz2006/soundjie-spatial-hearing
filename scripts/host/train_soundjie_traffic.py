"""Small PS multi-label refinement. Demo calibration is explicitly not unseen data."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from scipy.signal import butter, sosfilt
import torch
from scripts.host.prepare_soundjie_d3 import decode_audio, resample_mono_16k

NAMES=['ambulance_siren','traffic_noise','speech']

def demo_label_policy():
    return dict(basis='user_scene_weak_not_source_verified',
        source='user identifies traffic and ambulance background in the selected clip',
        siren_source_independently_verified=False,framewise_audition_performed=False,
        interval_seconds=[0,30],assumes_ambulance_and_traffic_throughout=True,
        metric_interpretation='agreement with supplied weak scene labels, not independent event ground truth')

def feature(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=1 or len(x)<512 or not np.isfinite(x).all():raise ValueError('invalid PCM')
    x=np.clip(np.rint(x*32768),-32768,32767)/32768
    frames=np.lib.stride_tricks.sliding_window_view(x,512)[::320]
    power=abs(np.fft.rfft(frames*np.hanning(512),axis=1))**2
    left=3;bands=[]
    for j in range(32):
        right=max(left+1,int(np.floor(3*(224/3)**((j+1)/32)+.5)))
        bands.append(np.log(np.maximum(power[:,left:right].mean(1),1e-12)));left=right
    f=np.stack(bands,1).astype(np.float32).astype(np.float64)
    mean=f.mean(0)
    return np.r_[mean-mean.mean(),f.std(0),abs(np.diff(f,axis=0)).mean(0) if len(f)>1 else np.zeros(32)].astype(np.float32)

def masked_loss(logits,y,mask,weight=None):
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none',pos_weight=weight)
    return (loss*mask).sum()/mask.sum().clamp_min(1)

def audit_groups(rows):
    groups={}
    for r in rows:
        g=r['source_group'];p=r['purpose']
        if p not in ('train','validation') or (g.startswith('demo:') and p!='train'):raise ValueError('invalid evaluation purpose')
        if g in groups and groups[g]!=p:raise ValueError('source group crossed split')
        groups[g]=p

def perturb(x,rng):
    x=x.copy()*rng.uniform(.3,1.1)
    if rng.random()<.8:
        delay=int(rng.integers(150,1400));x[delay:]+=rng.uniform(.08,.4)*x[:-delay]
    if rng.random()<.7:
        x=sosfilt(butter(2,[rng.uniform(80,250),rng.uniform(3200,6500)],btype='bandpass',fs=16000,output='sos'),x)
    level=max(1e-6,float(np.sqrt(np.mean(x*x))))
    x+=rng.normal(0,level*10**(-rng.uniform(15,35)/20),len(x))
    return x

def metrics(prob,y,mask):
    out={}
    for k,name in enumerate(NAMES):
        keep=mask[:,k]>0;pred=prob[keep,k]>=.65;truth=y[keep,k]>.5
        tp=int(np.sum(pred&truth));fp=int(np.sum(pred&~truth));fn=int(np.sum(~pred&truth));tn=int(np.sum(~pred&~truth))
        out[name]=dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn))
    return out

def export_model(path,mean,scale,model):
    arrays=[mean,scale,model[0].weight.detach().numpy(),model[0].bias.detach().numpy(),model[2].weight.detach().numpy(),model[2].bias.detach().numpy()]
    with Path(path).open('xb') as f:
        f.write(b'SJTF1\0\0\0'+struct.pack('<III',96,48,3))
        for a in arrays:f.write(np.asarray(a,dtype='<f4').tobytes())

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--demo',type=Path,required=True)
    p.add_argument('--cues',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=180)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);rng=np.random.default_rng(20261030);torch.manual_seed(20261030);torch.set_num_threads(4)
    rows=[json.loads(s) for s in a.manifest.read_text(encoding='utf-8').splitlines()];audit_groups(rows)
    feats=[];ys=[];masks=[];purposes=[];provenance=[]
    def add(x,y,mask,purpose,source,start,variants):
        for j in range(variants+1):
            z=x if j==0 else perturb(x,rng)
            feats.append(feature(z));ys.append(y);masks.append(mask);purposes.append(purpose)
            provenance.append(dict(source=source,start_sample=start,variant=j,purpose=purpose,
                label_basis=demo_label_policy()['basis'] if purpose=='demo_calibration' else 'public_source_manifest'))
    for i,r in enumerate(rows):
        path=Path(r['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('source hash mismatch: '+str(path))
        x=resample_mono_16k(decode_audio(path));x=np.pad(x,(0,max(0,16000-len(x))))
        starts=np.unique(np.linspace(0,len(x)-16000,6 if r['purpose']=='train' else 3).astype(int))
        for start in starts:
            z=x[start:start+16000]
            if np.sqrt(np.mean(z*z))<.0001:continue
            add(z,r['labels'],r['label_mask'],r['purpose'],r['source_group'],int(start),2 if r['purpose']=='train' else 0)
        if i%25==0:print('PREPARED',i+1,len(rows),flush=True)
    x=resample_mono_16k(decode_audio(a.demo));cfg=json.loads(a.cues.read_text(encoding='utf-8'))
    # All target-video windows belong to demo calibration, never validation.
    for start in range(0,len(x)-16000+1,8000):
        t=start/16000;overlap=sum(max(0,min(t+1,c['end'])-max(t,c['start'])) for c in cfg['cues'])
        speech=1 if overlap>=.5 else 0;mask=[1,1,1 if overlap>=.5 or overlap==0 else 0]
        add(x[start:start+16000],[1,1,speech],mask,'demo_calibration','demo:'+cfg['clip_id'],start,10)
    X=np.stack(feats);Y=np.array(ys,np.float32);M=np.array(masks,np.float32);P=np.array(purposes)
    train=P!='validation';val=P=='validation';demo=P=='demo_calibration';plain=np.array([r['variant']==0 for r in provenance])
    mean=X[train].mean(0);scale=X[train].std(0).clip(.15);Z=(X-mean)/scale
    model=torch.nn.Sequential(torch.nn.Linear(96,48),torch.nn.ReLU(),torch.nn.Linear(48,3))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.002)
    positives=(Y[train]*M[train]).sum(0);negatives=((1-Y[train])*M[train]).sum(0)
    weight=torch.tensor(np.clip(negatives/np.maximum(1,positives),1,6),dtype=torch.float32)
    tensors=[torch.from_numpy(q) for q in (Z,Y,M)];best=None;best_loss=float('inf');history=[]
    for epoch in range(a.epochs):
        indices=rng.permutation(np.flatnonzero(train));model.train();loss_total=0
        for start in range(0,len(indices),128):
            idx=indices[start:start+128];optimizer.zero_grad();loss=masked_loss(model(tensors[0][idx]),tensors[1][idx],tensors[2][idx],weight)
            loss.backward();optimizer.step();loss_total+=float(loss.detach())*len(idx)
        model.eval()
        with torch.no_grad():v=float(masked_loss(model(tensors[0][val]),tensors[1][val],tensors[2][val]))
        history.append(dict(epoch=epoch+1,train_loss=loss_total/len(indices),validation_loss=v))
        if v<best_loss:best_loss=v;best=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        if (epoch+1)%30==0:print('EPOCH',epoch+1,'VAL_LOSS',v,flush=True)
    model.load_state_dict(best);model.eval()
    with torch.no_grad():prob=torch.sigmoid(model(tensors[0])).numpy()
    export_model(a.output/'traffic.bin',mean,scale,model)
    np.savez_compressed(a.output/'features.npz',x=X,y=Y,mask=M,purpose=P,prediction=prob,mean=mean,scale=scale)
    report=dict(schema=1,names=NAMES,architecture='PS float32 96-48-3 MLP independent sigmoid, no DPU model replacement',thresholds=[.65]*3,best_epoch=best_epoch,
        source_records=len(rows),windows={s:int(np.sum(P==s)) for s in set(P)},source_group_separated=True,
        unseen_acceptance=False,demo_label_policy=demo_label_policy(),limits=['weak uploader-title ambulance labels; not siren-country invariance','user video included in demo calibration','digital perturbations are not loudspeaker/microphone or board evidence'],
        validation=metrics(prob[val],Y[val],M[val]),demo_direct=metrics(prob[demo&plain],Y[demo&plain],M[demo&plain]),demo_augmented=metrics(prob[demo&~plain],Y[demo&~plain],M[demo&~plain]),
        manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),demo_sha256=hashlib.sha256(a.demo.read_bytes()).hexdigest(),model_sha256=hashlib.sha256((a.output/'traffic.bin').read_bytes()).hexdigest())
    for name,obj in [('report.json',report),('history.json',history),('provenance.json',provenance)]:
        (a.output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    torch.save(dict(state=best,mean=mean,scale=scale),a.output/'checkpoint.pt')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
