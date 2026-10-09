"""Resumable full-Imagenette matched controls. Independent of ephemeral runner state."""
import argparse, json, hashlib, urllib.request, tarfile, time, types, os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
P=ROOT/'vitreco_validation'
MODELS={'tiny':'deit_tiny_patch16_224.fb_in1k','small':'deit_small_patch16_224.fb_in1k','swin':'swin_tiny_patch4_window7_224.ms_in1k'}

def download(url,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        with urllib.request.urlopen(url,timeout=120) as src,open(str(p)+'.partial','wb') as dst:
            while chunk:=src.read(1024*1024):dst.write(chunk)
        Path(str(p)+'.partial').replace(p)

def main():
    import torch, timm, numpy as np, torch.nn.functional as F, zstandard as zs
    from safetensors.torch import load_file
    from PIL import Image
    a=argparse.ArgumentParser();a.add_argument('--model',choices=MODELS,required=True);a.add_argument('--seed',type=int,default=151);a.add_argument('--threads',type=int,default=2);a=a.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);timm.layers.set_fused_attn(False)
    torch.manual_seed(a.seed);name=MODELS[a.model];out=P/'results'/a.model/str(a.seed);out.mkdir(parents=True,exist_ok=True)
    archive=P/'data/imagenette2-160.tgz'
    if not (P/'data/READY').exists():
        download('https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',archive)
        assert hashlib.md5(archive.read_bytes()).hexdigest()=='e793b78cc4c9e9a4ccc0c1155377a412'
        with tarfile.open(archive) as t:t.extractall(P/'data',filter='data')
        (P/'data/READY').write_text('MD5 verified')
    for file in ['config.json','model.safetensors']:download(f'https://huggingface.co/timm/{name}/resolve/main/{file}',P/'models'/name/file)
    cfg=json.loads((P/'models'/name/'config.json').read_text());model=timm.create_model(name,pretrained=False,pretrained_cfg=cfg['pretrained_cfg']).eval()
    w=load_file(str(P/'models'/name/'model.safetensors'))
    if a.model=='swin':
        from timm.models.swin_transformer import checkpoint_filter_fn
        w=checkpoint_filter_fn(w,model)
    model.load_state_dict(w,strict=True);original={k:v.clone() for k,v in model.state_dict().items()};del w
    old=ROOT/'work2/research_wide/results/imagenette'/name/'checks.json'
    sha=hashlib.sha256((P/'models'/name/'model.safetensors').read_bytes()).hexdigest();assert sha==json.loads(old.read_text())['checkpoint_sha256']
    split=json.loads((ROOT/'work2/research_wide/datasets.json').read_text())['imagenette']
    def resolve(p):
        parts=Path(p).parts;return P/'data'/Path(*parts[parts.index('imagenette2-160'):])
    for sha_img,d in split['image_hashes'].items():assert hashlib.sha256(resolve(d['path']).read_bytes()).hexdigest()==sha_img
    train=[(resolve(p),y) for p,y in split['train']];mapping={Path(p).parent.name:y for part in ['train','validation','evaluation'] for p,y in split[part]}
    test=[(p,mapping[p.parent.name]) for p in sorted((P/'data/imagenette2-160/val').glob('*/*')) if p.suffix.lower() in ['.jpeg','.jpg','.png']];assert len(test)==3925
    assert not(set(p for p,y in train)&set(p for p,y in test))
    pilotpaths={resolve(p) for p,y in split['evaluation']};pilot=np.array([p in pilotpaths for p,y in test]);assert pilot.sum()==250
    tr=timm.data.create_transform(**timm.data.resolve_model_data_config(model),is_training=False)
    def images(samples):
        xx=[]
        for p,y in samples:
            with Image.open(p) as im:xx.append(tr(im.convert('RGB')))
        return torch.stack(xx)
    tx=images(train);ty=torch.tensor([y for p,y in train]);labels=np.array([y for p,y in test])
    with torch.inference_mode():teacher=torch.cat([model(tx[i:i+8]) for i in range(0,len(tx),8)]).clone()
    names=[n+'.weight' for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and m.in_features%32==0];qat=[False]
    for n,m in model.named_modules():
        if n+'.weight' not in names:continue
        def forward(self,x):
            w=self.weight
            if qat[0]:
                g=w.reshape(-1,32);s=(g.detach().abs().amax(1,keepdim=True)/127).clamp_min(torch.finfo(w.dtype).tiny);rounded=(g/s).round().clamp(-127,127)*s;w=(g+(rounded-g).detach()).reshape_as(w)
            return F.linear(x,w,self.bias)
        m.forward=types.MethodType(forward,m)
    def export():
        state={};parts=[];specs=[]
        for k,v in model.state_dict().items():
            ar=v.detach().numpy()
            if k in names:
                g=ar.reshape(-1,32);s=np.maximum(np.abs(g).max(1)/127,np.finfo(np.float32).tiny).astype('<f4');q=np.clip(np.rint(g/s[:,None]),-127,127).astype(np.int16)
                parts.extend([s.tobytes(),(q+128).astype(np.uint8).tobytes()]);specs.append(dict(name=k,shape=list(ar.shape),mode='int8',parts=2));state[k]=torch.from_numpy((q.astype(np.float32)*s[:,None]).reshape(ar.shape))
            else:parts.append(ar.tobytes());specs.append(dict(name=k,shape=list(ar.shape),mode='raw',dtype=str(ar.dtype),parts=1));state[k]=v.clone()
        import struct
        h=json.dumps(dict(tensors=specs,group=32,lengths=[len(p) for p in parts]),separators=(',',':')).encode();blob=b'VRC2'+struct.pack('<I',len(h))+h+b''.join(parts);compressed=zs.ZstdCompressor(level=3).compress(blob);assert zs.ZstdDecompressor().decompress(compressed)==blob
        return compressed,state
    def save(filename,obj):
        temp=out/(filename+'.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(out/filename)
    protocol=dict(model=name,checkpoint_sha256=sha,seed=a.seed,torch=torch.__version__,timm=timm.__version__,numpy=np.__version__,threads=a.threads,training_images=60,training_batch=8,epochs=4,updates=32,lr=1e-5,weight_decay=0,loss='0.9*4*KL(teacher(T=2)||student(T=2))+0.1*CE',test_samples=3925,pilot_samples=250,fresh_samples=3675,head_classes=1000,preprocessing=timm.data.resolve_model_data_config(model),test_manifest=[(str(p.relative_to(P)),y) for p,y in test],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save('protocol.json',protocol)
    results=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    for stage in ['original','rtn_group32','unpruned_ft','unpruned_qat','vit_reco']:
        if any(r['stage']==stage for r in results):continue
        model.load_state_dict(original);qat[0]=False;model.eval();torch.manual_seed(a.seed);masks={};hooks=[];compressed=None
        if stage=='vit_reco':
            for n,param in model.named_parameters():
                if n.endswith('weight') and ('.mlp.fc1.' in n or '.mlp.fc2.' in n):
                    mask=torch.ones_like(param).flatten();mask[torch.topk(param.detach().abs().flatten(),int(param.numel()*.1),largest=False).indices]=0;masks[n]=mask.reshape_as(param);hooks.append(param.register_hook(lambda grad,n=n:grad*masks[n]))
        def apply_masks():
            with torch.no_grad():
                for n,param in model.named_parameters():
                    if n in masks:param.mul_(masks[n])
        apply_masks()
        if stage in ['unpruned_ft','unpruned_qat','vit_reco']:
            opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.);history=[]
            for epoch in range(4):
                model.train();qat[0]=stage!='unpruned_ft' and epoch>=2;order=torch.randperm(len(tx));start=time.perf_counter();losses=[]
                for j in range(0,len(tx),8):
                    idx=order[j:j+8];opt.zero_grad(set_to_none=True);pred=model(tx[idx]);loss=.9*4*F.kl_div(F.log_softmax(pred/2,1),F.softmax(teacher[idx]/2,1),reduction='batchmean')+.1*F.cross_entropy(pred,ty[idx]);loss.backward();opt.step();apply_masks();losses.append(loss.item())
                history.append(dict(epoch=epoch+1,qat=qat[0],loss=float(np.mean(losses)),seconds=time.perf_counter()-start));print('TRAIN',a.model,stage,history[-1],flush=True)
            save(stage+'_training.json',history);del opt
        for n,param in model.named_parameters():
            if n in masks:assert torch.count_nonzero(param.detach()[masks[n]==0])==0
        for hook in hooks:hook.remove()
        qat[0]=False
        if stage in ['rtn_group32','unpruned_qat','vit_reco']:
            compressed,state=export();(out/(stage+'.vrc.zst')).write_bytes(compressed);model.load_state_dict(state);del state
        elif stage=='unpruned_ft':torch.save(model.state_dict(),out/'unpruned_ft.pt')
        model.eval();ys=[];start=time.perf_counter()
        with torch.inference_mode():
            for j in range(0,len(test),16):
                ys.append(model(images(test[j:j+16])).numpy())
                if j%800==0:print('EVAL',a.model,stage,j,len(test),round(time.perf_counter()-start,1),flush=True)
        logits=np.concatenate(ys);correct=logits.argmax(1)==labels;np.savez_compressed(out/(stage+'.npz'),logits=logits,labels=labels,pilot=pilot)
        row=dict(stage=stage,n=len(test),correct=int(correct.sum()),top1=float(correct.mean()*100),top5=float((np.argpartition(logits,-5,axis=1)[:,-5:]==labels[:,None]).any(1).mean()*100),pilot_top1=float(correct[pilot].mean()*100),fresh_top1=float(correct[~pilot].mean()*100),evaluation_seconds=time.perf_counter()-start)
        if compressed:row.update(archive_bytes=len(compressed),archive_sha256=hashlib.sha256(compressed).hexdigest(),compression_factor=sum(v.numel()*v.element_size() for v in original.values())/len(compressed))
        results.append(row);save('results.json',results);print('RESULT',a.model,row,flush=True)
    save('COMPLETE.json',dict(stages=len(results),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))

if __name__=='__main__':main()
