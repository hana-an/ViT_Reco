"""Fair CPU loading benchmark: all INT8 archives use the same direct-byte codec.

Run prepare while other evaluation proceeds, then benchmark only after it ends.
Fresh subprocess per measured sample; imported-runtime, warm-filesystem-cache
loading, NOT cold OS-cache boot or inference latency. No generic bit-unpacking.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
P = ROOT / 'vitreco_validation'
OUT = P / 'loading_benchmark'
MODELS = {'tiny':'deit_tiny_patch16_224.fb_in1k',
          'small':'deit_small_patch16_224.fb_in1k',
          'swin':'swin_tiny_patch4_window7_224.ms_in1k'}
STAGES = ['rtn_group32', 'unpruned_qat', 'vit_reco', 'qat_fp32_safetensors']

def decode(data):
    import numpy as np
    import torch
    import zstandard as zs
    import struct
    blob = zs.ZstdDecompressor().decompress(data)
    assert blob[:4] == b'VRC2'
    hlen = struct.unpack_from('<I', blob, 4)[0]
    h = json.loads(blob[8:8+hlen])
    assert h['group'] == 32
    pos = 8+hlen
    parts = []
    view = memoryview(blob)
    for length in h['lengths']:
        parts.append(view[pos:pos+length]); pos += length
    assert pos == len(blob)
    state = {}; j = 0
    for spec in h['tensors']:
        if spec['mode'] == 'int8':
            scales = np.frombuffer(parts[j], dtype='<f4')
            w = np.frombuffer(parts[j+1], dtype=np.uint8).astype(np.float32).reshape(-1,32)
            w -= 128
            w *= scales[:,None]
            ar = w.reshape(spec['shape']); j += 2
        else:
            ar = np.frombuffer(parts[j], dtype=spec['dtype']).reshape(spec['shape']).copy(); j += 1
        state[spec['name']] = torch.from_numpy(ar)
    assert j == len(parts)
    return state

def verify_decode(data, state):
    """Independent integer-offset formulation, bytewise compare every tensor."""
    import struct
    import numpy as np
    import zstandard as zs
    blob=zs.ZstdDecompressor().decompress(data)
    n=struct.unpack_from('<I',blob,4)[0]; h=json.loads(blob[8:8+n]); pos=8+n; j=0
    for spec in h['tensors']:
        lens=h['lengths'][j:j+spec['parts']]
        if spec['mode']=='int8':
            scales=np.frombuffer(blob,dtype='<f4',count=lens[0]//4,offset=pos)
            codes=np.frombuffer(blob,dtype=np.uint8,count=lens[1],offset=pos+lens[0])
            signed=codes.astype(np.int16)-128
            ref=(signed.astype(np.float32).reshape(-1,32)*scales[:,None]).reshape(spec['shape'])
        else:
            ref=np.frombuffer(blob,dtype=spec['dtype'],count=int(np.prod(spec['shape'])),offset=pos).reshape(spec['shape'])
        assert state[spec['name']].numpy().tobytes()==ref.tobytes(),spec['name']
        pos+=sum(lens);j+=spec['parts']
    return len(state)

def path_for(model,stage):
    if stage=='qat_fp32_safetensors':return OUT/(model+'_qat_fp32.safetensors')
    return P/'results'/model/'151'/(stage+'.vrc.zst')

def worker(model,stage):
    if (OUT/'STOP').exists():
        import signal
        os.kill(os.getppid(),signal.SIGTERM)
        raise SystemExit('Controller stopped for a clean benchmark restart.')
    import torch
    import timm
    import numpy
    import zstandard
    from safetensors.torch import load_file
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(151)
    path=path_for(model,stage)
    def high_water_kib():
        # VmHWM belongs to this exec's address space; ru_maxrss can retain
        # the controller's pre-exec high-water after fork/exec on Linux.
        return int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmHWM:')))
    baseline_hwm=high_water_kib()
    start=time.perf_counter_ns()
    if stage=='qat_fp32_safetensors':
        t1=start
        state=load_file(str(path))
    else:
        data=path.read_bytes(); t1=time.perf_counter_ns()
        state=decode(data)
        del data
    t2=time.perf_counter_ns()
    net=timm.create_model(MODELS[model],pretrained=False).eval()
    t3=time.perf_counter_ns()
    net.load_state_dict(state,strict=True)
    del state
    t4=time.perf_counter_ns()
    peak=high_water_kib()
    result=dict(model=model,stage=stage,archive_bytes=path.stat().st_size,
                read_ms=(t1-start)/1e6,decode_ms=(t2-t1)/1e6,
                construct_ms=(t3-t2)/1e6,install_ms=(t4-t3)/1e6,
                total_ms=(t4-start)/1e6,process_peak_rss_mib=peak/1024,
                pre_load_high_water_rss_mib=baseline_hwm/1024)
    print(json.dumps(result),flush=True)

def prepare():
    from safetensors.torch import save_file,load_file
    import torch
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    audit=[]
    for model in MODELS:
        for stage in STAGES[:3]:
            path=path_for(model,stage)
            if not path.exists():continue
            data=path.read_bytes();state=decode(data)
            count=verify_decode(data,state)
            audit.append(dict(model=model,stage=stage,tensors=count,bitwise_equal=True,
                              archive_sha256=hashlib.sha256(data).hexdigest()))
            if stage=='unpruned_qat':
                target=path_for(model,'qat_fp32_safetensors')
                save_file(state,str(target))
                loaded=load_file(str(target))
                assert all(v.numpy().tobytes()==loaded[k].numpy().tobytes() for k,v in state.items())
                del loaded
            del state,data
    (OUT/'codec_audit.json').write_text(json.dumps(audit,indent=2))
    print('Verified archives:',len(audit),flush=True)

def benchmark(reps):
    import numpy as np
    import torch,timm,zstandard
    for model in MODELS:
        assert (P/'results'/model/'151'/'COMPLETE.json').exists(),'Wait for validation to finish: '+model
    prepare()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    jobs=[(m,s) for m in MODELS for s in STAGES]
    rng=random.Random(491)
    rows=[]
    for rep in range(-1,reps):
        rng.shuffle(jobs)
        for model,stage in jobs:
            cp=subprocess.run([sys.executable,__file__,'--worker',model,stage],env=env,capture_output=True,text=True,check=True)
            row=json.loads(cp.stdout.strip().splitlines()[-1]);row['repetition']=rep
            print(json.dumps(row),flush=True)
            if rep>=0:rows.append(row)
            (OUT/'samples.json').write_text(json.dumps(rows,indent=2))
    summary=[]
    for model in MODELS:
        for stage in STAGES:
            r=[x for x in rows if x['model']==model and x['stage']==stage]
            d=dict(model=model,stage=stage,n=len(r),archive_bytes=r[0]['archive_bytes'])
            for metric in ['read_ms','decode_ms','construct_ms','install_ms','total_ms','process_peak_rss_mib','pre_load_high_water_rss_mib']:
                a=np.array([x[metric] for x in r])
                d[metric]=dict(median=float(np.median(a)),min=float(a.min()),max=float(a.max()),q25=float(np.percentile(a,25)),q75=float(np.percentile(a,75)))
            summary.append(d)
    meta=dict(torch=torch.__version__,timm=timm.__version__,numpy=np.__version__,zstandard=zstandard.__version__,
              cpu=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),''),
              torch_threads=2,interop_threads=1,openblas_threads=1,seed=491,repetitions=reps,
              boundary='Timer: file read + direct dequantization + model construction + strict state installation. Imports, first forward pass and preprocessing excluded.',
              cache='Warm OS filesystem cache; one discarded subprocess warmup per condition. No cache eviction or physical disk-cold claim.',
              memory='Linux /proc/self/status VmHWM is absolute process high-water RSS for the current exec address space, including imported runtime. Fresh child per sample; pre-load high-water also recorded. Not inference peak memory. ru_maxrss is deliberately avoided because inherited pre-exec high-water can dominate it.',
              design='Same direct-byte decoder for RTN, unpruned QAT and ViT-ReCo. FP32 safetensors contains exactly the dequantized unpruned-QAT tensors and uses standard memory-mapped load_file; its combined mapping/read time is in decode_ms and later page touches in install_ms.',
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(dict(metadata=meta,results=summary),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',nargs=2);p.add_argument('--prepare',action='store_true');p.add_argument('--reps',type=int,default=5);a=p.parse_args()
    if a.worker:worker(*a.worker)
    elif a.prepare:prepare()
    else:benchmark(a.reps)
