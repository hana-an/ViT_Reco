"""Generate a factual report from completed extended-validation stages only."""
import json,time
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
P=Path(__file__).resolve().parent/'vitreco_validation'
rows=[];comparisons=[]
for f in sorted((P/'results').glob('*/*/results.json')):
    model=f.parents[1].name;seed=f.parent.name
    for r in json.loads(f.read_text()):rows.append(dict(model=model,seed=seed,**r))
    candidate=f.parent/'vit_reco.npz'
    if not candidate.exists():continue
    a=np.load(candidate);labels=a['labels'];ca=a['logits'].argmax(1)==labels
    for baseline in ['original','rtn_group32','unpruned_ft','unpruned_qat']:
        other=f.parent/(baseline+'.npz')
        if not other.exists():continue
        b=np.load(other);assert np.array_equal(labels,b['labels']);cb=b['logits'].argmax(1)==labels
        for scope,mask in [('full',np.ones(len(labels),dtype=bool)),('fresh',~a['pilot'])]:
            delta=ca[mask].astype(int)-cb[mask].astype(int);n=len(delta);counts=np.array([(delta==-1).sum(),(delta==0).sum(),(delta==1).sum()]);draws=np.random.default_rng(77).multinomial(n,counts/n,size=10000);boots=100*(draws[:,2]-draws[:,0])/n
            bad,good=int(counts[0]),int(counts[2]);pv=binomtest(bad,bad+good,.5).pvalue if bad+good else 1.
            comparisons.append(dict(model=model,seed=seed,baseline=baseline,scope=scope,n=n,delta_pp=float(delta.mean()*100),ci95_pp=np.percentile(boots,[2.5,97.5]).tolist(),lost=bad,gained=good,p=pv))
report={'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'completed_stages':rows,'comparisons':comparisons,'scope':'Imagenette-160, 1000-class original ImageNet heads; full validation 3925 images; fresh excludes 250 pilot images. One training seed so far. Intervals conditional on trained models; p values unadjusted exploratory.'}
(P/'summary.json').write_text(json.dumps(report,indent=2))
text=['# ViT-ReCo: extended validation progress','',report['scope'],'','| Model | Seed | Stage | N | Top-1 (%) | Fresh top-1 (%) |','|---|---|---|---:|---:|---:|']
for r in rows:text.append(f"| {r['model']} | {r['seed']} | {r['stage']} | {r['n']} | {r['top1']:.3f} | {r['fresh_top1']:.3f} |")
text+=['','## Paired comparisons: ViT-ReCo minus baseline','','| Model | Baseline | Split | Difference (pp) | 95% bootstrap CI |','|---|---|---|---:|---|']
for r in comparisons:text.append(f"| {r['model']} | {r['baseline']} | {r['scope']} | {r['delta_pp']:+.3f} | [{r['ci95_pp'][0]:+.3f}, {r['ci95_pp'][1]:+.3f}] |")
text+=['','Only completed stages appear. Missing rows are not negative or zero results. Repeated seeds, extended ImageNet-V2 evaluation, and published-method comparisons remain separate experiments. No superiority or accuracy-equivalence conclusion is implied.']
(P/'RESULTS.md').write_text('\n'.join(text)+'\n')
print(json.dumps(report,indent=2))
