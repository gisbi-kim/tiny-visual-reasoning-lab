"""Independent-process reload, validation-only calibration and held-out evaluation."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
from engine import DecisionEngine
from schema import ACTIONS
from oracle import conditional_can_vary

ROOT=Path(__file__).resolve().parent

def read_rows(name):
    return [json.loads(s) for s in (ROOT/'data'/f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def metrics(predictions):
    confusion=[[0]*len(ACTIONS) for _ in ACTIONS]
    correct=[];confidence=[];nll=[];brier=[]
    groups=defaultdict(list);semantics=defaultdict(list);minimal_pairs=defaultdict(list)
    condition=defaultdict(list);language=defaultdict(list);branch_difficulty=defaultdict(list)
    for p in predictions:
        gold=p['action'];pred=p['prediction']['action'];probs=p['prediction']['probabilities']
        ok=gold==pred
        confusion[ACTIONS.index(gold)][ACTIONS.index(pred)]+=1
        correct.append(ok);confidence.append(max(probs.values()))
        nll.append(-math.log(max(probs[gold],1e-12)))
        brier.append(sum((probs[a]-int(a==gold))**2 for a in ACTIONS))
        groups[p['scene_id']].append(p)
        semantics[(p['scene_id'],json.dumps(p['program'],sort_keys=True))].append(p)
        if p.get('contrast_pair'):minimal_pairs[(p['contrast_pair'],p['language'])].append(p)
        condition[p['family']].append(ok);language[p['language']].append(ok)
        varies=conditional_can_vary(p['program'])
        if varies is not None:branch_difficulty['image_dependent_branch' if varies else 'catalog_determined_branch'].append(ok)
    ece=0
    for lo in np.arange(0,1,.1):
        ix=[i for i,c in enumerate(confidence) if lo<=c<lo+.1 or lo>.89 and c==1]
        if ix: ece+=len(ix)/len(correct)*abs(np.mean([correct[i] for i in ix])-np.mean([confidence[i] for i in ix]))
    paired=[]
    for group in groups.values():
        for i,a in enumerate(group):
            for b in group[i+1:]:
                if a['action']!=b['action']:
                    paired.append(a['prediction']['action']==a['action'] and b['prediction']['action']==b['action'])
    paraphrase=[]
    for group in semantics.values():
        if len(group)>1: paraphrase.append(all(p['prediction']['action']==p['action'] for p in group))
    minimal=[all(p['prediction']['action']==p['action'] for p in group) for group in minimal_pairs.values()
             if len(set(p['action'] for p in group))>=2]
    # Cluster bootstrap over scenes; photo reuse between scenes remains a limitation.
    sizes=np.array([len(g) for g in groups.values()])
    hits=np.array([sum(p['action']==p['prediction']['action'] for p in g) for g in groups.values()])
    rng=np.random.default_rng(3407)
    samples=rng.integers(0,len(sizes),size=(2000,len(sizes)))
    boots=hits[samples].sum(axis=1)/sizes[samples].sum(axis=1)
    def div(x,y):return x/y if y else None
    return dict(count=len(correct),accuracy=float(np.mean(correct)),scene_bootstrap_ci95=np.quantile(boots,[.025,.975]).tolist(),
        ci_note='Scene-cluster bootstrap; repeated original photographs across scenes may still induce dependence.',
        action_order=ACTIONS,confusion=confusion,
        action_recall={a:div(confusion[i][i],sum(confusion[i])) for i,a in enumerate(ACTIONS)},
        none_precision=div(confusion[-1][-1],sum(row[-1] for row in confusion)),none_recall=div(confusion[-1][-1],sum(confusion[-1])),
        condition_accuracy={k:float(np.mean(v)) for k,v in condition.items()},language_accuracy={k:float(np.mean(v)) for k,v in language.items()},
        conditional_difficulty={k:{'count':len(v),'accuracy':float(np.mean(v))} for k,v in branch_difficulty.items()},
        changed_instruction_both_correct= float(np.mean(paired)) if paired else None,changed_instruction_pairs=len(paired),
        minimal_contrast_both_correct=float(np.mean(minimal)) if minimal else None,minimal_contrast_pairs=len(minimal),
        same_meaning_all_correct=float(np.mean(paraphrase)) if paraphrase else None,same_meaning_groups=len(paraphrase),
        nll=float(np.mean(nll)),brier=float(np.mean(brier)),ece_10_bins=float(ece),
        median_latency_ms=float(np.median([p['prediction']['latency_ms'] for p in predictions])),
        p95_latency_ms=float(np.quantile([p['prediction']['latency_ms'] for p in predictions],.95)))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--device',default='cuda',choices=['cuda','cpu','mps','auto'])
    parser.add_argument('--calibrate',action='store_true')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--suites',nargs='+',default=['test_id','test_language','test_composition','test_combined'])
    parser.add_argument('--output',default='results')
    args=parser.parse_args()
    checkpoint=Path(args.checkpoint)
    output=Path(args.output)
    start=time.perf_counter()
    engine=DecisionEngine(checkpoint,args.device)
    load_seconds=time.perf_counter()-start
    if args.calibrate:
        rows=read_rows('calibration')
        values=[];targets=[]
        for i,row in enumerate(rows):
            with Image.open(ROOT/'data'/row['image']) as image:
                values.append(engine.logits(image,row['instruction']).cpu().tolist())
            targets.append(ACTIONS.index(row['action']))
            if (i+1)%100==0:print('CALIBRATION',i+1,len(rows),flush=True)
        import torch
        logits=torch.tensor(values,dtype=torch.float32)
        gold=torch.tensor(targets)
        temps=np.geomspace(.1,10,161)
        # All uncalibrated saved temperatures must be one for this fitted scalar.
        if not math.isclose(engine.temperature/engine.calibration['temperature'],1.):
            raise RuntimeError('Calibration expects base saved temperatures of one')
        losses=[float(torch.nn.functional.cross_entropy(logits/float(t),gold)) for t in temps]
        best=int(np.argmin(losses))
        calibration=dict(temperature=float(temps[best]),nll=float(losses[best]),rows=len(rows),split='calibration',
                         uncalibrated_nll=float(torch.nn.functional.cross_entropy(logits,gold)))
        dump(checkpoint/'calibration.json',calibration)
        engine.temperature=calibration['temperature']
    # Warm-up is excluded from steady-state latency.
    row=read_rows('validation')[0]
    with Image.open(ROOT/'data'/row['image']) as image:engine.predict(image,row['instruction'])
    report=dict(checkpoint=str(checkpoint),device=str(engine.device),dtype=str(engine.dtype),load_seconds=load_seconds,
        latency_scope='Synchronized DecisionEngine.predict: processor, transfer, backbone, head and probabilities; includes any lazy pixel read/decode inside the call. Image.open is outside the timer.',suites={})
    for name in args.suites:
        rows=read_rows(name)
        if args.limit: rows=rows[:args.limit]
        predictions=[]
        for i,row in enumerate(rows):
            with Image.open(ROOT/'data'/row['image']) as image:
                pred=engine.predict(image,row['instruction'])
            predictions.append(dict(**row,prediction=pred))
            if (i+1)%100==0:print('EVALUATION',name,i+1,len(rows),flush=True)
        report['suites'][name]=metrics(predictions)
        dump(output/f'{name}_predictions.json',predictions)
        dump(output/'report.json',report)
        print(name,report['suites'][name]['accuracy'],flush=True)
    print('REPORT',output/'report.json',flush=True)

if __name__=='__main__':main()
