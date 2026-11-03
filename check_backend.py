"""Separate-process trained-weight parity check; also records quantized layers."""
import os
os.environ.setdefault('UNSLOTH_COMPILE_DISABLE','1')
import argparse,json,hashlib
from pathlib import Path
from PIL import Image
from schema import questions
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--backend',choices=['unsloth','portable'],required=True);p.add_argument('--output',required=True);p.add_argument('--split',default='train');p.add_argument('--count',type=int,default=16);args=p.parse_args()
    if args.backend=='unsloth':
        from unsloth import FastDecisionModel
        import torch
        model,processor=FastDecisionModel.from_pretrained(args.checkpoint,load_in_4bit=True,dtype=torch.bfloat16)
        FastDecisionModel.for_inference(model)
        encoder=model._backbone()
        def predict(image,text):return FastDecisionModel.predict(model,processor,text,questions(),images=[image])['select']
    else:
        from engine import DecisionEngine
        e=DecisionEngine(args.checkpoint,'cuda');encoder=e.encoder.get_base_model()
        def predict(image,text):return e.predict(image,text)
    rows=[json.loads(s) for s in Path(f'data/{args.split}.jsonl').read_text(encoding='utf-8').splitlines()][:args.count]
    outputs=[]
    for row in rows:
        with Image.open(Path('data')/row['image']) as im:result=predict(im,row['instruction'])
        outputs.append({'scene_id':row['scene_id'],'instruction':row['instruction'],'gold':row['action'],'prediction':result})
    report={'backend':args.backend,'accuracy':sum(r['gold']==r['prediction'].get('action',r['prediction'].get('answer')) for r in outputs)/len(outputs),
        'quantized_modules':{name:str(module.weight.dtype) for name,module in encoder.named_modules() if type(module).__name__=='Linear4bit'},
        'rows':outputs}
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('BACKEND_RESULT',args.backend,report['accuracy'],len(report['quantized_modules']),flush=True)
if __name__=='__main__':main()
