"""Fixed paired comparison of a generative Qwen and the released decision model."""
import argparse, collections, hashlib, json, random, re, statistics, time
from pathlib import Path
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
BASE = 'Qwen/Qwen3.5-4B'
REV = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
OUT = ROOT/'results/generative-comparison'
OUT.mkdir(parents=True, exist_ok=True)

def select():
    rows = [json.loads(line) for line in (ROOT/'data/v2_test.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = collections.defaultdict(list)
    for row in rows: groups[(row['family'], row['language'])].append(row)
    rng = random.Random(3407)
    chosen=[]
    for key in sorted(groups): chosen.extend(rng.sample(groups[key], 4))
    assert len(chosen)==96
    return chosen

def main():
    p=argparse.ArgumentParser();p.add_argument('mode', choices=['decision','generative']);args=p.parse_args()
    rows=select(); mode=args.mode
    if mode=='decision':
        from engine import DecisionEngine
        engine=DecisionEngine(ROOT/'weights-v2','cuda')
        def predict(im, prompt):
            result=engine.predict(im,prompt)
            return result['action'],result['action'],None
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        q=BitsAndBytesConfig.from_dict(json.loads((ROOT/'weights-v2/quantization.json').read_text()))
        processor=AutoProcessor.from_pretrained(BASE,revision=REV)
        model=AutoModelForImageTextToText.from_pretrained(BASE,revision=REV,dtype=torch.bfloat16,
              attn_implementation='eager',quantization_config=q,device_map={'':'cuda'}).eval()
        def predict(im, prompt):
            instruction=('The image has six photos in a 2 by 3 grid: A B C on the top row and D E F on the bottom row. '
                'Each photo is represented by its main object. Adjacent means immediately next to a tile, without wrapping. '
                'Choose A, B, C, D, E, F, or NONE if no tile qualifies. '
                'Respond with only the single letter or NONE.\nUser instruction: '+prompt)
            messages=[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':instruction}]}]
            inputs=processor.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=False,
                         tokenize=True,return_dict=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=8,do_sample=False,use_cache=True)
            new=output[0,inputs['input_ids'].shape[1]:]
            raw=processor.decode(new,skip_special_tokens=True).strip()
            match=re.fullmatch(r'\s*(NONE|[A-F])\s*[.!]?\s*',raw,re.IGNORECASE)
            return match.group(1).upper() if match else None,raw,len(new)
    def run(row):
        with Image.open(ROOT/'data'/row['image']) as im: image=im.convert('RGB')
        torch.cuda.synchronize();start=time.perf_counter()
        action,raw,tokens=predict(image,row['instruction'])
        torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
        return dict(scene_id=row['scene_id'],image=row['image'],image_sha256=hashlib.sha256((ROOT/'data'/row['image']).read_bytes()).hexdigest(),
                    instruction=row['instruction'],family=row['family'],language=row['language'],gold=row['action'],
                    action=action,raw=raw,generated_tokens=tokens,latency_ms=elapsed)
    for row in rows[:3]:run(row)
    output=[]
    for i,row in enumerate(rows):
        output.append(run(row))
        if (i+1)%12==0: print(mode,i+1,'/',len(rows),flush=True)
    result=dict(mode=mode,base=BASE,revision=REV,seed=3407,sample='4 random rows per task family and language from v2_test',
                count=len(output),correct=sum(r['gold']==r['action'] for r in output),
                invalid=sum(r['action'] is None for r in output),median_ms=statistics.median(r['latency_ms'] for r in output),
                prompt_policy='Greedy, thinking disabled, max 8 generated tokens, strict exact choice parsing' if mode=='generative' else 'Released calibrated v2 decision model',
                timing_scope='Warm, image already opened; preprocessing, model inference and output decoding; CUDA synchronized; model load excluded',rows=output)
    (OUT/(mode+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RESULT',mode,result['correct'],result['count'],result['invalid'],result['median_ms'],flush=True)
if __name__=='__main__':main()
