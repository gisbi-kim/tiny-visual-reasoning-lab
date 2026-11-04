"""Evaluate the original paired sample with two output budgets and thinking modes."""
import argparse, collections, hashlib, json, os, re, statistics, time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from compare_generation import BASE, REV, ROOT, select

OUT=ROOT/'results/generative-comparison'
STATUS=ROOT/'runs/generative-reasoning-status.json'

def parse_final(raw, thinking):
    # The thinking template has already emitted <think> in the input prompt.
    if thinking:
        if '</think>' not in raw:
            return None, None, False, ''
        raw=raw.split('</think>',1)[1]
    raw=re.sub(r'<\|(?:im_end|endoftext)\|>', '', raw).strip()
    if raw.startswith('<think>'):
        if '</think>' not in raw: return None,None,False,''
        raw=raw.split('</think>',1)[1].strip()
    strict=re.fullmatch(r'\s*(NONE|[A-F])\s*[.!]?\s*',raw,re.I)
    relaxed=re.fullmatch(r'\s*(?:(?:정답|답|answer|final answer)\s*[:：]\s*)?(NONE|[A-F])\s*[.!]?\s*',raw,re.I)
    return (strict.group(1).upper() if strict else None,
            relaxed.group(1).upper() if relaxed else None, True, raw)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--warmup',type=int,default=3)
    args=parser.parse_args()
    rows=select()
    previous=json.loads((OUT/'decision.json').read_text(encoding='utf-8'))['rows']
    assert len(rows)==len(previous)==96
    for r,old in zip(rows,previous):
        assert all(r[k]==old[k] for k in ('scene_id','instruction','family','language')) and r['action']==old['gold']
        assert hashlib.sha256((ROOT/'data'/r['image']).read_bytes()).hexdigest()==old['image_sha256']
    q=BitsAndBytesConfig.from_dict(json.loads((ROOT/'weights-v2/quantization.json').read_text()))
    processor=AutoProcessor.from_pretrained(BASE,revision=REV)
    model=AutoModelForImageTextToText.from_pretrained(BASE,revision=REV,dtype=torch.bfloat16,
        attn_implementation='eager',quantization_config=q,device_map={'':'cuda'}).eval()
    eos=model.generation_config.eos_token_id
    eos_ids=set(eos if isinstance(eos,list) else [eos])
    def run(row, budget, thinking):
        with Image.open(ROOT/'data'/row['image']) as im: image=im.convert('RGB')
        instruction=('The image has six photos in a 2 by 3 grid: A B C on the top row and D E F on the bottom row. '
            'Each photo is represented by its main object. Adjacent means immediately next to a tile, without wrapping. '
            'Choose A, B, C, D, E, F, or NONE if no tile qualifies. '
            'Respond with only the single letter or NONE.\nUser instruction: '+row['instruction'])
        messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':instruction}]}]
        torch.cuda.synchronize(); start=time.perf_counter()
        inputs=processor.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=thinking,
                    tokenize=True,return_dict=True,return_tensors='pt').to('cuda')
        with torch.inference_mode(): output=model.generate(**inputs,max_new_tokens=budget,do_sample=False,use_cache=True)
        tokens=output[0,inputs['input_ids'].shape[1]:]
        raw=processor.decode(tokens,skip_special_tokens=False)
        strict,relaxed,thought_closed,final=parse_final(raw,thinking)
        torch.cuda.synchronize(); elapsed=(time.perf_counter()-start)*1000
        finished=bool(len(tokens) and int(tokens[-1]) in eos_ids)
        return dict(scene_id=row['scene_id'], image=row['image'],instruction=row['instruction'],gold=row['action'],
                    family=row['family'],language=row['language'],strict_action=strict,relaxed_action=relaxed,
                    raw_with_special_tokens=raw,final_text=final,think_closed=thought_closed,
                    generated_tokens=len(tokens),eos_reached=finished,hit_token_limit=len(tokens)>=budget and not finished,
                    latency_ms=elapsed)
    configs=[(40,False),(40,True),(120,False),(120,True)]
    for budget,thinking in configs:
        key=f'thinking-{int(thinking)}-tokens-{budget}'
        for r in rows[:args.warmup]:run(r,budget,thinking)
        results=[]
        for i,row in enumerate(rows):
            results.append(run(row,budget,thinking))
            STATUS.write_text(json.dumps(dict(phase='running',config=key,completed=i+1,total=96),indent=2),encoding='utf-8')
            if (i+1)%8==0: print(key,i+1,'/ 96',flush=True)
        report=dict(config=key,budget=budget,thinking=thinking,count=96,strict_correct=sum(r['strict_action']==r['gold'] for r in results),
             relaxed_correct=sum(r['relaxed_action']==r['gold'] for r in results),
             no_final=sum(not r['think_closed'] for r in results),invalid=sum(r['relaxed_action'] is None for r in results),
             eos_reached=sum(r['eos_reached'] for r in results),hit_token_limit=sum(r['hit_token_limit'] for r in results),
             median_ms=statistics.median(r['latency_ms'] for r in results),median_generated_tokens=statistics.median(r['generated_tokens'] for r in results),rows=results)
        (OUT/(key+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('RESULT',key,'strict',report['strict_correct'],'relaxed',report['relaxed_correct'],
              'no_final',report['no_final'],'limit',report['hit_token_limit'],'median_ms',report['median_ms'],flush=True)
    STATUS.write_text(json.dumps(dict(phase='complete',configs=len(configs))),encoding='utf-8')

if __name__=='__main__': main()
