"""Post-training input ablations and mirrored-board consistency; no fitting."""
import argparse,json
from pathlib import Path
from PIL import Image
from engine import DecisionEngine
from oracle import execute
from prepare_data import RAW
from render import render,SIZE

ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--count',type=int,default=140);p.add_argument('--output',default='results/diagnostics.json');args=p.parse_args()
    rows=[json.loads(s) for s in (ROOT/'data/validation_select.jsonl').read_text(encoding='utf-8').splitlines()][:args.count]
    engine=DecisionEngine(args.checkpoint,args.device);records=[]
    permutation=[2,1,0,5,4,3]
    for i,row in enumerate(rows):
        with Image.open(ROOT/'data'/row['image']) as im:
            original=engine.predict(im,row['instruction'])['action']
            no_text=engine.predict(im,'Choose one tile.')['action']
        no_image=engine.predict(Image.new('RGB',SIZE,(245,246,248)),row['instruction'])['action']
        board=[row['object_keys'][j] for j in permutation]
        mirrored_gold,_=execute(board,row['program'])
        mirrored=None
        if mirrored_gold is not None:
            pictures=[]
            for j in permutation:
                with Image.open(RAW/row['source_ids'][j]) as im:pictures.append(im.convert('RGB').crop(row['boxes'][j]))
            mirrored=engine.predict(render(pictures),row['instruction'])['action']
        records.append({'scene_id':row['scene_id'],'instruction':row['instruction'],'gold':row['action'],
            'original':original,'without_specific_instruction':no_text,'without_image':no_image,
            'reordered_gold':mirrored_gold,'reordered_prediction':mirrored})
        if (i+1)%20==0:print('DIAGNOSTICS',i+1,len(rows),flush=True)
    def rate(key):return sum(r[key]==r['gold'] for r in records)/len(records)
    changed=[r for r in records if r['reordered_gold'] is not None and r['reordered_gold']!=r['gold']]
    report={'count':len(records),'source':'fixed validation selection subset; not a new independent test set',
        'original_accuracy':rate('original'),'blank_image_original_label_accuracy':rate('without_image'),
        'generic_instruction_original_label_accuracy':rate('without_specific_instruction'),
        'ablation_note':'Ablations retain original labels solely to measure reliance on inputs; they are not valid task accuracy on blank/underspecified inputs.',
        'reordering':'Swap columns 1 and 3 without mirroring pixels; hold text fixed and recompute gold with the oracle; skip ambiguous new boards.',
        'changed_gold_cases':len(changed),'changed_gold_both_correct':sum(r['original']==r['gold'] and r['reordered_prediction']==r['reordered_gold'] for r in changed)/len(changed) if changed else None,
        'rows':records}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DIAGNOSTICS_READY',output,flush=True)
if __name__=='__main__':main()
