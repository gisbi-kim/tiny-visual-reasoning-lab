"""Check the actual saved processor/Clef encoder for label leakage and image loss."""
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor
from clef_reference import encode_record
from schema import model_input, ACTIONS

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);args=p.parse_args()
processor=AutoProcessor.from_pretrained(args.checkpoint)
rows=[json.loads(s) for s in (root/'data/train.jsonl').read_text(encoding='utf-8').splitlines()]
row=rows[0]
image=Image.open(root/'data'/row['image']).convert('RGB')
source=model_input(image,row['instruction'])
records=[]
for label in ACTIONS:
    record=encode_record(processor.tokenizer,{**source,'gold':{'select':label},'object_keys':['dog','cat','banana']},
                         max_length=1024,processor=processor)
    records.append(record)
assert all(r.input_ids==records[0].input_ids for r in records)
media=records[0].media
assert media is not None and 'pixel_values' in media
assert all(torch.equal(r.media['pixel_values'],records[0].media['pixel_values']) for r in records)
blank=encode_record(processor.tokenizer,model_input(Image.new('RGB',image.size,'white'),row['instruction']),
                    max_length=1024,processor=processor)
assert not torch.equal(blank.media['pixel_values'],media['pixel_values'])
other=encode_record(processor.tokenizer,model_input(image,'Select the photo immediately below the cat.'),max_length=1024,processor=processor)
assert other.input_ids!=records[0].input_ids
report={'gold_changes_do_not_change_tokens':True,'gold_changes_do_not_change_pixels':True,
        'blank_image_changes_pixel_tensor':True,'instruction_changes_input_tokens':True,
        'sequence_length':len(records[0].input_ids),
        'media_shapes':{k:list(v.shape) for k,v in media.items() if torch.is_tensor(v)},
        'instruction':row['instruction'],'decoded_prompt':processor.tokenizer.decode(records[0].input_ids),
        'note':'Actual Clef encoder and saved processor probe; arbitrary gold/slot metadata are ignored, image pixels and raw user text are retained.'}
(root/'results').mkdir(exist_ok=True)
(root/'results/encoder-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='decoded_prompt'},ensure_ascii=False))
