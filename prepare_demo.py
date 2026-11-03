"""Create local demo crops from the official download; retain original aspect ratio."""
import json
from pathlib import Path
from prepare_data import build_index,crop,ROOT

def main():
    manifest=ROOT/'data/source-partitions.json'
    partitions=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else build_index()
    folder=ROOT/'data/demo';folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for key,items in partitions['test'].items():
        for i,item in enumerate(items[:10]):
            relative=f'demo/{key}-{i}.png'
            crop(item).save(ROOT/'data'/relative)
            records.append({'key':key,'image':relative,'source_id':item['id']})
    (ROOT/'data/demo.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(f'Demo ready: {len(records)} local crops. No images are uploaded.')

if __name__=='__main__':main()
