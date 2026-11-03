"""Prewritten free-form UI examples; fixed before final model evaluation."""
import argparse,json
from pathlib import Path
from PIL import Image
from render import render
from engine import DecisionEngine

ROOT=Path(__file__).resolve().parent
# Canonical layout: dog / banana / cat ; lifeboat / goldfish / backpack.
# Independent handwritten gold, not passed to inference.
CASES=[
    ('고양이 있는 칸에서 좌측으로 갔다가 밑으로 내려가면 어디야?','E'),
    ('개 사진 아래 칸으로 내려간 다음 오른쪽 옆 칸을 찍어줘.','E'),
    ('배가 있는 줄에서 동물 사진 하나만 찾아줄래?','E'),
    ('사진 여섯 장 중 동물이 나온 게 세 장 이상이면 바나나, 그보다 적으면 배낭으로 해줘.','B'),
    ('과일보다 탈것 사진이 더 많을 때만 고양이를 선택하고, 동점이거나 적으면 가방을 골라.','F'),
    ('바나나랑 같은 가로줄인데 동물은 아닌 사진이 뭐지?','B'),
    ('고양이 오른편 바로 옆 사진을 선택해 줘.','NONE'),
    ('개와 같은 세로줄에 있는 과일 사진을 골라봐.','NONE'),
    ('Start on the cat. Take a step left, then a step down. Which tile do you land on?','E'),
    ('I want the animal picture on the same horizontal line as the boat.','E'),
    ('Pick the banana if three or more of these pictures show animals. Otherwise, pick the backpack.','B'),
    ('Only pick the cat if vehicle pictures outnumber fruit pictures. In a tie, or if there are fewer vehicles, pick the bag.','F'),
    ('On the row with the banana, which picture is left after removing all the animals?','B'),
    ('What is directly to the right of the cat? Select that tile.','NONE'),
]
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda');p.add_argument('--output',default='results/freeform.json');args=p.parse_args()
    entries=json.loads((ROOT/'data/demo.json').read_text(encoding='utf-8'))
    indices=json.loads((ROOT/'docs/demo-selection.json').read_text(encoding='utf-8'))['indices']
    keys=['dog','banana','cat','lifeboat','goldfish','backpack'];pictures=[];sources=[]
    for key in keys:
        item=[x for x in entries if x['key']==key][indices[key]];sources.append(item['source_id'])
        with Image.open(ROOT/'data'/item['image']) as im:pictures.append(im.convert('RGB'))
    image=render(pictures);engine=DecisionEngine(args.checkpoint,args.device)
    rows=[{'instruction':text,'gold':gold,'prediction':engine.predict(image,text)} for text,gold in CASES]
    report={'scope':'Small prewritten free-form demonstration on one fixed board; not an independent benchmark or representative accuracy estimate.',
        'device':args.device,'object_keys':keys,'source_ids':sources,'accuracy':sum(r['gold']==r['prediction']['action'] for r in rows)/len(rows),'rows':rows}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FREEFORM',report['accuracy'],flush=True)
if __name__=='__main__':main()
