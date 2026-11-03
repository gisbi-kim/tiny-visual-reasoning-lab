"""Local UI: photograph grid and verbatim user text go to the actual 4B model."""
import argparse,base64,json,random,threading
from io import BytesIO
from pathlib import Path
from PIL import Image,UnidentifiedImageError
from flask import Flask,jsonify,render_template,request
from render import render,SIZE

ROOT=Path(__file__).resolve().parent

def create_app(engine):
    app=Flask(__name__)
    app.json.sort_keys=False
    app.config['MAX_CONTENT_LENGTH']=3*1024*1024
    lock=threading.Lock();rng=random.Random()
    examples=json.loads((ROOT/'data/demo.json').read_text(encoding='utf-8'))
    by_key={}
    for item in examples:by_key.setdefault(item['key'],[]).append(item)
    @app.get('/')
    def index():return render_template('index.html',device=str(engine.device))
    @app.get('/health')
    def health():return jsonify(ready=True,device=str(engine.device),model='Qwen3.5-4B + LoRA + Clef')
    @app.post('/scene')
    def scene():
        canonical=(request.get_json(silent=True) or {}).get('canonical',False)
        keys=['dog','banana','cat','lifeboat','goldfish','backpack'] if canonical else rng.sample(list(by_key),6)
        items=[by_key[k][0] if canonical else rng.choice(by_key[k]) for k in keys]
        pictures=[]
        for item in items:
            with Image.open(ROOT/'data'/item['image']) as im:pictures.append(im.convert('RGB'))
        buffer=BytesIO();render(pictures).save(buffer,format='PNG')
        return jsonify(image='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())
    @app.post('/predict')
    def predict():
        payload=request.get_json(silent=True)
        if not isinstance(payload,dict):return jsonify(error='JSON 입력이 필요합니다.'),400
        text=payload.get('instruction','')
        if not isinstance(text,str) or not text.strip() or len(text)>1000:
            return jsonify(error='지시를 1~1000자로 입력해 주세요.'),400
        try:
            encoded=payload['image']
            if not isinstance(encoded,str) or not encoded.startswith('data:image/png;base64,'):
                raise ValueError('PNG 격자 이미지가 필요합니다.')
            image=Image.open(BytesIO(base64.b64decode(encoded.split(',',1)[1],validate=True)))
            if image.size!=SIZE:raise ValueError('이미지 크기는 384 × 256이어야 합니다.')
            with lock:prediction=engine.predict(image.convert('RGB'),text)
            return jsonify(prediction)
        except (ValueError,KeyError,UnidentifiedImageError,OSError) as exc:return jsonify(error=str(exc)),400
        except RuntimeError as exc:return jsonify(error=str(exc)),500
    return app

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='weights')
    p.add_argument('--device',default='auto',choices=['auto','cuda','cpu','mps'])
    p.add_argument('--port',type=int,default=7861);args=p.parse_args()
    if not (ROOT/'data/demo.json').exists():raise SystemExit('Run: python fetch_assets.py --dataset-only && python prepare_demo.py')
    if not (Path(args.checkpoint)/'joint_head.safetensors').exists():raise SystemExit('Run: python download_model.py')
    from engine import DecisionEngine
    engine=DecisionEngine(args.checkpoint,args.device)
    print(f'Ready: {engine.device}; http://127.0.0.1:{args.port}',flush=True)
    create_app(engine).run(host='127.0.0.1',port=args.port,debug=False,threaded=True)

if __name__=='__main__':main()
