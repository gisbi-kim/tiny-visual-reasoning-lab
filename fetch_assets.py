"""Download original Tiny ImageNet and a pinned public 4B Qwen snapshot."""
from pathlib import Path
import hashlib,json,urllib.request,zipfile,time
import argparse
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parent
BASE_MODEL='Qwen/Qwen3.5-4B'
BASE_REVISION='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
DATASET_SHA256='6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-only',action='store_true')
    args=parser.parse_args()
    (ROOT/'docs').mkdir(exist_ok=True)
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    archive=raw/'tiny-imagenet-200.zip'
    source='https://cs231n.stanford.edu/tiny-imagenet-200.zip'
    if not archive.exists():
        print('DOWNLOADING',source,flush=True)
        temporary=archive.with_suffix('.part')
        urllib.request.urlretrieve(source,temporary)
        temporary.replace(archive)
    if digest(archive)!=DATASET_SHA256:raise RuntimeError('Original dataset checksum differs from the recorded training source')
    with zipfile.ZipFile(archive) as z:
        if not (raw/'tiny-imagenet-200/wnids.txt').exists():
            for member in z.infolist():
                target=(raw/member.filename).resolve()
                if not target.is_relative_to(raw.resolve()):raise RuntimeError('Unsafe archive path')
            z.extractall(raw)
    meta={'url':source,'bytes':archive.stat().st_size,'sha256':digest(archive)}
    (ROOT/'docs/dataset-download.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('DATASET_READY',meta,flush=True)
    if args.dataset_only:return
    model=BASE_MODEL
    revision=BASE_REVISION
    (ROOT/'docs/base-model.json').write_text(json.dumps({'model':model,'revision':revision},indent=2),encoding='utf-8')
    print('MODEL_DOWNLOAD',model,revision,flush=True)
    folder=Path(snapshot_download(model,revision=revision))
    files={str(p.relative_to(folder)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in folder.rglob('*') if p.is_file()}
    (ROOT/'docs/base-model-files.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    print('ASSETS_READY',sum(v['bytes'] for v in files.values()),flush=True)
if __name__=='__main__':main()
