"""Download this project's trained adapter/head, verifying every released file."""
import argparse,hashlib,json,shutil,tempfile,urllib.request,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='weights')
    p.add_argument('--version',choices=['latest','v1','v2'],default='latest');args=p.parse_args()
    manifest=ROOT/'docs'/('release.json' if args.version=='latest' else f'release-{args.version}.json')
    if not manifest.exists():raise SystemExit('No published trained checkpoint is recorded in this checkout yet.')
    release=json.loads(manifest.read_text(encoding='utf-8'));destination=Path(args.output).resolve()
    if destination.exists():
        if all((destination/name).is_file() and sha256(destination/name)==info['sha256'] for name,info in release['files'].items()):
            print('Verified existing trained weights:',destination);return
        raise SystemExit('Output exists but does not match the release. Choose a new --output directory to preserve it.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='tiny-model-',dir=destination.parent) as temporary:
        root=Path(temporary);archive=root/'model.zip';unpacked=root/'unpacked';unpacked.mkdir()
        print('Downloading trained LoRA + Clef head:',release['url'],flush=True)
        request=urllib.request.Request(release['url'],headers={'User-Agent':'tiny-visual-reasoning-lab'})
        with urllib.request.urlopen(request,timeout=120) as response,archive.open('wb') as output:shutil.copyfileobj(response,output)
        if archive.stat().st_size!=release['bytes'] or sha256(archive)!=release['sha256']:raise RuntimeError('Release archive checksum mismatch')
        with zipfile.ZipFile(archive) as z:
            if set(z.namelist())!=set(release['files']):raise RuntimeError('Unexpected archive entries')
            for name in z.namelist():
                if not (unpacked/name).resolve().is_relative_to(unpacked.resolve()):raise RuntimeError('Unsafe archive path')
            z.extractall(unpacked)
        for name,info in release['files'].items():
            path=unpacked/name
            if path.stat().st_size!=info['bytes'] or sha256(path)!=info['sha256']:raise RuntimeError(f'File checksum mismatch: {name}')
        unpacked.rename(destination)
    print('Verified trained weights:',destination)
    print('The separate Qwen base model downloads from Hugging Face on first inference (~9.34 GB).')

if __name__=='__main__':main()
