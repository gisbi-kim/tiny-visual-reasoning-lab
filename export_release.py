"""Package a completed, calibrated run; no base weights or dataset photographs."""
import argparse,json,shutil,zipfile
from datetime import datetime,timezone
from pathlib import Path
from download_model import sha256
from schema import ACTIONS,questions

ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='main');p.add_argument('--tag',default='model-v1');args=p.parse_args()
    run=ROOT/'runs'/args.run;checkpoint=run/'best';config=read(run/'config.json');metrics=read(run/'metrics.json')
    if read(run/'status.json')['phase']!='trained':raise RuntimeError('Training has not finished')
    if not (checkpoint/'calibration.json').exists():raise RuntimeError('Calibrate/evaluate the selected checkpoint before publishing')
    validation=read(run/'validation.json');best=max(validation,key=lambda x:x['accuracy'])
    docs=ROOT/'docs';dist=ROOT/'dist';dist.mkdir(exist_ok=True)
    selected=dist/'release-weights'
    if selected.exists():raise RuntimeError('Preserve the previous export: choose a new export location or inspect it first')
    selected.mkdir()
    names=['adapter_config.json','adapter_model.safetensors','joint_head_config.json','joint_head.safetensors',
        'processor_config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja',
        'unsloth_decision_config.json','quantization.json','calibration.json']
    for name in names:shutil.copy2(checkpoint/name,selected/name)
    dump(selected/'inference_schema.json',{'actions':ACTIONS,'questions':questions()})
    percentage=100*config['trainable_parameters']/config['total_parameters']
    card=f'''# Tiny Visual Reasoning Lab — Qwen3.5-4B LoRA + Clef

Base: {config['base_model']}, revision {config['base_revision']}.
This archive contains trained LoRA, a trained Clef selection head, processor and calibration.
The separate base model is downloaded from Hugging Face. No Tiny ImageNet photographs are included.

Trained locally on {config['gpu']} for {metrics['train_runtime']:.1f} seconds; selected step {best['step']}.
The time above is the complete training/selection run. The released checkpoint contains updates only through its selected step ({best['step']*config['batch']*config['accum']:,} sample presentations).
Trainable logical parameters: {config['trainable_parameters']:,}/{config['total_parameters']:,} ({percentage:.4f}%).
LoRA: {config['lora_trainable_parameters']:,}; Clef head: {config['head_parameters']:,}.
These counts include the decision head; 4-bit storage does not reduce the logical denominator.

Data: 31,360 synthetic bilingual instructions over curated Tiny ImageNet 2×3 grids (40 classes).
Output: A–F or NONE. This is a constrained selection experiment, not an unrestricted reasoning assistant.
Evaluation, limitations, exact data-generation code and attribution:
https://github.com/gisbi-kim/tiny-visual-reasoning-lab

Project/Clef code and weights: Apache-2.0, subject to the base model's original terms.
Image rights remain with original owners; images are downloaded separately from Stanford.
'''
    (selected/'README.md').write_text(card,encoding='utf-8')
    shutil.copy2(ROOT/'LICENSE',selected/'LICENSE');shutil.copy2(ROOT/'CLEF_LICENSE.txt',selected/'CLEF_LICENSE.txt')
    files={p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(selected.iterdir())}
    archive=dist/'tiny-visual-reasoning-4b.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in files:z.write(selected/name,name)
    release={'repository':'gisbi-kim/tiny-visual-reasoning-lab','tag':args.tag,
        'url':f'https://github.com/gisbi-kim/tiny-visual-reasoning-lab/releases/download/{args.tag}/{archive.name}',
        'bytes':archive.stat().st_size,'sha256':sha256(archive),'files':files,
        'base_model':config['base_model'],'base_revision':config['base_revision'],'created_at':datetime.now(timezone.utc).isoformat()}
    dump(docs/'release.json',release)
    dump(docs/'training-config.json',config);dump(docs/'training-metrics.json',metrics)
    dump(docs/'validation-history.json',[{'step':v['step'],'accuracy':v['accuracy']} for v in validation])
    dump(docs/'training-history.json',read(run/'history.json'));dump(docs/'encoding.json',read(run/'encoding.json'))
    print('RELEASE_READY',archive,archive.stat().st_size,flush=True)

if __name__=='__main__':main()
