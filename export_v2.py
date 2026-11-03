"""Export v2 without modifying the v1 archive or the published latest manifest."""
import json, shutil, zipfile
from datetime import datetime, timezone
from pathlib import Path
from download_model import sha256
from schema import ACTIONS, questions

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    run = ROOT/'runs/v2'
    checkpoint = run/'best'
    config, metrics = read(run/'config.json'), read(run/'metrics.json')
    if read(run/'status.json')['phase'] != 'trained':
        raise RuntimeError('Training has not finished')
    if not (checkpoint/'calibration.json').exists():
        raise RuntimeError('Calibrate the selected checkpoint before export')
    validation = read(run/'validation.json')
    best = max(validation, key=lambda x: x['accuracy'])
    selected = ROOT/'dist/v2/release-weights'
    if selected.exists():
        raise RuntimeError('Preserve the existing v2 export; inspect it before rerunning')
    selected.mkdir(parents=True)
    names = ['adapter_config.json', 'adapter_model.safetensors', 'joint_head_config.json',
             'joint_head.safetensors', 'processor_config.json', 'tokenizer_config.json',
             'tokenizer.json', 'chat_template.jinja', 'unsloth_decision_config.json',
             'quantization.json', 'calibration.json']
    for name in names:
        shutil.copy2(checkpoint/name, selected/name)
    inference_config = read(selected/'unsloth_decision_config.json')
    inference_config['lab_version'] = 'v2'
    dump(selected/'unsloth_decision_config.json', inference_config)
    dump(selected/'inference_schema.json', {'actions': ACTIONS, 'questions': questions()})
    card = f'''# Tiny Visual Reasoning Lab v2 — all supported tasks trained

Base: {config['base_model']}, revision {config['base_revision']}.
This archive contains LoRA, a Clef selection head, processor and scalar calibration.
Base weights download separately from Hugging Face. No dataset images are included.

Continuation from model-v1 (selected parent step 2400), preserving its learned adapter and head.
V2 training: {metrics['train_runtime'] / 60:.1f} minutes on {config['gpu']}.
Completed step {validation[-1]['step']}; released checkpoint selected at step {best['step']} using 336 validation rows.
V2 sample presentations through the selected step: {best['step'] * config['batch'] * config['accum']:,}.
The parent run took 135.7 minutes; its selected checkpoint used 2400 updates. The v2 time is additional training.

37,632 synthetic bilingual instructions over Tiny ImageNet grids (40 classes, six photos per grid).
All 12 supported task families and all six authored forms per language are in training (144 forms).
Number words and conservative directional synonyms provide further surface variation.
Evaluation holds out source photographs; it does not claim novel-task or novel-form generalization.
Trainable logical parameters: {config['trainable_parameters']:,} / {config['total_parameters']:,}
({100 * config['trainable_parameters'] / config['total_parameters']:.4f}%).
Base weights remain frozen; LoRA and the selection head are trained.

Output: A–F or NONE. This model does not generate a reasoning explanation.
Reproducible data generation, exact training records, before/after evaluation and limitations:
https://github.com/gisbi-kim/tiny-visual-reasoning-lab

Project/Clef code and weights: Apache-2.0, subject to base-model terms.
Image rights remain with original owners; data downloads separately from Stanford.
'''
    (selected/'README.md').write_text(card, encoding='utf-8')
    for name in ('LICENSE', 'CLEF_LICENSE.txt'):
        shutil.copy2(ROOT/name, selected/name)
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(selected.iterdir())}
    archive = ROOT/'dist/tiny-visual-reasoning-4b-v2.zip'
    if archive.exists():
        raise RuntimeError('Preserve the previous v2 archive')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in files:
            z.write(selected/name, name)
    release = dict(repository='gisbi-kim/tiny-visual-reasoning-lab', tag='model-v2',
                   url=f'https://github.com/gisbi-kim/tiny-visual-reasoning-lab/releases/download/model-v2/{archive.name}',
                   bytes=archive.stat().st_size, sha256=sha256(archive), files=files,
                   base_model=config['base_model'], base_revision=config['base_revision'],
                   parent_release='model-v1', created_at=datetime.now(timezone.utc).isoformat())
    dump(ROOT/'docs/release-v2.json', release)
    for source, name in [('config.json', 'training-config.json'), ('metrics.json', 'training-metrics.json'),
                         ('history.json', 'training-history.json'), ('encoding.json', 'encoding.json')]:
        dump(ROOT/'docs/v2'/name, read(run/source))
    dump(ROOT/'docs/v2/validation-history.json', [{'step': v['step'], 'accuracy': v['accuracy']} for v in validation])
    print('V2_RELEASE_READY', archive, archive.stat().st_size, flush=True)

if __name__ == '__main__':
    main()
