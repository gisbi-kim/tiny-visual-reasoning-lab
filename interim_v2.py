"""Optional CPU-only spot check while the independent GPU training continues."""
from collections import defaultdict
from datetime import datetime, timezone
import argparse, json, os, random, shutil, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITES = ('test_language', 'test_composition', 'test_combined')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def select_rows(rows, count=56):
    """Fixed seed, round-robin family/language strata; selection ignores model answers."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row['family'], row['language'])].append(row)
    rng = random.Random(85307)
    keys = sorted(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < min(count, len(rows)):
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop())
            if len(selected) == min(count, len(rows)):
                break
    return selected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=56)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    from download_model import sha256
    status_path = ROOT/'runs/v2/interim-status.json'
    run = ROOT/'runs/v2'
    phase = read(run/'status.json')['phase']
    if phase != 'training':
        raise RuntimeError('Start an interim snapshot only while training, outside checkpoint validation/save')
    history = read(run/'validation.json')
    best = max(history, key=lambda x: x['accuracy'])
    step = best['step']
    checkpoint = run/f'interim/step-{step}/weights'
    if checkpoint.exists():
        raise RuntimeError('This snapshot already exists; preserve its prior result')
    destination = ROOT/f'results/v2/interim-step-{step}.json'
    def status(phase, **extra):
        dump(status_path, dict(phase=phase, step=step, at=datetime.now(timezone.utc).isoformat(), output=str(destination), **extra))
    status('snapshot')
    checkpoint.mkdir(parents=True)
    source = run/'best'
    hashes = {p.name: sha256(p) for p in source.iterdir() if p.is_file()}
    for name in hashes:
        shutil.copy2(source/name, checkpoint/name)
    assert hashes == {name: sha256(source/name) for name in hashes}, 'Checkpoint changed during snapshot'
    assert hashes == {name: sha256(checkpoint/name) for name in hashes}, 'Snapshot differs'
    current_best = max(read(run/'validation.json'), key=lambda x: x['accuracy'])
    assert current_best['step'] == step and read(run/'status.json')['phase'] == 'training'
    dump(checkpoint.parent/'snapshot.json', dict(step=step, validation_accuracy=best['accuracy'], files=hashes))
    status('loading_cpu')
    import torch
    torch.set_num_threads(args.threads)
    from PIL import Image
    from engine import DecisionEngine
    start = time.perf_counter()
    engine = DecisionEngine(checkpoint, 'cpu')
    # Use a validation row for warm-up, never the answer to an evaluated row.
    warmup = json.loads((ROOT/'data/v2_validation_select.jsonl').read_text(encoding='utf-8').splitlines()[0])
    with Image.open(ROOT/'data'/warmup['image']) as image:
        engine.predict(image, warmup['instruction'])
    report = dict(step=step, checkpoint=str(checkpoint), selected_validation_accuracy=best['accuracy'],
                  scope='Interim spot check only, not the full 1,120-row suites. Fixed seeded family/language-stratified subset; no test-driven weight selection or tuning. V1 predictions use saved portable CUDA, interim V2 uses portable CPU float32. Final comparison uses full identical suites and CUDA for both models.',
                  per_suite=args.count, device='cpu', threads=args.threads,
                  created_at=datetime.now(timezone.utc).isoformat(), suites={})
    for suite in SUITES:
        original = read(ROOT/f'results/{suite}_predictions.json')
        chosen = select_rows(original, args.count)
        outputs = []
        for i, row in enumerate(chosen):
            with Image.open(ROOT/'data'/row['image']) as image:
                prediction = engine.predict(image, row['instruction'])
            outputs.append(dict(scene_id=row['scene_id'], source_ids=row['source_ids'],
                                instruction=row['instruction'], gold=row['action'],
                                v1_prediction=row['prediction']['action'], prediction=prediction))
            if (i+1) % 8 == 0:
                status('evaluating_cpu', suite=suite, completed=i+1, count=len(chosen))
                print(suite, i+1, len(chosen), flush=True)
        v1 = sum(r['gold'] == r['v1_prediction'] for r in outputs)/len(outputs)
        v2 = sum(r['gold'] == r['prediction']['action'] for r in outputs)/len(outputs)
        report['suites'][suite] = dict(count=len(outputs), v1_subset_accuracy=v1,
                                      v2_subset_accuracy=v2, delta_percentage_points=100*(v2-v1), rows=outputs)
        report['elapsed_seconds'] = time.perf_counter()-start
        dump(destination, report)
    status('complete', elapsed_seconds=time.perf_counter()-start)
    print('INTERIM_COMPLETE', destination, flush=True)

if __name__ == '__main__':
    main()
