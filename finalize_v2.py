"""Wait for v2, evaluate both versions on identical rows, and prepare a local release."""
import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATUS = ROOT/'runs/v2-finalize-status.json'

def status(phase, **extra):
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(phase=phase, at=datetime.now(timezone.utc).isoformat(), **extra), indent=2), encoding='utf-8')
    temp.replace(STATUS)

def main():
    os.environ['PYTHONUTF8'] = '1'
    start = time.monotonic()
    status('waiting_for_training')
    while True:
        path = ROOT/'runs/v2/status.json'
        if path.exists() and json.loads(path.read_text())['phase'] == 'trained':
            break
        if time.monotonic() - start > 12000:
            raise TimeoutError('Training did not finish within the monitor limit')
        time.sleep(20)
    time.sleep(5)
    (ROOT/'results/v2').mkdir(parents=True, exist_ok=True)
    tasks = [
        ('baseline', ['evaluate.py', '--checkpoint', 'weights', '--device', 'cuda', '--warmup-split', 'v2_validation', '--suites', 'v2_test', '--output', 'results/v2/baseline_v1']),
        ('evaluation', ['evaluate.py', '--checkpoint', 'runs/v2/best', '--device', 'cuda', '--calibrate', '--calibration-split', 'v2_calibration', '--warmup-split', 'v2_validation', '--suites', 'test_language', 'test_composition', 'test_combined', 'test_id', 'v2_test', 'v2_validation_select', '--output', 'results/v2/finetuned']),
        ('freeform', ['check_freeform.py', '--checkpoint', 'runs/v2/best', '--output', 'results/v2/freeform.json']),
    ]
    for name, backend, device in [('native', 'unsloth', 'cuda'), ('cuda', 'portable', 'cuda'), ('cpu', 'portable', 'cpu')]:
        tasks.append((f'{name}_reload', ['check_backend.py', '--checkpoint', 'runs/v2/best', '--backend', backend, '--device', device, '--split', 'v2_reload_probe', '--count', '24', '--output', f'results/v2/{name}-reload.json']))
    tasks += [('export', ['export_v2.py']), ('report', ['write_v2_report.py'])]
    for name, args in tasks:
        status(name)
        print('START', name, flush=True)
        with (ROOT/f'v2-{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', *args], cwd=ROOT, env=os.environ, stdout=log, stderr=subprocess.STDOUT, check=True)
        print('DONE', name, flush=True)
    status('ready_for_review')
    print('V2_READY_FOR_REVIEW', flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=str(exc))
        raise
