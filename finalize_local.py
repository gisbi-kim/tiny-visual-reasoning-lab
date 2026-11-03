"""Wait for a completed local run, then evaluate and package it sequentially."""
import json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
STATUS=ROOT/'runs/finalize-status.json'
def status(phase,**extra):
    temporary=STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps({'phase':phase,'at':datetime.now(timezone.utc).isoformat(),**extra},indent=2),encoding='utf-8')
    temporary.replace(STATUS)
def main():
    os.environ['PYTHONUTF8']='1';start=time.monotonic();status('waiting_for_training')
    while True:
        path=ROOT/'runs/main/status.json'
        if path.exists() and json.loads(path.read_text())['phase']=='trained':break
        if time.monotonic()-start>12000:raise TimeoutError('Training did not finish within the monitor limit')
        time.sleep(20)
    tasks=[
        ('evaluation',['evaluate.py','--checkpoint','runs/main/best','--device','cuda','--calibrate']),
        ('diagnostics',['diagnostics.py','--checkpoint','runs/main/best']),
        ('freeform',['check_freeform.py','--checkpoint','runs/main/best']),
        ('native_reload',['check_backend.py','--checkpoint','runs/main/best','--backend','unsloth','--split','validation_select','--output','results/native-reload.json']),
        ('portable_cuda_reload',['check_backend.py','--checkpoint','runs/main/best','--backend','portable','--device','cuda','--split','validation_select','--output','results/cuda-reload.json']),
        ('portable_cpu_reload',['check_backend.py','--checkpoint','runs/main/best','--backend','portable','--device','cpu','--split','validation_select','--output','results/cpu-reload.json']),
        ('export',['export_release.py']),
        ('report',['write_report.py']),
    ]
    for name,args in tasks:
        status(name);print('START',name,flush=True)
        with (ROOT/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',*args],cwd=ROOT,env=os.environ,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',name,flush=True)
    status('ready_for_review');print('LOCAL_EVALUATION_AND_PACKAGE_READY',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=str(exc));raise
