"""Write a human-readable report from completed run/evaluation artifacts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def percent(value):return f'{100*value:.2f}%'

def main():
    config=read('docs/training-config.json');train=read('docs/training-metrics.json')
    report=read('results/report.json');validation=read('docs/validation-history.json')
    # PIL opens lazily: decoding triggered by convert() is inside predict timing.
    report['latency_scope']='Synchronized DecisionEngine.predict: processor, transfer, backbone, head and probabilities; includes any lazy pixel read/decode inside the call. Image.open is outside the timer.'
    (ROOT/'results/report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    best=max(validation,key=lambda x:x['accuracy']);release=read('docs/release.json')
    fraction=config['trainable_parameters']/config['total_parameters']
    rows=[]
    labels={'test_id':'새 사진','test_language':'새 사진 + 보류한 문장 표현','test_composition':'새 사진 + 보류한 조건 조합','test_combined':'새 사진 + 보류한 표현·조건 조합'}
    for key,m in report['suites'].items():
        ci=m['scene_bootstrap_ci95']
        rows.append(f"| {labels.get(key,key)} | {m['count']:,} | {percent(m['accuracy'])} | {percent(ci[0])}–{percent(ci[1])} |")
    family_names={'direct':'대상 찾기','neighbor':'인접 위치','row_filter':'같은 행의 범주','col_filter':'같은 열의 범주',
        'neighbor_filter':'인접 위치 + 범주','chain':'두 단계 이동','count_branch':'개수 기준 분기',
        'compare_count':'범주별 개수 비교','row_except':'같은 행에서 제외','neighbor_row_filter':'이동한 행에서 범주 찾기'}
    low=sorted(report['suites']['test_id']['condition_accuracy'].items(),key=lambda item:item[1])[:3]
    difficult=' / '.join(f"{family_names.get(k,k)} {percent(v)}" for k,v in low)
    composition=report['suites']['test_composition']['condition_accuracy']
    diagnostic=read('results/diagnostics.json');freeform=read('results/freeform.json')
    backends={name:read(f'results/{name}-reload.json') for name in ('native','cuda','cpu')}
    def answer(row):return row['prediction'].get('action',row['prediction'].get('answer'))
    reference=backends['cuda']['rows']
    reload_report={'scope':'First 16 fixed validation-selection examples, not a separate accuracy benchmark.',
        'count':len(reference),'accuracy':{name:b['accuracy'] for name,b in backends.items()},
        'native_cuda_top1_agreement':sum(answer(a)==answer(b) for a,b in zip(backends['native']['rows'],reference))/len(reference),
        'cpu_cuda_top1_agreement':sum(answer(a)==answer(b) for a,b in zip(backends['cpu']['rows'],reference))/len(reference),
        'native_selection_accuracy_140':best['accuracy'],'portable_validation_accuracy_140':diagnostic['original_accuracy'],
        'note':'Native uses uncalibrated scores; portable CUDA/CPU apply the saved scalar calibration. Final test scores use portable CUDA. Physical MPS not tested.'}
    (ROOT/'docs/reload-final.json').write_text(json.dumps(reload_report,indent=2),encoding='utf-8')
    freeform_failures='\n'.join(f"- 지시: {r['instruction']} → 정답 `{r['gold']}`, 예측 `{r['prediction']['action']}`" for r in freeform['rows'] if r['gold']!=r['prediction']['action'])
    text=f'''# 학습·평가 기록

## 실제 학습

2026-11-03, Windows의 **{config['gpu']} (16 GB)** 한 장에서 로컬 학습했습니다. CPU는 Ryzen 7 9800X3D, 시스템 RAM은 약 64 GB입니다. 기존 MNIST 프로젝트의 소스/가중치는 수정하지 않았으며, 설치되어 있던 학습 Python 환경을 재사용했습니다. 클라우드 GPU나 유료 데이터 생성 API를 사용하지 않았습니다.

- 기본 모델: `{config['base_model']}`, 고정 revision `{config['base_revision']}`.
- 본학습 경과 시간: **{train['train_runtime']/60:.1f}분**. 고정 검증 실행·체크포인트 저장을 포함한 trainer 실행 시간이며, 사전 다운로드·데이터 생성·최종 평가는 제외합니다.
- 완료 step: **{validation[-1]['step']:,}**, batch {config['batch']}, gradient accumulation {config['accum']}. 학습 예제 {config['train_rows']:,}개.
- 선택한 체크포인트: **step {best['step']:,}**, 검증 140개 정확도 {percent(best['accuracy'])}. 동일한 최고 점수면 먼저 저장된 체크포인트를 유지합니다. 이 값은 최종 test 점수가 아닙니다.
- 전체 run의 예제 제시 횟수는 {validation[-1]['step']*config['batch']*config['accum']:,}회, 배포 체크포인트 시점까지는 {best['step']*config['batch']*config['accum']:,}회입니다. 데이터셋 크기와 제시 횟수는 다릅니다. 중간 체크포인트가 선택됐다면 이후 학습한 step은 배포 가중치에 포함되지 않습니다.
- NF4 4bit + BF16 연산, LoRA rank {config['lora_rank']} / alpha {config['lora_alpha']}, dropout 0. 기본 모델의 기존 가중치는 고정합니다.
- LoRA learning rate {config['lr']}, head learning rate {config['head_lr']}, AdamW, cosine schedule, warmup 60 steps, seed {config['seed']}.
- 학습 시 최대 PyTorch 할당 **{train['peak_allocated_gb']:.2f} GiB**, 최대 예약 **{train['peak_reserved_gb']:.2f} GiB**. 다른 앱과 디스플레이가 쓰는 GPU 메모리는 이 수치에 포함되지 않습니다.

## 몇 %를 학습했나

| 항목 | 논리 파라미터 수 |
|---|---:|
| 학습한 LoRA | {config['lora_trainable_parameters']:,} |
| 학습한 Clef decision head | {config['head_parameters']:,} |
| 학습한 합계 | {config['trainable_parameters']:,} |
| 기본 모델·LoRA·head 전체 | {config['total_parameters']:,} |

전체의 **{percent(fraction)}**가 학습 대상입니다. 4bit 압축 텐서의 저장 원소 수를 분모로 쓰면 학습 비율이 잘못 커지므로 PEFT의 논리 파라미터 수를 사용했습니다. Unsloth 콘솔에 표시되는 압축 저장량 기준 비율과 다른 이유입니다. “4B”는 기본 모델의 제품명이며 위 표는 이 로딩 구성에서의 실제 계산값입니다.

## 최종 평가

최종 test는 새 Python 프로세스에서 저장한 가중치를 일반 Transformers + PEFT + Clef 참조 구현으로 다시 읽어 실행했습니다. 검증으로 선택한 가중치에 별도의 보정용 560개로 temperature만 맞췄습니다. 최종 test로 가중치나 temperature를 선택하지 않았습니다.

| 평가 | 문제 수 | 정확도 | 장면 단위 bootstrap 95% 구간 |
|---|---:|---:|---:|
{chr(10).join(rows)}

네 test suite는 같은 최종 사진 풀을 공유하며, 한 사진이 여러 장면에 재사용됩니다. 위 구간은 장면 단위로 resampling하므로 사진 재사용에 의한 의존성을 완전히 반영하지 못합니다. 언어·작업군별 점수와 혼동행렬, NONE precision/recall, 대조쌍, 확률 보정, 지연 시간은 [기계 판독 결과](../results/report.json)를 확인하세요.

새 사진 테스트에서 정확도가 낮았던 세 작업군은 **{difficult}**입니다. 전체 점수만으로 모든 종류의 추론을 동일하게 잘한다고 해석하면 안 됩니다. `results/failure-examples.json`에는 각 test suite에서 처음 나온 오답 세 개를 원본 사진 ID·지시·실제 확률과 함께 기록합니다.

보류한 조합도 난이도 차이가 큽니다. **이동한 칸의 열에서 범주 찾기: {percent(composition['neighbor_col_filter'])}**, **같은 행에서 범주를 찾은 뒤 이동하기: {percent(composition['row_filter_then_neighbor'])}**입니다. 후자의 새로운 작업 순서에서는 일반화가 특히 약했습니다. 자유로운 표현·새로운 조건 조합을 안정적으로 해결하는 범용 추론 모델로 해석하면 안 됩니다.

## 파일은 어디서 받나

| 파일 | 출처 | 다운로드 크기 |
|---|---|---:|
| 기본 Qwen 모델·processor 원본 | Hugging Face `Qwen/Qwen3.5-4B`, 위 revision | 9,342,907,469 bytes |
| 학습한 LoRA + Clef head + processor + 보정값 | [이 프로젝트 Release]({release['url']}) | {release['bytes']:,} bytes |
| 원본 Tiny ImageNet | Stanford CS231n | 248,100,043 bytes |

첫 실행 이후 빨라지는 것은 기본 모델과 학습 가중치를 로컬에 캐시하기 때문입니다. 기본 모델 전체를 다시 학습하거나 앱 시작 때 학습하는 방식이 아닙니다. 학습 가중치는 `download_model.py`가 archive 및 개별 파일 SHA-256을 검사합니다. 모델 원본 파일 hash는 `base-model-files.json`, release hash는 `release.json`에 있습니다.

별도 환경변수를 지정하지 않았다면 기본 모델은 사용자 홈의 `~/.cache/huggingface/hub`에, 이 프로젝트의 학습 가중치는 프로젝트 안 `weights/`에 저장됩니다. `HF_HOME`을 설정하면 기본 모델 캐시 위치가 달라집니다.

## 재현·검증 범위

실제 입력 의존성도 점검했습니다. 고정 검증 140개에서 배포 코드 정확도는 {percent(diagnostic['original_accuracy'])}, 사진을 지운 뒤 **원래 정답과 비교한 비율**은 {percent(diagnostic['blank_image_original_label_accuracy'])}, 구체 지시를 일반적인 요청으로 바꾼 경우는 {percent(diagnostic['generic_instruction_original_label_accuracy'])}였습니다. 뒤의 두 값은 빈 이미지나 불충분한 지시 자체의 정답률이 아니라, 기존 입력에 의존하는지 보는 진단입니다. 사진의 열을 바꿔 정답이 달라진 {diagnostic['changed_gold_cases']}개 사례에서 원본·변경본을 모두 맞힌 비율은 {percent(diagnostic['changed_gold_both_correct'])}입니다. [전체 진단](../results/diagnostics.json).

한 가지 예제 사진판에 사전 작성한 구어체 지시 14개에서는 {sum(r['gold']==r['prediction']['action'] for r in freeform['rows'])}개를 맞혔습니다. 작은 시연이며 일반 성능 추정치가 아닙니다. 아래 오답도 그대로 공개합니다. [모든 입력·실제 확률](../results/freeform.json).

{freeform_failures}

최종 가중치를 새 프로세스로 불러온 검증 16개에서 학습용 코드와 배포 CUDA의 답 일치율은 {percent(reload_report['native_cuda_top1_agreement'])}, CPU와 배포 CUDA는 {percent(reload_report['cpu_cuda_top1_agreement'])}였습니다. 140개 전체에서는 학습용 코드 {percent(best['accuracy'])}, 배포 코드 {percent(diagnostic['original_accuracy'])}로 완전히 같지는 않았습니다. 따라서 최종 test 점수는 실제 배포 코드를 기준으로 제시했습니다. [재로딩 기록](reload-final.json).

- [학습 설정](training-config.json), [학습 시간·메모리](training-metrics.json), [검증 추이](validation-history.json), [데이터·코드 hash](provenance.json).
- [데이터 구성과 한계](DATA_CARD.md), [실제 encoder 입력 검사](../results/encoder-audit.json).
- [공개 v1 다운로드·설치·HTTP 검사](INSTALL_CHECK.md). 새 clone과 빈 모델 캐시를 사용했으며, Python 패키지 환경은 재사용했습니다.
- [저장 후 GPU 재로딩 시험](reload-smoke.json), [CPU 재로딩 시험](cpu-reload-smoke.json). 16개 overfit 시험의 100%는 배포 경로 확인용이며 일반화 성능이 아닙니다.
- Mac MPS는 이 장비에서 검증하지 못했습니다. CPU/MPS는 float32 기본 모델을 사용하므로 16 GB Mac에서의 실행을 보장하지 않습니다.

이 모델은 여섯 사진의 제한된 선택 문제를 학습한 실험입니다. 자연어 틀과 의미 조합을 다양하게 만들었지만, 일반적인 장문 추론이나 모든 물체·표현에 대한 정확성을 보장하지 않습니다. 이미지 자체가 모호할 수 있고 Qwen 사전학습의 ImageNet 노출 여부도 알 수 없습니다.
'''
    (ROOT/'docs/TRAINING.md').write_text(text,encoding='utf-8')
    headline=f'''실제 학습: **{config['gpu']}에서 {train['train_runtime']/60:.1f}분**, 검증으로 선택한 step **{best['step']:,}**. 전체 논리 파라미터의 **{percent(fraction)}**를 학습했습니다.

| 평가 | 문제 수 | 정확도 | 장면 단위 bootstrap 95% 구간 |
|---|---:|---:|---:|
{chr(10).join(rows)}

[학습·가중치 출처·검증 범위](docs/TRAINING.md) · [전체 평가 수치](results/report.json) · [실패 예제](results/failure-examples.json)
'''
    readme=ROOT/'README.md';body=readme.read_text(encoding='utf-8')
    before,tail=body.split('<!-- RESULTS_START -->',1);_,after=tail.split('<!-- RESULTS_END -->',1)
    body=before+'<!-- RESULTS_START -->\n'+headline+'<!-- RESULTS_END -->'+after
    body=body.replace('를 실제로 학습합니다.','를 실제로 학습한 실험입니다.')
    readme.write_text(body,encoding='utf-8')
    failures=[]
    for suite in report['suites']:
        predictions=read(f'results/{suite}_predictions.json')
        wrong=[p for p in predictions if p['action']!=p['prediction']['action']]
        for row in wrong[:3]:
            failures.append({'suite':suite,**{key:row[key] for key in ('scene_id','source_ids','boxes','object_keys','instruction','program','action','prediction')}})
    (ROOT/'results/failure-examples.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
    print('WROTE docs/TRAINING.md and first three failures per suite (no cherry-picking)')

if __name__=='__main__':main()
