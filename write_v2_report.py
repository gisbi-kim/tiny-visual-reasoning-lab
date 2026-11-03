"""Report v1/v2 on identical photo-held-out rows; keep historical v1 reports intact."""
from collections import defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))

def dump(path, value):
    target = ROOT/path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def pct(value):
    return f'{100 * value:.2f}%'

def answer(row):
    return row['prediction'].get('action', row['prediction'].get('answer'))

def original_suite_comparison(after_report):
    original = read('results/report.json')['suites']
    labels = {'test_language': '새 사진 + v1에서 보류했던 표현',
              'test_composition': '새 사진 + v1에서 보류했던 조건 조합',
              'test_combined': '새 사진 + v1에서 보류했던 표현·조합',
              'test_id': '새 사진 + v1에서도 학습한 유형'}
    result = {}
    for suite, label in labels.items():
        old = read(f'results/{suite}_predictions.json')
        new = read(f'results/v2/finetuned/{suite}_predictions.json')
        assert len(old) == len(new) == 1120
        for a, b in zip(old, new):
            for key in ('scene_id', 'source_ids', 'image', 'instruction', 'action', 'program', 'variant'):
                assert a[key] == b[key], f'Original suite rows differ: {suite}/{key}'
        v1 = sum(r['action'] == answer(r) for r in old)/len(old)
        v2 = sum(r['action'] == answer(r) for r in new)/len(new)
        assert abs(v1-original[suite]['accuracy']) < 1e-12
        assert abs(v2-after_report['suites'][suite]['accuracy']) < 1e-12
        result[suite] = dict(label=label, count=len(new), v1_accuracy=v1, v2_accuracy=v2,
                             delta_percentage_points=100*(v2-v1))
    dump('results/v2/original-suite-comparison.json', {
        'scope': 'Exactly the original v1 test rows and images. Forms/compositions withheld in v1 are trained in v2, so the v2 scores no longer measure unseen-form/task generalization. Test photos remain excluded from training.',
        'suites': result})
    return result

def main():
    before = read('results/v2/baseline_v1/report.json')['suites']['v2_test']
    after_report = read('results/v2/finetuned/report.json')
    original_suites = original_suite_comparison(after_report)
    original_table = '\n'.join(f"| {r['label']} | {pct(r['v1_accuracy'])} | {pct(r['v2_accuracy'])} | {r['delta_percentage_points']:+.2f}%p |" for r in original_suites.values())
    after = after_report['suites']['v2_test']
    old_rows = read('results/v2/baseline_v1/v2_test_predictions.json')
    new_rows = read('results/v2/finetuned/v2_test_predictions.json')
    assert len(old_rows) == len(new_rows) == 2016
    for a, b in zip(old_rows, new_rows):
        for key in ('scene_id', 'source_ids', 'instruction', 'action', 'program', 'variant'):
            assert a[key] == b[key], f'Before/after rows differ: {key}'
    transitions = defaultdict(int)
    forms = defaultdict(lambda: [0, 0, 0])
    for a, b in zip(old_rows, new_rows):
        old_ok, new_ok = a['action'] == answer(a), b['action'] == answer(b)
        transitions[f'v1_{"correct" if old_ok else "wrong"}_v2_{"correct" if new_ok else "wrong"}'] += 1
        group = forms[f"{b['language']}|form_{b['variant']}"]
        group[0] += 1; group[1] += old_ok; group[2] += new_ok
    comparison = dict(count=len(new_rows), v1_accuracy=before['accuracy'], v2_accuracy=after['accuracy'],
                      delta_percentage_points=100 * (after['accuracy'] - before['accuracy']),
                      transitions=dict(transitions), condition_accuracy={
                          f: {'v1': before['condition_accuracy'][f], 'v2': after['condition_accuracy'][f]}
                          for f in after['condition_accuracy']},
                      language_accuracy={l: {'v1': before['language_accuracy'][l], 'v2': after['language_accuracy'][l]}
                                         for l in after['language_accuracy']},
                      form_accuracy={k: {'count': n, 'v1': a/n, 'v2': b/n} for k, (n, a, b) in forms.items()},
                      scope='Identical v2 test rows, all instruction families/forms trained in v2; held-out source photos. Not a novel-form/task generalization estimate.')
    comparison['original_v1_suites'] = original_suites
    dump('results/v2/comparison.json', comparison)
    failures = [r for r in new_rows if r['action'] != answer(r)]
    dump('results/v2/failure-examples.json', failures[:20])
    config = read('docs/v2/training-config.json')
    train = read('docs/v2/training-metrics.json')
    validation = read('docs/v2/validation-history.json')
    best = max(validation, key=lambda x: x['accuracy'])
    release = read('docs/release-v2.json')
    free_old, free_new = read('results/freeform.json'), read('results/v2/freeform.json')
    for a, b in zip(free_old['rows'], free_new['rows']):
        assert a['instruction'] == b['instruction'] and a['gold'] == b['gold']
    backends = {k: read(f'results/v2/{k}-reload.json') for k in ('native', 'cuda', 'cpu')}
    reference = backends['cuda']['rows']
    for k, result in backends.items():
        assert len(result['rows']) == len(reference) == 24
        assert [(r['scene_id'], r['instruction']) for r in result['rows']] == [(r['scene_id'], r['instruction']) for r in reference]
    reload = dict(count=24, accuracy={k: v['accuracy'] for k, v in backends.items()},
                  native_cuda_top1_agreement=sum(answer(a) == answer(b) for a, b in zip(backends['native']['rows'], reference))/24,
                  cpu_cuda_top1_agreement=sum(answer(a) == answer(b) for a, b in zip(backends['cpu']['rows'], reference))/24,
                  native_validation_accuracy_336=best['accuracy'],
                  portable_validation_accuracy_336=after_report['suites']['v2_validation_select']['accuracy'],
                  scope='Fixed validation probe: two examples per family, both languages, all seven actions. Final scores use portable CUDA; physical Mac/MPS not tested.')
    dump('docs/v2/reload-final.json', reload)
    family_names = {'direct': '대상 찾기', 'neighbor': '인접 위치', 'row_filter': '같은 행의 범주',
                    'col_filter': '같은 열의 범주', 'neighbor_filter': '인접 위치 + 범주', 'chain': '두 단계 이동',
                    'count_branch': '개수 기준 분기', 'compare_count': '범주별 개수 비교', 'row_except': '같은 행에서 제외',
                    'neighbor_row_filter': '이동한 칸의 행에서 범주 찾기', 'neighbor_col_filter': '이동한 칸의 열에서 범주 찾기',
                    'row_filter_then_neighbor': '같은 행의 범주를 찾은 뒤 이동'}
    table = '\n'.join(f"| {family_names[f]} | {pct(before['condition_accuracy'][f])} | {pct(after['condition_accuracy'][f])} |" for f in family_names)
    free_failures = '\n'.join(f"- {r['instruction']} → 정답 `{r['gold']}`, 실제 예측 `{answer(r)}`" for r in free_new['rows'] if r['gold'] != answer(r)) or '이 14개에서는 오답이 없었습니다. 다른 입력의 정확성을 보장하는 결과는 아닙니다.'
    text = f'''# v2 학습·평가: 지원하는 표현과 조건을 모두 학습

v1은 일부 표현과 조건 조합을 학습에서 제외하여 일반화를 시험했습니다. v2는 실사용 범위의 정확도를 목표로 **12개 작업군과 언어별 6개 표현(총 144개)을 모두 학습**합니다. 새 사진으로 만든 동일한 2,016개 문제에서 기존 v1과 v2를 직접 비교했습니다.

| 동일한 평가 문제 | v1 | v2 |
|---|---:|---:|
| 전체 2,016개 | {pct(before['accuracy'])} | {pct(after['accuracy'])} |
| 한국어 | {pct(before['language_accuracy']['ko'])} | {pct(after['language_accuracy']['ko'])} |
| 영어 | {pct(before['language_accuracy']['en'])} | {pct(after['language_accuracy']['en'])} |

차이는 **{comparison['delta_percentage_points']:+.2f}%p**입니다. 이전 README의 79.29%·67.77% 등은 다른 평가 데이터의 점수이므로 이 표와 직접 빼서 비교하면 안 됩니다. v1의 기존 가중치·평가 기록은 그대로 보존합니다.

## 이전 79.29%·67.77%·54.82%와 같은 문제로 직접 비교

사용자가 확인하려던 기존 평가 문제도 사진·지시·정답을 바꾸지 않고 그대로 v2에 실행했습니다. 각 행은 같은 1,120개 문제의 전후 비교입니다.

| 기존 평가 구분 | v1 | v2 | 변화 |
|---|---:|---:|---:|
{original_table}

v1에서 보류했던 표현·조합은 이제 v2 학습에 포함됩니다. 따라서 v2의 이 점수는 해당 유형을 학습한 뒤의 개선 폭이며, 여전히 안 배운 표현·조합을 푼 점수로 해석하면 안 됩니다. **평가 사진 자체는 계속 학습에서 제외**했습니다. [같은 문제 여부 검사와 비교 수치](../results/v2/original-suite-comparison.json).

사용자의 중간 비교 요청에 따라 학습 도중 각 항목의 고정 표본 56개를 CPU로 확인하는 절차도 추가했습니다. 이 표본 결과를 보고 학습 설정이나 체크포인트 선택 기준을 바꾸지 않으며, 최종 표는 전체 1,120개를 양쪽 모두 CUDA로 실행한 결과입니다. 중간 표본 점수와 전체 점수를 혼용하지 않습니다.

## 실제 학습과 가중치 계보

- 기본 모델: `{config['base_model']}`, revision `{config['base_revision']}`. 원본은 Hugging Face에서 별도로 다운로드합니다.
- 시작점: v1에서 검증으로 선택한 step 2,400의 LoRA와 Clef head. 새로운 adapter/head를 무작위 초기화하지 않고 이어서 학습했습니다. [496개 LoRA 텐서의 실제 갱신 확인](v2-continuation-smoke.json).
- v2 추가 학습 시간: **{train['train_runtime']/60:.1f}분**, **{config['gpu']} (16 GB)** 한 장. Windows / Ryzen 7 9800X3D / RAM 약 64 GB. 클라우드 GPU·유료 데이터 생성 API를 사용하지 않았습니다.
- 완료 step **{validation[-1]['step']:,}**; 배포에 선택한 step **{best['step']:,}**. 선택용 검증 336개에서 {pct(best['accuracy'])}; 최종 test로 선택하지 않았습니다. 동률이면 앞선 체크포인트를 유지합니다.
- batch {config['batch']}, accumulation {config['accum']}; v2 학습 데이터 **{config['train_rows']:,}개**. 전체 run 제시 횟수 {validation[-1]['step'] * config['batch'] * config['accum']:,}회, 선택한 체크포인트까지 {best['step'] * config['batch'] * config['accum']:,}회입니다. 선택 이후의 업데이트는 배포 가중치에 포함되지 않습니다.
- v1의 전체 run은 별도로 135.7분이었으며 선택된 부모 가중치에는 처음 2,400 step만 반영됐습니다. 위 v2 시간은 추가 학습 시간입니다.
- NF4 4bit 기본 모델 고정, BF16 계산; LoRA rank/alpha 16/16, LoRA LR {config['lr']}, head LR {config['head_lr']}, AdamW, cosine schedule, seed {config['seed']}.
- 학습 대상: LoRA **{config['lora_trainable_parameters']:,}**, Clef head **{config['head_parameters']:,}**, 합계 **{config['trainable_parameters']:,} / {config['total_parameters']:,} = {pct(config['trainable_parameters']/config['total_parameters'])}**. 압축 저장 크기가 아닌 논리 파라미터 수입니다.
- 최대 PyTorch 할당 {train['peak_allocated_gb']:.2f} GiB / 예약 {train['peak_reserved_gb']:.2f} GiB. 디스플레이·다른 앱의 GPU 사용량은 별도입니다.

## 데이터와 평가 범위

37,632개 지시, 4,825개 격자, 12,830개 원본 사진, 서로 다른 문장 36,328개를 사용했습니다. 기존에 보류했던 표현과 두 조건 조합도 전부 학습에 포함합니다. `3장 / 세 장`, `3 / three`, `왼쪽 / 좌측` 등의 표기도 함께 사용합니다. 문장 틀은 코딩 에이전트가 작성한 합성 데이터이며, 별도의 사람 검수 데이터가 아닙니다.

학습·검증·보정·평가의 원본 사진 ID와 RGB hash는 분리했습니다. **기존 test 사진을 학습에 넣지 않았습니다.** 검증 336개로 체크포인트를 고르고, 별도 보정용 1,008개로 temperature만 맞췄습니다. v1과 v2를 같은 v2 test 2,016개에 배포 코드로 실행했습니다. [데이터 상세와 검사](V2_DATA_CARD.md).

평가는 **학습한 지시 유형을 처음 보는 사진에 적용하는 능력**을 측정합니다. 모든 가능한 자연어·모든 추론을 학습했다는 뜻은 아닙니다. 평가 원본 사진 풀은 v1에서 사용한 것과 같고 v1의 결과가 v2 개발에 영향을 주었으므로 완전히 새로운 비공개 시험으로 주장하지 않습니다. Qwen 사전학습의 ImageNet 노출 여부, 근접 중복 여부는 알 수 없습니다.

| 조건 유형 (각 168개) | v1 | v2 |
|---|---:|---:|
{table}

v1 오답→v2 정답 {transitions['v1_wrong_v2_correct']}개, v1 정답→v2 오답 {transitions['v1_correct_v2_wrong']}개입니다. [언어·표현·유형별 비교](../results/v2/comparison.json), [전체 수치](../results/v2/finetuned/report.json), [순서상 처음 나온 오답 20개](../results/v2/failure-examples.json)를 공개합니다. 사진이 여러 격자에 반복 사용되므로 전체 문제 수를 독립적인 사진 수로 해석하면 안 됩니다.

## 실제 구어체 시연과 재로딩

기존에 작성한 동일한 사진판·구어체 14개에서 v1 {sum(r['gold'] == answer(r) for r in free_old['rows'])}/14 → v2 {sum(r['gold'] == answer(r) for r in free_new['rows'])}/14였습니다. 이미 확인한 예제이고 하나의 사진판만 사용하므로 독립적인 성능 추정치는 아닙니다. [모든 실제 답과 확률](../results/v2/freeform.json).

{free_failures}

12개 유형을 모두 포함한 고정 검증 24개를 새 프로세스로 재로딩했습니다. 학습용 코드와 배포 CUDA 답 일치율 {pct(reload['native_cuda_top1_agreement'])}, CPU와 배포 CUDA 일치율 {pct(reload['cpu_cuda_top1_agreement'])}. 전체 검증 336개에서는 학습용 코드 {pct(best['accuracy'])}, 배포 CUDA {pct(reload['portable_validation_accuracy_336'])}입니다. 최종 점수는 실제 배포 코드 기준입니다. [재로딩 기록](v2/reload-final.json).

Mac/MPS 실기는 검증하지 못했습니다. CPU/MPS는 float32 기본 모델을 읽어 모델 자체만 약 18.3 GB가 필요합니다. 기존 v1의 빈 모델 캐시 다운로드 검증은 [INSTALL_CHECK.md](INSTALL_CHECK.md)에 있으며, v2는 같은 고정 기본 모델을 사용합니다.

## 다운로드와 재현

배포 파일: [model-v2]({release['url']}) (**{release['bytes']:,} bytes**, SHA-256 `{release['sha256']}`). 학습한 LoRA·head·processor·보정값만 포함하며 기본 모델과 사진은 별도 다운로드합니다. 첫 실행 후 빨라지는 이유는 로컬 캐시 재사용입니다.

```bash
# v1을 보관하고 v2 설치
python download_model.py --version v2 --output weights-v2
python app.py --checkpoint weights-v2 --device cuda

# v2 학습 재현: NVIDIA CUDA 학습 환경 필요
python fetch_assets.py
python prepare_data.py
python prepare_v2.py
python download_model.py --version v1 --output weights-v1
python train.py --run reproduction-v2 --init-checkpoint weights-v1 --train-split v2_train --validation-split v2_validation_select --steps 4704 --seconds 9900 --batch 8 --accum 1 --val-count 336 --validate-every 400 --four-bit --lr 0.0001 --head-lr 0.00005
python evaluate.py --checkpoint runs/reproduction-v2/best --device cuda --calibrate --calibration-split v2_calibration --warmup-split v2_validation --suites v2_test --output results/reproduction-v2
```

[학습 설정](v2/training-config.json) · [시간·메모리](v2/training-metrics.json) · [검증 추이](v2/validation-history.json) · [데이터·부모 가중치 hash](v2/provenance.json) · [과거 v1 학습 기록](TRAINING.md).
'''
    (ROOT/'docs/V2_TRAINING.md').write_text(text, encoding='utf-8')
    notes = f'''All 12 supported task families and all 144 authored Korean/English forms are now trained, including those held out in v1. V2 continues the v1 adapter/head for {train['train_runtime']/60:.1f} minutes on one RTX 4070 Ti SUPER.

On the same 2,016 photo-held-out examples: v1 {pct(before['accuracy'])} → v2 {pct(after['accuracy'])}. This tests trained instruction types on new photographs, not novel-task generalization. Detailed family scores, failures, provenance and limitations are in docs/V2_TRAINING.md.

The archive contains trained LoRA, the Clef head, processor and calibration. Qwen/Qwen3.5-4B at revision {config['base_revision']} downloads separately from Hugging Face; Tiny ImageNet downloads separately from Stanford. V1 remains available as model-v1.

Download: `python download_model.py --version v2 --output weights-v2`
Run: `python app.py --checkpoint weights-v2 --device cpu` (or cuda).
'''
    (ROOT/'docs/V2_RELEASE_NOTES.md').write_text(notes, encoding='utf-8')
    print('WROTE_V2_REPORT', comparison['v1_accuracy'], comparison['v2_accuracy'], flush=True)

if __name__ == '__main__':
    main()
