"""Complete the engineering article with measured final results and source links."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))

def pct(value):
    return f'{100*value:.2f}%'

def main():
    comparison = read('results/v2/original-suite-comparison.json')['suites']
    common = read('results/v2/comparison.json')
    result = read('results/v2/finetuned/report.json')['suites']['v2_test']
    config = read('docs/v2/training-config.json')
    metrics = read('docs/v2/training-metrics.json')
    history = read('docs/v2/validation-history.json')
    selected = max(history, key=lambda row: row['accuracy'])
    release = read('docs/release-v2.json')
    reload = read('docs/v2/reload-final.json')
    calibration = read('runs/v2/best/calibration.json')
    freeform = read('results/v2/freeform.json')
    freeform_v1 = read('results/freeform.json')
    text = (ROOT/'docs/BLOG_DRAFT.md').read_text(encoding='utf-8')
    def paragraph(prefix, replacement):
        nonlocal text
        pieces = text.split('\n\n')
        matches = [i for i, p in enumerate(pieces) if p.startswith(prefix)]
        if len(matches) != 1:
            raise RuntimeError(f'Article paragraph must match once: {prefix}')
        pieces[matches[0]] = replacement
        text = '\n\n'.join(pieces)
    paragraph('> 작성 중인 원고이다.',
        'Tiny ImageNet 사진과 자연어 지시로 작은 선택 문제를 구성하고, Qwen3.5-4B 기반 decision model을 학습한 기록이다. v1에서 드러난 표현·조건 조합의 약점을 v2의 데이터 구성으로 보완하였다. 이 글에서는 데이터가 무엇을 의미하는지부터 학습 대상, 실제 개선 폭, 실패 사례와 재현 방법까지 살펴본다. [코드와 가중치](https://github.com/gisbi-kim/tiny-visual-reasoning-lab), [실험 자료 안내](v2/INDEX.md)를 함께 공개한다.')
    training_table = f'''| 항목 | 실제 기록 |
|---|---|
| 장비 | {config['gpu']} · Windows · RAM 약 64 GB |
| 추가 학습 예제 | {config['train_rows']:,}개 |
| 추가 학습 완료 | {history[-1]['step']:,} step · {metrics['epoch']:.2f} epoch |
| Trainer 실행 시간 | {metrics['train_runtime']/60:.1f}분 |
| 선택한 배포 체크포인트 | v2 step {selected['step']:,} |
| 선택 시점까지 v2 예제 제시 | {selected['step']*config['batch']*config['accum']:,}회 |
| 고정 검증 336개에서 선택 점수 | {pct(selected['accuracy'])} |
| 최대 PyTorch 할당 / 예약 | {metrics['peak_allocated_gb']:.2f} / {metrics['peak_reserved_gb']:.2f} GiB |'''
    paragraph('최종 글에는 실제 run 시간',
        f'''실제 추가 학습은 {metrics['train_runtime']/60:.1f}분 동안 진행되었다. Trainer 기록 기준이며, 사전 다운로드·데이터 생성·최종 평가 시간은 제외한다. v1의 전체 run 135.7분과 별개인 추가 시간이다. 고정 검증 336개에서 가장 높았던 step {selected['step']:,}의 가중치를 배포한다. 최고 점수가 같으면 먼저 저장된 체크포인트를 유지한다.

{training_table}

v1의 부모 체크포인트는 step 2,400에서 선택되어 19,200회의 예제 제시를 반영했다. v2 배포 체크포인트는 거기서 {selected['step']*config['batch']*config['accum']:,}회를 추가로 본 상태이다. 데이터셋이 바뀌었으므로 두 단계의 epoch를 단순히 더하지 않는다. 전체 run이 1 epoch를 완료했더라도 이후 step의 업데이트가 앞서 선택된 배포 가중치에 포함되는 것은 아니다.

![v2의 학습 loss와 이동평균, 고정 검증 정확도의 변화](figures/v2-training-history.svg)

위 그림의 loss는 기록된 logging 간격의 값과 20개 기록의 이동평균이다. 아래 검증 곡선은 학습에 넣지 않은 검증 사진으로 계산했다. 전체 점수만 보고 새로운 조합의 개선을 추정하지 않고, 다음 절에서 실제 같은 평가 문제끼리 비교한다. [설정](v2/training-config.json), [전체 loss 기록](v2/training-history.json), [검증 추이](v2/validation-history.json), [실제 시간·메모리](v2/training-metrics.json).''')
    for key, label in [('test_language', '새 사진 + 같은 뜻을 바꿔 쓴 표현'),
                       ('test_composition', '새 사진 + 새로운 조건 조합'),
                       ('test_combined', '새 사진 + 바꿔 쓴 표현과 새로운 조건 조합')]:
        row = comparison[key]
        old = f"| {label} |{pct(row['v1_accuracy'])}|평가 후 입력|평가 후 입력|"
        new = f"| {label} | {pct(row['v1_accuracy'])} | **{pct(row['v2_accuracy'])}** | **{row['delta_percentage_points']:+.2f}%p** |"
        if old not in text:
            raise RuntimeError(f'Missing comparison row: {key}')
        text = text.replace(old, new)
    paragraph('별도로 모든 유형을 균형 있게 포함한',
        f'''![기존 세 평가 항목의 동일 문제에서 계산한 v1과 v2 정확도](figures/original-suite-comparison.svg)

세 항목은 각각 같은 1,120개 문제를 사용하며, 전체 예측의 사진 ID·문장·정답이 일치하는지 코드로 검사하였다. [기존 문제 전후 비교 JSON](../results/v2/original-suite-comparison.json)과 [문제별 실제 예측](../results/v2/finetuned/)에서 확인할 수 있다.

별도로 모든 유형을 균형 있게 포함한 v2_test 2,016문제에서는 v1 **{pct(common['v1_accuracy'])} → v2 {pct(common['v2_accuracy'])}**였다. 한국어는 {pct(common['language_accuracy']['ko']['v1'])} → {pct(common['language_accuracy']['ko']['v2'])}, 영어는 {pct(common['language_accuracy']['en']['v1'])} → {pct(common['language_accuracy']['en']['v2'])}였다. 이 집계는 위의 세 항목과 다른 문제 집합이다. 서로 다른 표의 전후 점수를 섞어서 개선 폭을 계산하지 않는다. [동일 2,016문제 비교](../results/v2/comparison.json).''')
    names = {'direct': '대상 찾기', 'neighbor': '인접 위치', 'row_filter': '같은 행의 범주',
             'col_filter': '같은 열의 범주', 'neighbor_filter': '인접 위치와 범주', 'chain': '두 단계 이동',
             'count_branch': '개수 기준 분기', 'compare_count': '범주별 개수 비교', 'row_except': '같은 행에서 제외',
             'neighbor_row_filter': '이동한 칸의 행에서 범주 찾기', 'neighbor_col_filter': '이동한 칸의 열에서 범주 찾기',
             'row_filter_then_neighbor': '같은 행의 범주를 찾은 뒤 이동'}
    # Examples come from the very same test set as the per-family scores.
    rows = read('results/v2/finetuned/v2_test_predictions.json')
    instructions = {}
    for row in rows:
        if row['language'] == 'ko':
            instructions.setdefault(row['family'], set()).add(row['instruction'])
    def family_examples(key):
        chosen = sorted(instructions[key], key=lambda value: (len(value), value))[:3]
        assert len(chosen) == 3
        return '<ul>' + ''.join(f'<li>{value.replace("|", "&#124;")}</li>' for value in chosen) + '</ul>'
    families = '\n'.join(f"| {names[k]} | {family_examples(k)} | {pct(v['v1'])} | {pct(v['v2'])} |" for k,v in common['condition_accuracy'].items())
    examples = []
    for suite in ('test_language','test_composition','test_combined'):
        rows = read(f'results/v2/finetuned/{suite}_predictions.json')
        wrong = next((r for r in rows if r['action'] != r['prediction']['action']), None)
        if wrong:
            examples.append(f"- `{suite}`: “{wrong['instruction']}” → 정답 **{wrong['action']}**, 예측 **{wrong['prediction']['action']}**. 사진판 ID `{wrong['scene_id']}`.")
    paragraph('대상 찾기, 두 단계 이동,',
        f'''대상 찾기, 두 단계 이동, 개수 분기, 범주별 개수 비교는 서로 다른 어려움이 있다. 아래 표는 별도 v2_test 2,016문제에서 계산한 조건별 결과이다. 조건마다 168문제이며, 위의 기존 세 항목 표와 구분한다.

| 조건 유형 | 예시 지시 (실제 평가 문장) | v1 | v2 |
|---|---|---:|---:|
{families}

다음은 기존 세 평가 항목에서 순서상 처음 나온 v2 오답이다. 결과를 좋게 보이게 할 목적으로 실패 유형을 골라내지 않았다. 각 문장의 사진판과 정답 정의는 위에서 연결한 전체 예측 파일의 source IDs와 프로그램으로 추적할 수 있다.

{chr(10).join(examples)}

고정 사진판의 같은 구어체 14개에서는 **v1 {sum(r['gold']==r['prediction']['action'] for r in freeform_v1['rows'])}/14 → v2 {sum(r['gold']==r['prediction']['action'] for r in freeform['rows'])}/14**였다. 이 시연에서는 오히려 한 문제를 덜 맞혔다. 하나의 사진판에서 이미 작성해 둔 작은 시연이므로 일반적인 성능 하락을 추정할 근거는 부족하지만, 작성한 표현 틀의 개선을 자유로운 말투 전체의 개선으로 확대해서도 안 된다. 개수와 비교 조건을 구어체로 바꾸었을 때의 오답을 그대로 남겼다. [구어체 실제 예측](../results/v2/freeform.json).

CPU/CUDA/학습용 구현을 별도 프로세스로 다시 불러온 24개 검증 표본에서 CPU–CUDA 답 일치율은 {pct(reload['cpu_cuda_top1_agreement'])}, 학습용–배포 CUDA 일치율은 {pct(reload['native_cuda_top1_agreement'])}였다. 전체 검증 336개에서는 학습용 {pct(reload['native_validation_accuracy_336'])}, 배포 코드 {pct(reload['portable_validation_accuracy_336'])}였다. 구현과 수치 정밀도가 다를 수 있으므로 최종 성능은 실제 배포 CUDA 경로로 계산하였다. [재로딩 검사](v2/reload-final.json).''')
    paragraph('선택 확률은 별도의',
        f'''선택 확률은 별도의 1,008개 보정 데이터로 temperature를 조정했다. 선택한 값은 {calibration['temperature']:.4f}이며, 보정 데이터의 negative log-likelihood는 {calibration['uncalibrated_nll']:.4f} → {calibration['nll']:.4f}였다. 양의 scalar로 점수를 나누므로 가장 큰 점수의 위치는 바뀌지 않는다. 이 보정은 top-1 정확도를 직접 올리는 방법이 아니라 확신의 정도를 조정하는 절차이다. [보정 기록](v2/logs/calibration/v2.json).''')
    paragraph('1,600step 중간 모델의',
        '''1,600 step 중간 모델의 고정 56문제씩에서는 v1 대비 새 표현 76.79% → 96.43%, 새 조합 69.64% → 98.21%, 표현·조합 58.93% → 98.21%로 개선되었다. 작은 표본이고 v2는 CPU, v1은 저장된 CUDA 예측이므로 잠정적인 관찰이었다. 앞 절의 같은 1,120문제 전체 최종 GPU 비교를 주된 근거로 삼고, 중간 표본은 개선 과정을 보여 주는 보조 자료로 남긴다. [1,600 step 표본](../results/v2/interim-step-1600.json), [2,000 step 표본](../results/v2/interim-step-2000.json).''')
    paragraph('첫 실행 후 빨라지는 것은',
        f'''첫 실행 후 빨라지는 것은 저장된 기본 모델과 학습 가중치를 다시 사용하기 때문이다. 실행할 때마다 학습을 다시 하는 구조가 아니다.

| 구성 | 출처와 크기 |
|---|---|
| 기본 VLM | Hugging Face `Qwen/Qwen3.5-4B`, 원본 snapshot 약 9.34 GB |
| 학습한 LoRA·head·processor·보정값 | GitHub `model-v2`, {release['bytes']:,} bytes |
| 원본 사진 데이터 | Stanford Tiny ImageNet, 248,100,043 bytes |

기본 모델 revision은 `{release['base_revision']}`로 고정하였다. 학습한 가중치 archive의 SHA-256은 `{release['sha256']}`이다. [v2 다운로드 manifest](release-v2.json), [기본 모델 파일 hash](base-model-files.json), [v1 보관본](https://github.com/gisbi-kim/tiny-visual-reasoning-lab/releases/tag/model-v1)을 제공한다.

```bash
git clone https://github.com/gisbi-kim/tiny-visual-reasoning-lab.git
cd tiny-visual-reasoning-lab
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python download_model.py --version v2 --output weights-v2
python fetch_assets.py --dataset-only
python prepare_demo.py
python app.py --checkpoint weights-v2 --device cpu
```

Python 3.11 또는 3.12의 가상환경에서 실행한다. NVIDIA CUDA에서는 해당 환경에 맞는 PyTorch와 bitsandbytes를 설치한 뒤 `--device cuda`를 사용한다. 브라우저 주소는 `http://127.0.0.1:7861`이다. 자세한 설치·학습 재현 명령은 [README](../README.md)와 [v2 학습 기록](V2_TRAINING.md)에 있다.''')
    paragraph('본문은 이해를 위한 설명이고,',
        '''본문은 이해를 위한 설명이고, 수치의 근거는 기계 판독 가능한 JSON과 실제 로그이다. [실험 자료 안내](v2/INDEX.md)에는 설정·학습 추이·전체 예측·중간 평가·실패 사례·환경·가중치·검사 기록을 연결하였다. [콘솔 로그와 실행별 기록](v2/logs/), [공개 파일과 원본의 hash manifest](v2/logs/manifest.json), [공개 다운로드·앱 검사](v2/INSTALL_CHECK.md)도 함께 제공한다. 기록에서는 개인 홈 경로와 터미널 제어문자만 정리하고 수치를 보존하였다.

이 사례에서 확인하려는 실용적인 교훈은 이미 학습된 VLM에 원하는 판단 작업을 명확한 데이터로 가르치는 방법이다. 표현과 조건의 범위를 정하고, 일관된 정답을 만들고, 같은 평가 문제의 전후 변화와 실패까지 확인해야 적응 효과를 설명할 수 있다.''')
    if any(marker in text for marker in ('평가 후 입력','작성 중인 원고','최종 글에는','최종 원고에서는')):
        raise RuntimeError('Unresolved draft placeholder')
    from build_blog_examples import photo_sections
    photo_intro, photo_success, photo_failure = photo_sections(read('docs/v2/photo-examples.json'))
    text = text.replace('| | 왼쪽 | 가운데 | 오른쪽 |', photo_intro + '\n\n| | 왼쪽 | 가운데 | 오른쪽 |', 1)
    text = text.replace('중간에는 고정 표본56개씩을 CPU로 평가한다.', photo_success + '\n\n중간에는 고정 표본56개씩을 CPU로 평가한다.', 1)
    text = text.replace('고정 사진판의 같은 구어체 14개에서는', photo_failure + '\n\n고정 사진판의 같은 구어체 14개에서는', 1)
    # Tell the experiment in the order in which its questions arise.
    text = text.replace('Tiny ImageNet 사진과 자연어 지시로 작은 선택 문제를 구성하고, Qwen3.5-4B 기반 decision model을 학습한 기록이다. v1에서 드러난 표현·조건 조합의 약점을 v2의 데이터 구성으로 보완하였다. 이 글에서는 데이터가 무엇을 의미하는지부터 학습 대상, 실제 개선 폭, 실패 사례와 재현 방법까지 살펴본다.', '사진 여섯 장과 자연어 지시를 주면, 조건에 맞는 사진을 고를 수 있을까? Tiny ImageNet과 Qwen3.5-4B를 이용해 이 문제를 만들고 학습했다. 처음 보는 사진에서 잘 동작하는지, 같은 조건을 다른 말로 묻거나 조건을 조합하면 어떻게 되는지 차례로 살펴보자.')
    text = text.replace('v1은 Qwen의 사전학습 가중치에 새 LoRA와 head를 붙여 학습했고, v2는 그 학습 결과를 이어받았다.', '먼저 Qwen의 사전학습 가중치에 새 LoRA와 head를 붙여 학습한다.')
    start = text.index('v2에서는12개 조건 유형')
    end = text.index('## 5.', start)
    text = text[:start] + '''처음에는 일부 표현과 두 조건 조합을 따로 남겨 두고 나머지로 학습 문제를 만들었다. 처음 보는 문장이나 조합을 만났을 때 단순히 사진만 바뀐 경우와 얼마나 다른지 알아보기 위해서이다. 원본 사진 12,023장으로 격자 4,024개와 학습 예제 31,360개를 만들었고, 서로 다른 문장은 29,350개였다. 한 격자에 여러 지시를 붙이므로 사진 수와 예제 수는 다르다. [처음 데이터 집계](data-summary.json).

''' + text[end:]
    start = text.index('## 5.')
    end = text.index('## 6.', start)
    text = text[:start] + '''## 5. 처음 학습한 모델을 여러 방식으로 시험해 보자

첫 학습은 RTX 4070 Ti SUPER에서 135.7분 동안 진행했다. 학습 예제 31,360개를 사용해 batch 8로 4,400 step을 실행했고, 별도의 검증 문제에서 가장 좋았던 2,400 step 가중치를 골랐다. 이후 비교하기 쉽도록 이 첫 모델을 **v1**이라고 부르자. [학습 설정과 기록](TRAINING.md).

먼저 학습에서 보지 않은 사진으로, 배운 유형의 문제를 풀렸다. 1,120문제에서 **93.48%**를 맞혔다. 사진을 바꿔도 이 범위의 문제는 대체로 풀 수 있다는 결과이다. 그렇다면 같은 뜻을 다른 말로 물어도 괜찮을까? 또 배운 조건들을 새로운 순서로 엮으면 어떨까?

이를 시험하려고 같은 뜻을 다른 문장 구조로 표현한 질문과 여러 조건을 결합한 질문을 만들었다. 처음 모델은 이 문장 틀과 두 조건 조합을 학습하지 않았다. 가령 “고양이 왼쪽 사진”을 배운 뒤 “고를 사진 오른쪽에 고양이가 있어야 한다”고 묻는다. 조건 조합도 마찬가지다. 예를 들어 “고양이와 같은 행에 있는 과일을 골라”와 “사진에서 아래로 한 칸 이동해”라는 요소를 각각 익힌 모델에게, “고양이와 같은 행에서 과일 사진을 찾고, 그 과일의 바로 아래 사진을 골라”라고 묻는다. 같은 행에서 과일을 찾은 다음 그 위치를 출발점으로 삼아 한 칸 이동해야 한다. 아래 세 시험 모두 사진도 학습에 쓰지 않은 사진이며, 항목마다 1,120문제이다.

| 처음 모델에 준 시험 | 맞힌 문제 / 전체 문제 | 정확도 |
|---|---:|---:|
| 새 사진 + 같은 뜻을 바꿔 쓴 표현 | 888 / 1,120 | 79.29% |
| 새 사진 + 새로운 조건 조합 | 759 / 1,120 | 67.77% |
| 새 사진 + 바꿔 쓴 표현과 새로운 조건 조합 | 614 / 1,120 | 54.82% |

사진만 바꾼 93.48%보다 표현을 바꾸면 낮아지고, 조건을 조합하면 더 낮아졌다. 표현과 조합을 모두 바꾸면 절반을 조금 넘는다. 세 시험은 서로 다른 문제 집합이므로 차이를 특정 원인 하나의 순수한 효과로 단정할 수는 없다. 그래도 사용할 만한 표현과 조건이 학습에서 빠졌을 때 어디서 어려워지는지 살펴볼 근거는 된다. [처음 평가 기록](TRAINING.md).

이제 질문을 바꿔 보자. 실제 사용자가 입력할 것으로 예상하는 표현과 조건을 일부러 제외하지 않고 학습에 넣으면, 같은 사진 문제에서 얼마나 나아질까?

## 6. 부족했던 표현과 조건을 데이터에 넣어 보기

이번에는 지원할 12개 조건 유형에 한국어·영어 각각 6개의 문장 틀, 총 144개를 학습 데이터에 포함했다. “3장/세 장”, “3/three”, “왼쪽/좌측” 같은 표기도 섞었다. 방향·임계값·대상을 바꿨을 때 답이 달라지는 대조쌍도 만들었다. 이것을 모든 자연어를 망라한 데이터라고 보기는 어렵다. 지원 범위를 정하고 그 안의 예제를 보강한 것이다.

| 학습 데이터 | 처음 모델 | 보강한 데이터 |
|---|---:|---:|
| 원본 사진 | 12,023장 | 12,830장 |
| 2×3 사진판 | 4,024개 | 4,825개 |
| 지시·정답 예제 | 31,360개 | 37,632개 |
| 서로 다른 문장 | 29,350개 | 36,328개 |

사진 수는 각 버전 안의 고유 개수이다. 같은 학습 사진 풀을 사용하므로 두 열을 더해 전체 고유 사진 수로 삼으면 안 된다. 원본 사진 ID와 RGB hash를 기준으로 학습·검증·보정·평가를 분리했고, 기존 평가 사진은 보강 학습에도 넣지 않았다. [데이터 구성과 검사](V2_DATA_CARD.md).

이 데이터를 사용해 처음 모델의 LoRA와 head를 이어서 학습했다. 이 두 번째 모델을 **v2**라고 부르자. 처음부터 새 모델을 학습한 것이 아니고, 바뀐 것은 학습 데이터와 이어서 학습한 가중치이다. 따라서 비교 결과만으로 데이터 변화와 추가 학습량 각각의 독립적인 효과를 분리할 수는 없다.

''' + text[end:]
    import re
    text = re.sub(r'^## (\d+)\.', lambda m: f'## {int(m.group(1))+1}.' if int(m.group(1)) >= 6 and m.start() > text.index('## 6. 부족했던') else m.group(0), text, flags=re.M)
    text = text.replace('## 10. v1과 v2에서 얻은 교훈', '## 10. 이 실험에서 배운 것')
    text = text.replace('사용자가 궁금했던 비교는 기존의 세 평가 문제를 그대로 다시 풀렸을 때의 변화이다.', '이제 앞에서 어려움을 확인했던 세 평가 문제를 그대로 다시 풀려 보자.')
    # Typography only: keep generated measurements and identifiers intact.
    text = text.replace('**였다.', '** 결과를 보였다.')
    for before, after in {
        '모두16': '모두 16', '합계64': '합계 64', '파라미터4,': '파라미터 4,',
        '약1.40': '약 1.40', '분명한40': '분명한 40', '해상도는64': '해상도는 64',
        'v2에서는12': 'v2에서는 12', '각각6개': '각각 6개', '총144': '총 144',
        '사진12,830': '사진 12,830', '격자4,825': '격자 4,825', '붙여37,632': '붙여 37,632',
        '문장은36,328': '문장은 36,328', '지원할12': '지원할 12', '조건과144': '조건과 144',
        'step2400': 'step 2,400', '학습량은37,632예제': '학습량은 37,632개 예제',
        'batch8': 'batch 8', '4,704step으로1epoch': '4,704 step으로 1 epoch',
        '하나당3,136': '하나당 3,136', '안에서4,704': '안에서 4,704',
        'RTX4070TiSUPER16GB': 'RTX 4070 Ti SUPER 16 GB', 'Ryzen79800X3D': 'Ryzen 7 9800X3D',
        'RAM64GB': 'RAM 64 GB', '학습률은1e-4': '학습률은 1e-4', '학습률은5e-5': '학습률은 5e-5',
        '동일한1,120': '동일한 1,120', '표본56': '표본 56', '전체1,120': '전체 1,120',
        '표현79.29': '표현 79.29', '조합67.77': '조합 67.77', '경우54.82': '경우 54.82',
        '같은4B': '같은 4B', '않은Qwen': '않은 Qwen', '고정revision': '고정 revision',
        '약18.3GB': '약 18.3 GB', '으며,16GB': '으며, 16 GB', '기록된loss': '기록된 loss',
    }.items():
        text = text.replace(before, after)
    from polish_blog import polish
    text = polish(text)
    # Show each saved test case once, immediately followed by the related training example.
    paired = read('docs/v2/training-evaluation-examples.json')['examples']
    photos = {row['suite']: row for row in read('docs/v2/photo-examples.json')['examples'] if row['kind'] == 'improved'}
    pieces = ['### 시험 문제와 보강 데이터를 나란히 보자',
        '정확도가 오른 문제는 실제로 어떻게 생겼을까? 각 시험에서 v1은 틀리고 v2는 맞힌 한국어 문제를 하나씩 가져왔다. 그 바로 아래에는 같은 유형과 문장 틀을 사용한 v2 학습 예제를 놓았다. 각 사진판은 윗줄 왼쪽부터 A·B·C, 아랫줄은 D·E·F다. 두 질문을 비교하면서 어떤 판단을 연습했는지 살펴보자.']
    for pair in paired:
        suite = pair['suite']; test = pair['test']; train = pair['training_example']
        photo = photos[suite]
        assert (photo['scene_id'], photo['instruction'], photo['gold'], photo['v1'], photo['v2']) == (test['scene_id'], test['instruction'], test['action'], test['v1_prediction'], test['v2_prediction'])
        assert not set(test['source_ids']) & set(train['source_ids'])
        pieces += [f"#### {photo['title']}",
            f"**시험 사진판** `{test['scene_id']}`",
            f"![{photo['alt']}](figures/{photo['file']})",
            f"> {test['instruction']}",
            f"정답 **{test['action']}** · v1 **{test['v1_prediction']}** → v2 **{test['v2_prediction']}**",
            photo['explanation'],
            f"**보강 학습 예제** `{train['scene_id']}`",
            f"![{photo['title']}의 보강 학습 사진판](figures/training-pair-{suite}.png)",
            f"> {train['instruction']}",
            f"이 학습 예제의 정답은 **{train['action']}**이다. [v1 예측](../{photo['v1_file']}) · [v2 예측](../{photo['v2_file']})"]
    pieces += ['각 쌍은 사진과 대상은 다르지만, 질문이 요구하는 판단의 구조는 닮아 있다. 보강 학습 예제는 전체 v2 학습 데이터에서 골랐으며, 선택한 체크포인트까지 이 예제들이 각각 제시됐는지는 확인하지 않았다. [각 사진의 출처와 질문·정답 기록](v2/training-evaluation-examples.json).']
    start = text.index('### 실제 사진에서 달라진 선택')
    marker = '## 9. 전체 정확도만으로 알 수 없는 것'
    assert text.count(marker) == 1
    end = text.index(marker, start)
    text = text[:start] + '\n\n'.join(pieces) + '\n\n' + text[end:]
    from article_comparison import add
    text = add(text)
    # Smooth transitions while retaining the experimental setup and evidence.
    edits = {
        '세 시험은 서로 다른 문제 집합이므로 차이를 특정 원인 하나의 순수한 효과로 단정할 수는 없다. 그래도 사용할 만한 표현과 조건이 학습에서 빠졌을 때 어디서 어려워지는지 살펴볼 근거는 된다.': '각 시험은 서로 다른 문제로 구성했다. 결과를 보니 모델이 어떤 표현과 조건에서 어려움을 겪는지 살펴볼 필요가 생겼다.',
        '이것을 모든 자연어를 망라한 데이터라고 보기는 어렵다. 지원 범위를 정하고 그 안의 예제를 보강한 것이다.': '먼저 이 12개 유형을 지원 범위로 정하고, 그 안에서 다양한 말투와 조건을 연습하도록 구성했다.',
        '처음부터 새 모델을 학습한 것이 아니고, 바뀐 것은 학습 데이터와 이어서 학습한 가중치이다. 따라서 비교 결과만으로 데이터 변화와 추가 학습량 각각의 독립적인 효과를 분리할 수는 없다.': 'v1의 가중치에서 출발해 보강한 데이터로 한 번 더 학습하는 방식이다.',
        '다만 초기 실험만으로는 사전학습의 기여를 분리하기 어렵다. 뒤에서 순수 Qwen도 같은 문제의 일부에 실행해 비교한다. LoRA 역시 적은 epoch로 충분하다는 보증이 아니라 파라미터 효율적인 적응 방법이다.': '뒤에서는 추가 학습 전의 기본 Qwen에도 같은 문제 일부를 풀려 본다. LoRA는 기반 모델 전체를 다시 학습하는 대신 적은 수의 파라미터로 이 작업에 적응시키는 방법이다.',
        '이 집계는 위의 세 항목과 다른 문제 집합이다. 서로 다른 표의 전후 점수를 섞어서 개선 폭을 계산하지 않는다.': '앞의 세 항목과는 별도로 만든 문제 집합이다.',
        '이미 이미지와 언어를 함께 다뤄 본 기반 모델이 있어 우리는 사물을 처음부터 가르치는 대신, 사진과 지시를 특정 선택에 연결하는 작업에 집중할 수 있었다. 뒤에서는 기본 Qwen에도 같은 문제 일부를 직접 물어본다. 다만 그 비교에서는 출력 방식과 추가 학습 여부가 함께 달라지므로, 사전학습의 기여만 따로 떼어 볼 수는 없다.': '이미 이미지와 언어를 함께 배운 기반 모델 덕분에, 우리는 사물을 처음부터 가르치기보다 사진과 지시를 원하는 선택에 연결하는 작업에 집중할 수 있었다. 뒤에서는 추가 학습 전의 기본 Qwen도 같은 문제에 도전한다.',
        '다만 14개 구어체 시연의 결과가 보여 주듯, 다음에 다루고 싶은 표현을 찾고 별도의 문제로 다시 시험하는 일도 계속 필요하다.': '14개 구어체 시연에서 발견한 오답은 다음에 어떤 표현을 보강하고 시험할지 알려 준다.',
        '선택 결과만으로는 사진 인식과 문장 해석 중 어느 단계에서 실패했는지 단정할 수 없다.': '이 문제를 통해 사진 속 대상을 알아보는 일과 문장 속 조건을 따라가는 일을 함께 살펴볼 수 있다.',
    }
    edits.update({
        '원본 해상도는 64×64이므로 확대하더라도 실제 세부 정보가 새로 생기는 것은 아니다.': '원본 해상도가 64×64라 사진에 따라 작은 대상은 알아보기 어렵다.',
        '사진 수는 각 버전 안의 고유 개수이다. 같은 학습 사진 풀을 사용하므로 두 열을 더해 전체 고유 사진 수로 삼으면 안 된다. 원본 사진 ID와 RGB hash를 기준으로 학습·검증·보정·평가를 분리했고, 기존 평가 사진은 보강 학습에도 넣지 않았다.': '두 버전은 같은 학습용 사진 풀을 사용하므로 사진 수가 서로 겹친다. 원본 사진 ID와 RGB 해시로 학습·검증·보정·평가용 사진을 분리해, 평가 사진은 보강 학습에서도 따로 두었다.',
        '이번 v2는 사전학습 모델에서 곧바로 시작한 첫 학습이 아니다. v1에서 검증으로 선택한 step 2,400의 LoRA와 head에서 이어서 학습한다. 계획한 추가 학습량은 37,632개 예제, batch 8, 4,704 step으로 1 epoch이다.': 'v2는 v1에서 선택한 step 2,400의 가중치를 이어받는다. 보강한 예제 37,632개를 한 번씩 보며 학습할 계획이다. 한 번에 8개씩 사용하므로 모두 4,704 step이 걸린다.',
        '한 epoch는 각 예제를 한 번씩 사용하는 단위이다. 같은 원리를 한 번만 본다는 뜻은 아니다. 조건 유형 하나당 3,136개 예제가 있으며, 서로 다른 사진과 표현에서 같은 원리를 반복하게 된다. 가중치도 한 epoch 안에서 4,704번 업데이트한다.': '학습 예제를 한 번씩 살펴보는 것을 1 epoch라고 부른다. 사진과 문장이 달라도 같은 판단 규칙을 묻는 예제가 여럿 있다. 예를 들어 “고양이 오른쪽”과 “개구리 오른쪽”은 다른 예제이지만 같은 방향을 찾는 연습이다. 각 조건 유형에는 이런 예제가 3,136개씩 들어 있다.',
        '데이터셋이 바뀌었으므로 두 단계의 epoch를 단순히 더하지 않는다. 전체 run이 1 epoch를 완료했더라도 이후 step의 업데이트가 앞서 선택된 배포 가중치에 포함되는 것은 아니다.': 'v2 학습은 이후 4,704 step까지 마쳤지만, 배포 모델에는 선택한 3,600 step까지의 학습이 담겨 있다.',
        '위 그림의 loss는 기록된 logging 간격의 값과 20개 기록의 이동평균이다. 아래 검증 곡선은 학습에 넣지 않은 검증 사진으로 계산했다. 전체 점수만 보고 새로운 조합의 개선을 추정하지 않고, 다음 절에서 실제 같은 평가 문제끼리 비교한다.': '위 그래프는 학습 중 기록한 loss와 최근 20개 값의 이동평균을 보여 준다. 아래 그래프는 학습에 쓰지 않은 사진으로 계산한 검증 정확도다. 이제 v1과 v2에게 같은 시험을 풀려 결과를 비교해 보자.',
        '남은 질문은 분명하다. 이번에 보강한 문장 틀 밖에서 질문을 바꾸면 어떻게 될까? 그 답을 얻으려면 새로운 표현과 사진을 따로 준비해 다시 시험해야 한다. 기반 모델의 ImageNet 노출이나 근접 중복 여부도 이번 기록만으로는 확인할 수 없다.': '다음에는 이번에 보강한 문장 틀 밖에서 새로운 사진과 질문을 준비해 보자. 그렇게 하면 배운 표현을 새로운 상황으로 얼마나 넓혀 쓸 수 있는지도 살펴볼 수 있다.',
        '한 사진판의 작은 시연이지만, 앞의 표가 좋아진 뒤에도 다음에 보완할 표현을 찾는 데 도움이 된다.': '이 시연은 다음에 어떤 말투를 더 살펴볼지 알려 준다.',
    })
    for before, after in edits.items():
        if text.count(before) != 1:
            raise RuntimeError(f'Expected one editorial passage: {before[:60]}')
        text = text.replace(before, after)
    from article_flow import improve_flow
    text = improve_flow(text)
    (ROOT/'docs/BLOG.md').write_text(text, encoding='utf-8')
    print('FINAL_ARTICLE_WRITTEN', flush=True)

if __name__ == '__main__':
    main()
