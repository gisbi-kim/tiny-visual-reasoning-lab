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
    for key, label in [('test_language', '새 사진 + v1에서 보류했던 표현'),
                       ('test_composition', '새 사진 + v1에서 보류했던 조건 조합'),
                       ('test_combined', '새 사진 + v1에서 보류했던 표현·조합')]:
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
    families = '\n'.join(f"| {names[k]} | {pct(v['v1'])} | {pct(v['v2'])} |" for k,v in common['condition_accuracy'].items())
    examples = []
    for suite in ('test_language','test_composition','test_combined'):
        rows = read(f'results/v2/finetuned/{suite}_predictions.json')
        wrong = next((r for r in rows if r['action'] != r['prediction']['action']), None)
        if wrong:
            examples.append(f"- `{suite}`: “{wrong['instruction']}” → 정답 **{wrong['action']}**, 예측 **{wrong['prediction']['action']}**. 사진판 ID `{wrong['scene_id']}`.")
    paragraph('대상 찾기, 두 단계 이동,',
        f'''대상 찾기, 두 단계 이동, 개수 분기, 범주별 개수 비교는 서로 다른 어려움이 있다. 아래 표는 별도 v2_test 2,016문제에서 계산한 조건별 결과이다. 조건마다 168문제이며, 위의 기존 세 항목 표와 구분한다.

| 조건 유형 | v1 | v2 |
|---|---:|---:|
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
    (ROOT/'docs/BLOG.md').write_text(text, encoding='utf-8')
    print('FINAL_ARTICLE_WRITTEN', flush=True)

if __name__ == '__main__':
    main()
