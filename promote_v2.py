"""Promote v2 documentation only after its public release asset exists."""
import json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def main():
    release = read('docs/release-v2.json')
    request = urllib.request.Request(release['url'], method='HEAD', headers={'User-Agent': 'tiny-visual-reasoning-lab'})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError('V2 asset is not public yet')
    comparison = read('results/v2/comparison.json')
    free_old = read('results/freeform.json')
    free_new = read('results/v2/freeform.json')
    old_hits = sum(r['gold'] == r['prediction']['action'] for r in free_old['rows'])
    new_hits = sum(r['gold'] == r['prediction']['action'] for r in free_new['rows'])
    train = read('docs/v2/training-metrics.json')
    validation = read('docs/v2/validation-history.json')
    best = max(validation, key=lambda x: x['accuracy'])
    pct = lambda v: f'{100*v:.2f}%'
    original_table = '\n'.join(f"| {r['label']} | {pct(r['v1_accuracy'])} | {pct(r['v2_accuracy'])} | {r['delta_percentage_points']:+.2f}%p |" for r in comparison['original_v1_suites'].values())
    path = ROOT/'README.md'
    body = path.read_text(encoding='utf-8')
    before, tail = body.split('<!-- RESULTS_START -->', 1)
    _, after = tail.split('<!-- RESULTS_END -->', 1)
    headline = f'''현재 배포는 **v2**입니다. 지원하는 **12개 조건 유형과 한·영 표현 144개를 전부 학습**했고, 사진만 학습·평가 사이에서 분리했습니다. 숫자 표기·수사와 방향 표현도 함께 다양화했습니다.

실제 추가 학습: **RTX 4070 Ti SUPER에서 {train['train_runtime']/60:.1f}분**. v1의 LoRA/head에서 이어서 학습했으며, 전체 논리 파라미터의 **1.40%**가 학습 대상입니다. v1 전체 run 135.7분과 별개인 추가 학습 시간이고, 배포 체크포인트는 v2 step **{best['step']:,}**입니다.

| 같은 새 사진 평가 2,016개 | v1 | v2 |
|---|---:|---:|
| 전체 | {pct(comparison['v1_accuracy'])} | **{pct(comparison['v2_accuracy'])}** |
| 한국어 | {pct(comparison['language_accuracy']['ko']['v1'])} | {pct(comparison['language_accuracy']['ko']['v2'])} |
| 영어 | {pct(comparison['language_accuracy']['en']['v1'])} | {pct(comparison['language_accuracy']['en']['v2'])} |

학습한 지시 유형을 새 사진에 적용하는 평가입니다. 처음 보는 표현·작업 조합에 대한 일반화 점수라는 뜻은 아닙니다. v1의 이전 79.29%·67.77% 등과는 평가 문제가 다릅니다.

기존 평가 문제도 그대로 다시 실행했습니다. 아래는 사진·지시가 동일한 문제끼리의 전후 비교입니다(각 1,120개).

| 이전 평가 구분 | v1 | v2 | 변화 |
|---|---:|---:|---:|
{original_table}

기존에 보류했던 표현·조합은 v2에서 학습했습니다. 따라서 v2에서는 새 표현·새 조합 일반화가 아니라 **해당 유형을 학습한 효과**를 보는 비교입니다. 평가 사진은 학습에 넣지 않았습니다. [원래 평가 문제의 비교 기록](results/v2/original-suite-comparison.json).

별도의 고정 사진판·구어체 시연 14개에서는 **v1 {old_hits}/14 → v2 {new_hits}/14**였습니다. 작은 시연이지만, 작성한 표현 틀의 성능 개선이 자유로운 말투 전체의 개선을 보장하지는 않음을 보여 줍니다. 개수·비교 조건의 실제 오답도 [시연 결과](results/v2/freeform.json)에 공개합니다.

[v2 학습·가중치 출처·한계](docs/V2_TRAINING.md) · [동일 문제 비교](results/v2/comparison.json) · [실패 예제](results/v2/failure-examples.json) · [과거 v1 실험](docs/TRAINING.md)
'''
    body = before + '<!-- RESULTS_START -->\n' + headline + '<!-- RESULTS_END -->' + after
    insert = '''이미 v1을 쓰고 있다면 기존 파일을 보관한 채 다음처럼 업데이트합니다. 새 설치의 `download_model.py` 기본값은 현재 v2입니다.

```bash
git pull
python download_model.py --version v2 --output weights-v2
python app.py --checkpoint weights-v2 --device cpu
```

CUDA 사용자는 마지막 명령의 `cpu`를 `cuda`로 바꿉니다. v1도 `--version v1 --output weights-v1`로 계속 받을 수 있습니다.

'''
    body = body.replace('## 실행\n\n', '## 실행\n\n' + insert, 1)
    start = body.index('## 데이터에 들인 작업')
    end = body.index('## 출처와 라이선스')
    replacement = '''## 데이터에 들인 작업

- 원본 200개 클래스 중 저해상도 사진에서 구분 가능한 40개를 선별했습니다. 코딩 에이전트가 사진 표본과 의미 정의를 검토했으며, 모든 사진을 사람이 검수한 데이터는 아닙니다.
- 원본 bounding box로 중심 대상을 자르고, 너무 작은 대상과 RGB가 완전히 같은 중복을 제외했습니다. 각 사진의 중심 대상 하나를 기준으로 2×3 격자를 구성합니다.
- v2는 **37,632개 학습 지시**, 4,825개 격자, 12,830개 원본 사진, 서로 다른 문장 36,328개입니다. 한국어·영어와 A–F/NONE 정답 수, 조건문의 참·거짓을 균형 있게 구성했습니다.
- 위치·행·열·범주·제외·개수 비교·조건 분기·두 단계 이동을 포함한 **12개 작업군 × 2개 언어 × 6개 표현**을 모두 학습합니다. `3장 / 세 장`, `3 / three`, `왼쪽 / 좌측` 등 의미를 유지한 표기도 섞었습니다.
- 같은 사진에서 방향·임계값·대상을 바꿔 정답이 달라지는 대조쌍 3,726개(언어별 집계)를 넣었습니다. 정답이 여러 개인 모호한 지시는 생성 단계에서 제외했습니다.
- 문장은 코딩 에이전트가 작성한 의미별 틀과 어휘 대체로 생성합니다. 별도의 사람 검수나 외부 모델 API 생성 데이터가 아니며, 문장 수가 독립적인 의미의 수와 같은 것은 아닙니다.
- 원본 사진 ID·전체 RGB hash로 학습/검증/보정/평가를 분리하고 모든 정답을 다시 계산해 검사했습니다. v1의 평가 사진을 v2 학습에 넣지 않았습니다.

[v2 데이터 구성·한계](docs/V2_DATA_CARD.md) · [상세 집계](docs/v2-data-summary.json) · [원본 필터](docs/source-quality.json) · [사전 클래스 검토](docs/curation.json) · [사물·범주 정의](docs/catalog.json).

## 무엇을 평가하나

v2의 주 평가는 **학습한 표현·조건을 새로운 사진에 적용하는 선택 정확도**입니다. 336개 검증 문제로 체크포인트를 고르고, 별도 1,008개로 확률을 보정한 뒤, 동일한 최종 2,016개 문제에 v1과 v2를 실행합니다. 모든 사진 풀은 서로 분리합니다.

평가에는 같은 원본 사진이 여러 격자에 재사용됩니다. v1에서 사용한 평가 사진 풀을 재사용하고 v1의 결과가 v2 개발에 영향을 주었으므로 완전히 새로운 비공개 평가라고 주장하지 않습니다. 기본 Qwen 사전학습에 ImageNet 사진이 포함됐는지, 근접 중복까지 없는지는 알 수 없습니다.

모든 가능한 자연어를 학습한 것은 아닙니다. 제한된 사진 범주·12개 조건 유형·작성한 문장 틀 안의 실험입니다. 언어·작업군·표현별 결과와 실제 실패를 [v2 학습 기록](docs/V2_TRAINING.md)에 함께 공개합니다. 일부 표현과 조합을 보류했던 [v1 실험 기록](docs/TRAINING.md)과 [model-v1](https://github.com/gisbi-kim/tiny-visual-reasoning-lab/releases/tag/model-v1)은 보존합니다.

## 재현

앱 실행에는 전체 학습 데이터 생성이 필요하지 않습니다. `prepare_demo.py`가 시연 사진 400장만 준비합니다. 학습 재현에는 NVIDIA CUDA와 `requirements-train.txt`가 필요합니다.

```bash
python -m pip install -r requirements-train.txt
python fetch_assets.py
python prepare_data.py
python prepare_v2.py
python download_model.py --version v1 --output weights-v1
python train.py --run reproduction-v2 --init-checkpoint weights-v1 --train-split v2_train --validation-split v2_validation_select --steps 4704 --seconds 9900 --batch 8 --accum 1 --val-count 336 --validate-every 400 --four-bit --lr 0.0001 --head-lr 0.00005
python evaluate.py --checkpoint runs/reproduction-v2/best --device cuda --calibrate --calibration-split v2_calibration --warmup-split v2_validation --suites v2_test --output results/reproduction-v2
python -m pytest -q
```

실행 환경에 따라 시간 한도 내 완료 step 수가 달라집니다. v2는 배포된 v1 가중치를 시작점으로 합니다. 원본 v1부터의 재현 설정은 [v1 기록](docs/TRAINING.md), 실제 v2 설정·시간·선택 step·데이터 hash는 [v2 기록](docs/V2_TRAINING.md)에 있습니다.

'''
    body = body[:start] + replacement + body[end:]
    path.write_text(body, encoding='utf-8')
    (ROOT/'docs/release.json').write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PROMOTED_PUBLIC_V2')

if __name__ == '__main__':
    main()
