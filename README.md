# Tiny Visual Reasoning Lab

Tiny ImageNet 사진 여섯 장과 한국어·영어 지시를 읽고 **A–F 또는 NONE**을 선택하는 로컬 실험입니다. 기존 MNIST 프로젝트와 별개의 코드·데이터·가중치를 사용합니다.

[블로그: v1에서 v2까지의 데이터 설계와 교훈](https://gisbi-kim.github.io/tiny-visual-reasoning-lab/) · [글 원문](docs/BLOG.md) · [전체 실험 자료·로그 안내](docs/v2/INDEX.md) · [공개 v2 다운로드·앱 검사](docs/v2/INSTALL_CHECK.md)

예: “고양이에서 왼쪽으로 한 칸, 그다음 아래로 한 칸 이동한 사진을 골라줘.” / “If there are at least three animal photographs, choose the banana; otherwise choose the backpack.”

**Qwen3.5-4B + LoRA + Clef decision head**를 실제로 학습한 실험입니다. 정답 생성기는 데이터 제작·검증에만 쓰이며 앱의 예측 경로에는 들어가지 않습니다. 출력은 선택지와 확률입니다. 자유로운 설명문이나 내부 추론 과정을 생성하는 모델은 아닙니다.

<!-- RESULTS_START -->
현재 배포는 **v2**입니다. 지원하는 **12개 조건 유형과 한·영 표현 144개를 전부 학습**했고, 사진만 학습·평가 사이에서 분리했습니다. 숫자 표기·수사와 방향 표현도 함께 다양화했습니다.

실제 추가 학습: **RTX 4070 Ti SUPER에서 148.1분**. v1의 LoRA/head에서 이어서 학습했으며, 학습 대상은 전체 논리 파라미터의 **1.40%** 수준입니다. v1 전체 run 135.7분과 별개인 추가 학습 시간이고, 배포 체크포인트는 v2 step **3,600**입니다.

| 같은 새 사진 평가 2,016개 | v1 | v2 |
|---|---:|---:|
| 전체 | 84.52% | **95.78%** |
| 한국어 | 84.33% | 95.63% |
| 영어 | 84.72% | 95.93% |

학습한 지시 유형을 새 사진에 적용하는 평가입니다. 처음 보는 표현·작업 조합에 대한 일반화 점수라는 뜻은 아닙니다. v1의 이전 79.29%·67.77% 등과는 평가 문제가 다릅니다.

기존 평가 문제도 그대로 다시 실행했습니다. 아래는 사진·지시가 동일한 문제끼리의 전후 비교입니다(각 1,120개).

| 이전 평가 구분 | v1 | v2 | 변화 |
|---|---:|---:|---:|
| 새 사진 + v1에서 보류했던 표현 | 79.29% | 95.71% | +16.43%p |
| 새 사진 + v1에서 보류했던 조건 조합 | 67.77% | 95.71% | +27.95%p |
| 새 사진 + v1에서 보류했던 표현·조합 | 54.82% | 95.62% | +40.80%p |
| 새 사진 + v1에서도 학습한 유형 | 93.48% | 95.36% | +1.88%p |

기존에 보류했던 표현·조합은 v2에서 학습했습니다. 따라서 v2에서는 새 표현·새 조합 일반화가 아니라 **해당 유형을 학습한 효과**를 보는 비교입니다. 평가 사진은 학습에 넣지 않았습니다. [원래 평가 문제의 비교 기록](results/v2/original-suite-comparison.json).

별도의 고정 사진판·구어체 시연 14개에서는 **v1 12/14 → v2 11/14**였습니다. 작은 시연이지만, 작성한 표현 틀의 성능 개선이 자유로운 말투 전체의 개선을 보장하지는 않음을 보여 줍니다. 개수·비교 조건의 실제 오답도 [시연 결과](results/v2/freeform.json)에 공개합니다.

[v2 학습·가중치 출처·한계](docs/V2_TRAINING.md) · [동일 문제 비교](results/v2/comparison.json) · [실패 예제](results/v2/failure-examples.json) · [과거 v1 실험](docs/TRAINING.md)
<!-- RESULTS_END -->

## 실행

이미 v1을 쓰고 있다면 기존 파일을 보관한 채 다음처럼 업데이트합니다. 새 설치의 `download_model.py` 기본값은 현재 v2입니다.

```bash
git pull
python download_model.py --version v2 --output weights-v2
python app.py --checkpoint weights-v2 --device cpu
```

CUDA 사용자는 마지막 명령의 `cpu`를 `cuda`로 바꿉니다. v1도 `--version v1 --output weights-v1`로 계속 받을 수 있습니다.

Python 3.11 또는 3.12와 Git이 필요합니다. 첫 실행은 기본 모델 약 **9.34 GB**, 별도 학습 가중치, 원본 Tiny ImageNet 약 **248 MB**를 다운로드합니다. 이후에는 로컬 캐시를 재사용합니다. GPU 메모리와 다운로드 파일 크기는 서로 다릅니다.

공개 v1을 새로 내려받아 빈 모델 캐시에서 실행하는 경로를 검증했습니다. 다운로드·준비·첫 예제 실행은 이 장비에서 약 5분 54초였으며, 설치된 Python 패키지는 재사용했습니다. [실제 설치·HTTP 검사 기록](docs/INSTALL_CHECK.md).

v2도 공개 Release에서 새로 받아 archive와 내부 파일 15개의 해시를 검사했습니다. 기존 기본 모델 캐시를 사용하여 CUDA 앱을 실행했으며 고정 HTTP 시연은 **6/7**이었습니다(개수 분기 오답). [v2 공개 파일·앱 검사](docs/v2/INSTALL_CHECK.md). 이 시간과 시연 결과를 빈 캐시 첫 설치나 전체 평가 정확도와 혼용하지 않습니다.

```bash
git clone https://github.com/gisbi-kim/tiny-visual-reasoning-lab.git
cd tiny-visual-reasoning-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python download_model.py
python fetch_assets.py --dataset-only
python prepare_demo.py
python app.py --device cpu
```

브라우저에서 <http://127.0.0.1:7861>을 엽니다. Windows에서는 `python3` 대신 `python`, 가상환경 활성화는 `.venv\Scripts\Activate.ps1`을 사용합니다.

Apple Silicon Mac에서는 같은 설치 후 다음 명령으로 MPS를 선택할 수 있습니다. **실제 Mac에서의 동작·속도는 검증하지 못했습니다.** CPU/MPS 경로는 float32로 기본 모델을 읽으므로 모델 자체만 약 18.3 GB이며, 활성값과 운영체제 메모리가 추가로 필요합니다. 16 GB Mac에 넉넉하게 들어간다고 보장하지 않습니다. MPS 연산 지원에 따라 아래 fallback 설정이 필요하고 CPU보다 빨라진다는 보장도 없습니다.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python app.py --device mps
```

NVIDIA CUDA는 해당 시스템에 맞는 PyTorch를 설치한 뒤 `pip install bitsandbytes==0.50.2`, `python app.py --device cuda`를 사용합니다. CUDA 경로에서는 기본 모델을 NF4 4bit로 로드합니다. Unsloth는 **학습용**이며 앱 실행에 필요하지 않습니다. 실측 장비·메모리·평가 결과는 [학습 기록](docs/TRAINING.md)에 기록합니다.

## 데이터에 들인 작업

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

## 출처와 라이선스

- 기본 모델: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. 기본 가중치는 GitHub에 재배포하지 않고 Hugging Face에서 받습니다. 원본 모델의 라이선스·이용 조건이 적용됩니다.
- 사진: [Stanford Tiny ImageNet 원본](https://cs231n.stanford.edu/tiny-imagenet-200.zip). 원본 사진과 만들어진 격자 이미지를 GitHub/가중치 파일에 포함하지 않습니다. 이미지의 권리는 원 소유자에게 있으며 코드의 Apache-2.0 라이선스가 사진에 적용되는 것은 아닙니다.
- Clef: [Cloudflare/clef](https://huggingface.co/Cloudflare/clef), Apache-2.0. `clef_reference.py`는 원본의 `joint_schema_model.py`를 그대로 보관한 파일입니다. 출처와 hash는 [clef_manifest.json](clef_manifest.json), 원 라이선스는 [CLEF_LICENSE.txt](CLEF_LICENSE.txt)에 있습니다.
- 이 프로젝트 코드: Apache-2.0. 학습 가중치의 다운로드 위치·크기·검증값은 `docs/release.json`에 기록합니다.
