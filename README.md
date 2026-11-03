# Tiny Visual Reasoning Lab

Tiny ImageNet 사진 여섯 장과 한국어·영어 지시를 읽고 **A–F 또는 NONE**을 선택하는 로컬 실험입니다. 기존 MNIST 프로젝트와 별개의 코드·데이터·가중치를 사용합니다.

예: “고양이에서 왼쪽으로 한 칸, 그다음 아래로 한 칸 이동한 사진을 골라줘.” / “If there are at least three animal photographs, choose the banana; otherwise choose the backpack.”

**Qwen3.5-4B + LoRA + Clef decision head**를 실제로 학습한 실험입니다. 정답 생성기는 데이터 제작·검증에만 쓰이며 앱의 예측 경로에는 들어가지 않습니다. 출력은 선택지와 확률입니다. 자유로운 설명문이나 내부 추론 과정을 생성하는 모델은 아닙니다.

<!-- RESULTS_START -->
실제 학습: **NVIDIA GeForce RTX 4070 Ti SUPER에서 135.7분**, 검증으로 선택한 step **2,400**. 전체 논리 파라미터의 **1.40%**를 학습했습니다.

| 평가 | 문제 수 | 정확도 | 장면 단위 bootstrap 95% 구간 |
|---|---:|---:|---:|
| 새 사진 | 1,120 | 93.48% | 91.44%–95.35% |
| 새 사진 + 보류한 문장 표현 | 1,120 | 79.29% | 75.87%–82.35% |
| 새 사진 + 보류한 조건 조합 | 1,120 | 67.77% | 63.77%–71.70% |
| 새 사진 + 보류한 표현·조건 조합 | 1,120 | 54.82% | 50.80%–59.04% |

[학습·가중치 출처·검증 범위](docs/TRAINING.md) · [전체 평가 수치](results/report.json) · [실패 예제](results/failure-examples.json)
<!-- RESULTS_END -->

## 실행

Python 3.11 또는 3.12와 Git이 필요합니다. 첫 실행은 기본 모델 약 **9.34 GB**, 별도 학습 가중치, 원본 Tiny ImageNet 약 **248 MB**를 다운로드합니다. 이후에는 로컬 캐시를 재사용합니다. GPU 메모리와 다운로드 파일 크기는 서로 다릅니다.

공개 v1을 새로 내려받아 빈 모델 캐시에서 실행하는 경로를 검증했습니다. 다운로드·준비·첫 예제 실행은 이 장비에서 약 5분 54초였으며, 설치된 Python 패키지는 재사용했습니다. [실제 설치·HTTP 검사 기록](docs/INSTALL_CHECK.md).

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

- 원본 200개 클래스 중 사전 검토한 40개를 사용합니다. 저해상도에서 구분이 어렵거나 음식이 도구를 가리는 클래스 등은 제외했습니다. 모든 사진을 사람이 검수한 데이터라는 뜻은 아닙니다.
- 원본 bounding box로 중심 대상을 잘라내고, 너무 작은 대상과 RGB가 완전히 같은 중복 이미지를 제외합니다. 64×64 원본을 확대하므로 실제 세부 정보가 늘지는 않습니다.
- 한 장당 중심 대상 하나로 정의하고, 2×3 격자의 위치·행·열·분류·제외·개수 비교·조건 분기·두 단계 이동을 조합합니다. “동물 사진 수”는 사진 안 모든 동물 개체 수와 다릅니다.
- 31,360개 학습 지시, 4,024개 격자, 12,023개 원본 사진, 29,350개의 서로 다른 문장을 생성했습니다. 한국어·영어 각각 15,680개, A–F/NONE 각각 4,480개입니다.
- 문장은 코딩 에이전트가 작성·점검한 의미별 문장 틀, 어휘 대체, 요청 표현을 조합한 합성 데이터입니다. 별도의 사람 검수 데이터가 아닙니다. 10개 학습 작업군마다 언어별 4개 표현을 사용하고, 추가 2개 표현은 평가에만 사용합니다. 조건문 참/거짓도 균형을 맞췄습니다. 자연어 문장 개수는 독립적인 의미 개수와 같지 않습니다.
- 같은 격자에서 방향·임계값·대상을 바꿔 답이 달라지는 대조쌍 3,148개(언어별 집계)를 넣었습니다. 최소 수정 쌍은 같은 문장 틀과 어휘 난수를 공유합니다.
- 정답 후보가 여러 개인 모호한 지시는 생성 단계에서 제외했습니다. 앱에서도 한 사진으로 결정되는 조건을 입력하는 것이 학습 범위에 맞습니다.
- 원본 사진 ID와 전체 RGB hash를 기준으로 학습/검증/보정/최종 평가를 분리했습니다. 공식 labeled validation 사진을 최종 평가에 사용하며, 라벨 없는 공식 test는 쓰지 않습니다. 최종 평가 네 종류는 같은 사진 풀을 공유합니다.

자세한 수치: [데이터 구성](docs/data-summary.json), [원본 필터](docs/source-quality.json), [사전 클래스 검토](docs/curation.json), [사물·범주 정의](docs/catalog.json).

## 무엇을 평가하나

| 구분 | 사진 | 문장 표현 | 조건 조합 | 개수 |
|---|---|---|---|---:|
| test_id | 새 사진 | 학습 표현군 | 학습 조합군 | 1,120 |
| test_language | 새 사진 | 보류한 표현군 | 학습 조합군 | 1,120 |
| test_composition | 새 사진 | 학습 표현군 | 보류한 2개 조합군 | 1,120 |
| test_combined | 새 사진 | 보류한 표현군 | 보류한 2개 조합군 | 1,120 |

보류한 조합은 **인접 위치의 열에서 범주 찾기**, **같은 행의 범주를 찾은 뒤 이동하기**입니다. 더 긴 문장을 만들었다는 이유만으로 일반 추론 능력이 개선됐다고 주장하지 않습니다. 정확도 외에 언어·작업군·선택지별 결과, 지시 변경 대조쌍, NONE precision/recall, 확률 보정 오차를 함께 기록합니다.

평가 사진은 학습용 LoRA에 들어가지 않지만, Qwen 사전학습에 ImageNet 이미지가 포함됐는지는 알 수 없습니다. 근접 중복과 의미적으로 비슷한 문장까지 완전히 독립임을 증명한 데이터는 아닙니다.

## 재현

앱 실행과 전체 학습 데이터 생성은 별개입니다. `prepare_demo.py`는 로컬 시연 사진 400장만 준비합니다. 아래 학습 명령은 NVIDIA CUDA 환경을 위한 것이며 Mac 실행에는 필요하지 않습니다.

```bash
python fetch_assets.py                 # 원본 사진 + 고정 revision 기본 모델
python prepare_data.py                # 전체 데이터·정답·누수 검사, 기존 데이터 보존
python -m pip install -r requirements-train.txt
python train.py --run reproduction --steps 4400 --seconds 9600 --batch 8 --accum 1 --validate-every 200 --val-count 140 --four-bit
python evaluate.py --checkpoint runs/reproduction/best --device cuda --calibrate
python -m pytest -q
```

실행 환경에 따라 처리량과 한도 내 학습 step 수가 다를 수 있습니다. 체크포인트는 고정된 검증 140개로 선택하고 보정용 560개로 temperature만 맞춥니다. 최종 test로 가중치를 고르지 않습니다. 실제 학습 명령, 완료 step, 경과 시간, 학습 파라미터 비율은 [학습 기록](docs/TRAINING.md)을 기준으로 봐 주세요.

## 출처와 라이선스

- 기본 모델: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. 기본 가중치는 GitHub에 재배포하지 않고 Hugging Face에서 받습니다. 원본 모델의 라이선스·이용 조건이 적용됩니다.
- 사진: [Stanford Tiny ImageNet 원본](https://cs231n.stanford.edu/tiny-imagenet-200.zip). 원본 사진과 만들어진 격자 이미지를 GitHub/가중치 파일에 포함하지 않습니다. 이미지의 권리는 원 소유자에게 있으며 코드의 Apache-2.0 라이선스가 사진에 적용되는 것은 아닙니다.
- Clef: [Cloudflare/clef](https://huggingface.co/Cloudflare/clef), Apache-2.0. `clef_reference.py`는 원본의 `joint_schema_model.py`를 그대로 보관한 파일입니다. 출처와 hash는 [clef_manifest.json](clef_manifest.json), 원 라이선스는 [CLEF_LICENSE.txt](CLEF_LICENSE.txt)에 있습니다.
- 이 프로젝트 코드: Apache-2.0. 학습 가중치의 다운로드 위치·크기·검증값은 `docs/release.json`에 기록합니다.
