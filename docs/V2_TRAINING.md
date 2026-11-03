# v2 학습·평가: 지원하는 표현과 조건을 모두 학습

v1은 일부 표현과 조건 조합을 학습에서 제외하여 일반화를 시험했습니다. v2는 실사용 범위의 정확도를 목표로 **12개 작업군과 언어별 6개 표현(총 144개)을 모두 학습**합니다. 새 사진으로 만든 동일한 2,016개 문제에서 기존 v1과 v2를 직접 비교했습니다.

| 동일한 평가 문제 | v1 | v2 |
|---|---:|---:|
| 전체 2,016개 | 84.52% | 95.78% |
| 한국어 | 84.33% | 95.63% |
| 영어 | 84.72% | 95.93% |

차이는 **+11.26%p**입니다. 이전 README의 79.29%·67.77% 등은 다른 평가 데이터의 점수이므로 이 표와 직접 빼서 비교하면 안 됩니다. v1의 기존 가중치·평가 기록은 그대로 보존합니다.

## 이전 79.29%·67.77%·54.82%와 같은 문제로 직접 비교

사용자가 확인하려던 기존 평가 문제도 사진·지시·정답을 바꾸지 않고 그대로 v2에 실행했습니다. 각 행은 같은 1,120개 문제의 전후 비교입니다.

| 기존 평가 구분 | v1 | v2 | 변화 |
|---|---:|---:|---:|
| 새 사진 + v1에서 보류했던 표현 | 79.29% | 95.71% | +16.43%p |
| 새 사진 + v1에서 보류했던 조건 조합 | 67.77% | 95.71% | +27.95%p |
| 새 사진 + v1에서 보류했던 표현·조합 | 54.82% | 95.62% | +40.80%p |
| 새 사진 + v1에서도 학습한 유형 | 93.48% | 95.36% | +1.88%p |

v1에서 보류했던 표현·조합은 이제 v2 학습에 포함됩니다. 따라서 v2의 이 점수는 해당 유형을 학습한 뒤의 개선 폭이며, 여전히 안 배운 표현·조합을 푼 점수로 해석하면 안 됩니다. **평가 사진 자체는 계속 학습에서 제외**했습니다. [같은 문제 여부 검사와 비교 수치](../results/v2/original-suite-comparison.json).

사용자의 중간 비교 요청에 따라 학습 도중 각 항목의 고정 표본 56개를 CPU로 확인하는 절차도 추가했습니다. 이 표본 결과를 보고 학습 설정이나 체크포인트 선택 기준을 바꾸지 않으며, 최종 표는 전체 1,120개를 양쪽 모두 CUDA로 실행한 결과입니다. 중간 표본 점수와 전체 점수를 혼용하지 않습니다.

## 실제 학습과 가중치 계보

- 기본 모델: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. 원본은 Hugging Face에서 별도로 다운로드합니다.
- 시작점: v1에서 검증으로 선택한 step 2,400의 LoRA와 Clef head. 새로운 adapter/head를 무작위 초기화하지 않고 이어서 학습했습니다. [496개 LoRA 텐서의 실제 갱신 확인](v2-continuation-smoke.json).
- v2 추가 학습 시간: **148.1분**, **NVIDIA GeForce RTX 4070 Ti SUPER (16 GB)** 한 장. Windows / Ryzen 7 9800X3D / RAM 약 64 GB. 클라우드 GPU·유료 데이터 생성 API를 사용하지 않았습니다.
- 완료 step **4,704**; 배포에 선택한 step **3,600**. 선택용 검증 336개에서 97.62%; 최종 test로 선택하지 않았습니다. 동률이면 앞선 체크포인트를 유지합니다.
- batch 8, accumulation 1; v2 학습 데이터 **37,632개**. 전체 run 제시 횟수 37,632회, 선택한 체크포인트까지 28,800회입니다. 선택 이후의 업데이트는 배포 가중치에 포함되지 않습니다.
- v1의 전체 run은 별도로 135.7분이었으며 선택된 부모 가중치에는 처음 2,400 step만 반영됐습니다. 위 v2 시간은 추가 학습 시간입니다.
- NF4 4bit 기본 모델 고정, BF16 계산; LoRA rank/alpha 16/16, LoRA LR 0.0001, head LR 5e-05, AdamW, cosine schedule, seed 3407.
- 학습 대상: LoRA **32,464,896**, Clef head **32,046,596**, 합계 **64,511,492 / 4,603,777,028 = 1.40%**. 압축 저장 크기가 아닌 논리 파라미터 수입니다.
- 최대 PyTorch 할당 6.20 GiB / 예약 7.06 GiB. 디스플레이·다른 앱의 GPU 사용량은 별도입니다.

## 데이터와 평가 범위

37,632개 지시, 4,825개 격자, 12,830개 원본 사진, 서로 다른 문장 36,328개를 사용했습니다. 기존에 보류했던 표현과 두 조건 조합도 전부 학습에 포함합니다. `3장 / 세 장`, `3 / three`, `왼쪽 / 좌측` 등의 표기도 함께 사용합니다. 문장 틀은 코딩 에이전트가 작성한 합성 데이터이며, 별도의 사람 검수 데이터가 아닙니다.

학습·검증·보정·평가의 원본 사진 ID와 RGB hash는 분리했습니다. **기존 test 사진을 학습에 넣지 않았습니다.** 검증 336개로 체크포인트를 고르고, 별도 보정용 1,008개로 temperature만 맞췄습니다. v1과 v2를 같은 v2 test 2,016개에 배포 코드로 실행했습니다. [데이터 상세와 검사](V2_DATA_CARD.md).

평가는 **학습한 지시 유형을 처음 보는 사진에 적용하는 능력**을 측정합니다. 모든 가능한 자연어·모든 추론을 학습했다는 뜻은 아닙니다. 평가 원본 사진 풀은 v1에서 사용한 것과 같고 v1의 결과가 v2 개발에 영향을 주었으므로 완전히 새로운 비공개 시험으로 주장하지 않습니다. Qwen 사전학습의 ImageNet 노출 여부, 근접 중복 여부는 알 수 없습니다.

| 조건 유형 (각 168개) | v1 | v2 |
|---|---:|---:|
| 대상 찾기 | 89.29% | 91.07% |
| 인접 위치 | 87.50% | 97.62% |
| 같은 행의 범주 | 94.05% | 97.02% |
| 같은 열의 범주 | 94.05% | 98.81% |
| 인접 위치 + 범주 | 85.12% | 99.40% |
| 두 단계 이동 | 88.10% | 99.40% |
| 개수 기준 분기 | 79.17% | 88.69% |
| 범주별 개수 비교 | 82.74% | 88.10% |
| 같은 행에서 제외 | 85.71% | 97.02% |
| 이동한 칸의 행에서 범주 찾기 | 90.48% | 97.02% |
| 이동한 칸의 열에서 범주 찾기 | 93.45% | 99.40% |
| 같은 행의 범주를 찾은 뒤 이동 | 44.64% | 95.83% |

v1 오답→v2 정답 250개, v1 정답→v2 오답 23개입니다. [언어·표현·유형별 비교](../results/v2/comparison.json), [전체 수치](../results/v2/finetuned/report.json), [순서상 처음 나온 오답 20개](../results/v2/failure-examples.json)를 공개합니다. 사진이 여러 격자에 반복 사용되므로 전체 문제 수를 독립적인 사진 수로 해석하면 안 됩니다.

## 실제 구어체 시연과 재로딩

기존에 작성한 동일한 사진판·구어체 14개에서 v1 12/14 → v2 11/14였습니다. 이미 확인한 예제이고 하나의 사진판만 사용하므로 독립적인 성능 추정치는 아닙니다. [모든 실제 답과 확률](../results/v2/freeform.json).

- 사진 여섯 장 중 동물이 나온 게 세 장 이상이면 바나나, 그보다 적으면 배낭으로 해줘. → 정답 `B`, 실제 예측 `F`
- 과일보다 탈것 사진이 더 많을 때만 고양이를 선택하고, 동점이거나 적으면 가방을 골라. → 정답 `F`, 실제 예측 `C`
- Only pick the cat if vehicle pictures outnumber fruit pictures. In a tie, or if there are fewer vehicles, pick the bag. → 정답 `F`, 실제 예측 `C`

12개 유형을 모두 포함한 고정 검증 24개를 새 프로세스로 재로딩했습니다. 학습용 코드와 배포 CUDA 답 일치율 100.00%, CPU와 배포 CUDA 일치율 100.00%. 전체 검증 336개에서는 학습용 코드 97.62%, 배포 CUDA 97.62%입니다. 최종 점수는 실제 배포 코드 기준입니다. [재로딩 기록](v2/reload-final.json).

Mac/MPS 실기는 검증하지 못했습니다. CPU/MPS는 float32 기본 모델을 읽어 모델 자체만 약 18.3 GB가 필요합니다. 기존 v1의 빈 모델 캐시 다운로드 검증은 [INSTALL_CHECK.md](INSTALL_CHECK.md)에 있으며, v2는 같은 고정 기본 모델을 사용합니다.

## 다운로드와 재현

배포 파일: [model-v2](https://github.com/gisbi-kim/tiny-visual-reasoning-lab/releases/download/model-v2/tiny-visual-reasoning-4b-v2.zip) (**174,218,456 bytes**, SHA-256 `8bfc48b4850ae200d4320c77bddba03eea4f52e2b66f737910aa8b15ec296240`). 학습한 LoRA·head·processor·보정값만 포함하며 기본 모델과 사진은 별도 다운로드합니다. 첫 실행 후 빨라지는 이유는 로컬 캐시 재사용입니다.

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
