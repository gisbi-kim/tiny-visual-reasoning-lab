# 실험 자료 안내

본문 설명은 [블로그 글](../BLOG.md), 핵심 결과는 [v2 학습 기록](../V2_TRAINING.md)에 있습니다. 아래 파일은 설명의 근거가 되는 실제 기록입니다.

| 확인할 내용 | 파일 |
|---|---|
| 기존 세 평가 항목: 같은 1,120문제의 v1→v2 | [original-suite-comparison.json](../../results/v2/original-suite-comparison.json) |
| 새로 구성한 동일 2,016문제의 v1→v2 | [comparison.json](../../results/v2/comparison.json) |
| 조건·언어·선택지별 정확도, 혼동행렬, NONE, 확률·지연 통계 | [finetuned/report.json](../../results/v2/finetuned/report.json) |
| 각 문제의 사진 ID·지시·정답·실제 예측 | [전체 평가 결과 폴더](../../results/v2/finetuned/) |
| 순서상 처음 나온 실패 사례 | [failure-examples.json](../../results/v2/failure-examples.json) |
| 학습 데이터 구성·분할·의미 정의 | [V2_DATA_CARD.md](../V2_DATA_CARD.md), [v2-data-summary.json](../v2-data-summary.json), [catalog.json](../catalog.json) |
| 원본 사진 선별·작은 대상·정확 중복 필터 | [curation.json](../curation.json), [source-quality.json](../source-quality.json) |
| 학습 설정·학습 가능한 모든 파라미터 이름 | [training-config.json](training-config.json) |
| 실제 시간·최대 메모리 | [training-metrics.json](training-metrics.json) |
| 기록된 loss·학습률·gradient norm·epoch | [training-history.json](training-history.json) |
| 단계별 검증 정확도 | [validation-history.json](validation-history.json) |
| 단계별 검증의 실제 정답·예측 | [logs/runs/v2/validation.json](logs/runs/v2/validation.json) |
| 부모 가중치·데이터·학습 코드 hash | [provenance.json](provenance.json) |
| 확률 보정용 데이터와 temperature | [logs/calibration/v2.json](logs/calibration/v2.json) |
| 원래 실행한 학습·평가 콘솔 로그 | [logs/console/](logs/console/) |
| v1·smoke 등 이전 실행의 설정과 기록 | [logs/runs/](logs/runs/) |
| CPU 중간 평가와 자유 입력 시연 | [results/v2/](../../results/v2/) |
| CPU/CUDA/학습용 구현의 재로딩 검사 | [reload-final.json](reload-final.json) |
| 기본 모델 출처·파일 hash | [base-model.json](../base-model.json), [base-model-files.json](../base-model-files.json) |
| 학습한 v2 가중치 위치·크기·파일 hash | [release-v2.json](../release-v2.json) |
| Python·라이브러리·운영체제 | [environment.json](../environment.json), [실행 당시 패키지 목록](logs/runs/v2/requirements.lock.txt) |
| 공개 가중치 다운로드·실제 앱 검사 | [INSTALL_CHECK.md](INSTALL_CHECK.md) |

## 로그를 읽을 때

`training-history.json`의 loss는 설정된 logging 간격의 기록입니다. 모든 microbatch의 사진과 예측을 저장한 자료가 아닙니다. 기록되지 않은 과거 예측을 사후에 만들어 로그로 제공하지 않았습니다. 전체 실행 시간과 배포 체크포인트의 선택 시점, 데이터셋 크기와 실제 샘플 제시 횟수도 구분해야 합니다.

중간 평가의 고정 표본 56개는 최종 전체 1,120개와 다른 집계입니다. 중간 v2는 CPU, 저장된 v1은 CUDA였으며, 최종 비교는 양쪽 모두 CUDA입니다. 작은 시연 사례의 점수를 전체 성능으로 해석하지 않습니다.

공개 로그에서는 개인 홈 경로와 터미널 제어문자만 정리했습니다. 각 파일의 원본·공개본 SHA-256과 변경 항목은 [manifest.json](logs/manifest.json)에 남겼습니다. 원본 사진과 optimizer/model checkpoint 바이너리는 이 로그 폴더에 포함하지 않습니다. 배포 가중치는 Release에서 별도로 받습니다.
