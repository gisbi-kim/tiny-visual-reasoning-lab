# 전체 평가 문제와 사진 배치 읽기

아래 JSON에는 실제 평가한 각 문제의 지시, 정답, 예측과 사진 출처가 기록돼 있다. 첫 네 시험은 각 1,120문제이며, v1과 v2가 같은 문제를 풀었다. v2_test는 별도로 구성한 2,016문제다.

| 시험 | v1 예측 | v2 예측 |
|---|---|---|
| 새 사진, 익숙한 유형 | [1,120문제](../../results/test_id_predictions.json) | [1,120문제](../../results/v2/finetuned/test_id_predictions.json) |
| 새 사진, 바꿔 쓴 표현 | [1,120문제](../../results/test_language_predictions.json) | [1,120문제](../../results/v2/finetuned/test_language_predictions.json) |
| 새 사진, 새로운 조건 조합 | [1,120문제](../../results/test_composition_predictions.json) | [1,120문제](../../results/v2/finetuned/test_composition_predictions.json) |
| 새 사진, 표현과 조합 | [1,120문제](../../results/test_combined_predictions.json) | [1,120문제](../../results/v2/finetuned/test_combined_predictions.json) |
| v2 별도 시험 | [2,016문제](../../results/v2/baseline_v1/v2_test_predictions.json) | [2,016문제](../../results/v2/finetuned/v2_test_predictions.json) |

각 배열의 같은 순서에 같은 문제가 있다. `source_ids`의 여섯 항목은 A, B, C, D, E, F 순서이며 A–C는 윗줄, D–F는 아랫줄이다. `object_keys`는 같은 순서의 사진별 중심 대상이다. `instruction`은 지시, `action`은 정답, `prediction.action`은 모델이 고른 답이다. `scene_id`, `program`, `trace`, `variant`, `boxes`, `render_seed`와 선택지 확률도 남아 있다.

예를 들어 첫 번째 바꿔 쓴 표현 문제의 `source_ids`는 A `val/images/val_3543.JPEG`, B `val/images/val_6537.JPEG`, C `val/images/val_8406.JPEG`, D `val/images/val_9544.JPEG`, E `val/images/val_203.JPEG`, F `val/images/val_7852.JPEG`다. 대응하는 대상은 금붕어, 기니피그, 기차, 개, 아이스크림, 곰이며 정답은 NONE이다. 원본 이미지 파일은 저장소에 포함하지 않는다. 그림을 다시 만들려면 Tiny ImageNet 원본과 [렌더링 코드](../../render.py)가 필요하다. 사진별 ID와 배치가 남아 있지만 원본 데이터의 별도 확보가 필요하다.

v1과 v2 기록에서 문제 수, 사진 ID와 대상 순서, 지시, 정답, 사진판 ID가 동일한지 확인했다. 이 기록은 평가 예측을 저장한 것이며, 모든 학습 배치의 예측이나 미기록 과거 실행까지 포함한다는 뜻은 아니다.
