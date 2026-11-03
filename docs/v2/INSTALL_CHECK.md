# 공개 v2 가중치 다운로드와 앱 검사

2026-11-03, 실제 학습에 사용한 Windows / RTX 4070 Ti SUPER 장비에서 검사했다.

## 공개 파일을 다시 내려받은 결과

`verify_v2_public.py download`로 GitHub Release의 공개 URL에 인증 없이 접속했다. 로컬 export 파일을 복사하지 않고, 존재하지 않던 `weights-v2/`에 새로 설치했다. 다운로드·압축 해제·검사에 **17.71초**가 걸렸다.

- [model-v2 공개 Release](https://github.com/gisbi-kim/tiny-visual-reasoning-lab/releases/tag/model-v2)
- archive: **174,218,456 bytes**
- SHA-256: `8bfc48b4850ae200d4320c77bddba03eea4f52e2b66f737910aa8b15ec296240`
- 압축 파일 크기·해시와 내부 파일 **15개**의 크기·해시를 모두 검사했다.
- [검사 JSON](public-download.json), [파일별 manifest](../release-v2.json)

기본 Qwen 모델은 v1과 동일한 고정 revision의 기존 Hugging Face 캐시를 사용했다. 설치된 Python 패키지도 재사용했다. 따라서 17.71초를 기본 모델까지 포함한 완전한 첫 설치 시간으로 해석하면 안 된다. 기본 모델 빈 캐시의 과거 v1 검사는 [기존 설치 기록](../INSTALL_CHECK.md)에 별도로 있다. 실제 Mac/MPS와 새 Python 환경의 패키지 설치는 이번 검사 범위에 포함하지 않았다.

## 다운로드한 가중치로 실제 HTTP 추론

```bash
python app.py --checkpoint weights-v2 --device cuda --port 7861
python verify_v2_public.py api
```

`/health`는 `ready: true`, `device: cuda`, `version: v2`를 반환했다. `/scene`에서 받은 실제 PNG 사진판을 `/predict`로 전달했으며, 모델을 대신하는 정답 규칙을 사용하지 않았다.

| 지시 요약 | 정답 | 실제 v2 |
|---|---|---|
| 개 | A | A |
| 바나나 | B | B |
| 배낭 | F | F |
| 고양이에서 왼쪽, 아래 | E | E |
| 고양이의 바로 오른쪽 | NONE | NONE |
| 동물 사진 3장 이상이면 바나나, 아니면 배낭 | B | **F (오답)** |
| 구명보트와 같은 줄의 동물 | E | E |

**6/7 정답**이었다. 기존 v1의 같은 시연도 6/7이었으므로 이 작은 시연에서는 향상이 없었다. 한 사진판의 고정 예제로, 전체 성능을 추정하는 시험은 아니다. [문장 원문·실제 확률·추론 시간](api-smoke.json)을 공개한다.

빈 지시는 HTTP 400, 무작위 사진판 생성과 앱 첫 화면은 HTTP 200이었다. 무작위 사진판이 기존 사진판과 달라졌으며 첫 화면에 v2 표시가 있는지도 확인했다. 기존 MNIST 앱의 7860 포트 역시 HTTP 200으로 응답했다. 기존 `weights/`와 `model-v1`은 보존했다.

학습용·배포 CUDA·CPU 구현에서 별도 프로세스로 가중치를 다시 읽은 24개 검증 표본은 모두 같은 답을 냈다(정확도 각각 23/24). 전체 최종 정확도는 [별도 평가 보고서](../../results/v2/finetuned/report.json), 재로딩 범위는 [reload-final.json](reload-final.json)을 참고한다.
