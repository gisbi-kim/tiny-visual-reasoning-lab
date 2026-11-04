"""Add runnable inference examples to the article from the checked-in scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def add_inference_examples(text):
    native = (ROOT / 'example_unsloth_inference.py').read_text(encoding='utf-8').strip()
    portable = (ROOT / 'example_app_inference.py').read_text(encoding='utf-8').strip()
    section = f'''## 12. 이미지부터 선택까지, 추론 코드 살펴보기

앞에서는 모델이 얼마나 잘 고르는지 살펴봤다. 이번에는 같은 모델을 코드에서 직접 불러 사진과 지시를 건네보자. 아래 예제는 글 첫머리에 등장한 사진판을 사용한다.

```text
A: 개       B: 바나나   C: 고양이
D: 구명보트 E: 금붕어   F: 배낭
```

“고양이에서 왼쪽으로 한 칸, 그다음 아래로 한 칸”이라는 지시를 주면 C → B → E이므로 정답은 E다. 학습에 사용한 Unsloth 코드와 앱이 사용하는 추론 코드를 각각 실행해 보자.

### 12.1 이미지 여섯 장을 사진판으로 만들기

모델에는 여섯 장의 파일을 따로 넣는 대신 **두 줄로 이어 붙인 이미지 한 장**을 넣는다. 맨 왼쪽 위부터 A, B, C, 아래 줄이 D, E, F다. 예제에 사용할 사진판은 저장소의 `docs/figures/example-canonical.png`에 이미 들어 있다.

직접 사진 여섯 장을 준비했다면, 프로젝트의 `render` 함수로 같은 모양을 만들 수 있다. 파일 이름은 자신의 사진에 맞게 바꾸면 된다.

```python
from PIL import Image
from render import render

# 파일을 A, B, C, D, E, F 순서로 나열한다.
paths = [
    "photos/a.jpg", "photos/b.jpg", "photos/c.jpg",
    "photos/d.jpg", "photos/e.jpg", "photos/f.jpg",
]

photos = []
for path in paths:
    with Image.open(path) as source:
        photos.append(source.convert("RGB"))

board = render(photos)       # 여섯 장을 384×256 사진판 한 장으로 합친다.
board.save("my-board.png")
```

`render()`는 사진의 비율을 유지해 각 칸에 배치한다. 이 이미지와 한국어 지시가 모델에 들어간다. 앱에서 만든 사진판도 같은 함수를 사용한다. [사진판을 만드는 코드](../render.py) · [앱의 사진 읽기](../app.py).

### 12.2 Unsloth로 학습한 모델 불러오기

이 프로젝트에서는 **Unsloth의 `FastDecisionModel`로 학습하고, 학습 중 검증할 때도 `predict()`를 호출했다.** 먼저 저장소에서 v2 가중치를 받는다. 아래 코드는 실험에 사용한 Windows·NVIDIA CUDA 환경과 Unsloth `2026.10.10`을 기준으로 한다. CUDA용 PyTorch를 설치한 뒤 학습용 의존성을 설치하면 된다.

```bash
python -m pip install -r requirements-train.txt
python -m pip install bitsandbytes==0.50.2
python download_model.py --version v2 --output weights-v2
```

`weights-v2`는 학습한 LoRA와 선택용 head를 담고 있다. 기본 Qwen 가중치는 첫 추론 때 Hugging Face에서 별도로 받아 캐시에 저장한다. 이미 `weights-v2`를 받았다면 다운로드 명령은 생략한다.

이제 사진을 읽고 모델을 불러와 질문해 보자. 아래는 저장소의 [실행 가능한 예제](../example_unsloth_inference.py) 전체다. 저장소 폴더에서 `python example_unsloth_inference.py`로 실행한다.

```python
{native}
```

순서대로 보면 다음과 같다.

1. `Image.open()`으로 사진판을 읽고 RGB 이미지로 바꾼다. 직접 만든 사진판을 쓰려면 `image_path`만 바꾸면 된다.
2. `from_pretrained()`는 `weights-v2`의 설정을 읽어 기본 Qwen, 학습한 LoRA와 head를 불러온다. `load_in_4bit=True`는 기본 모델의 GPU 메모리 사용량을 줄이는 설정이다.
3. `for_inference()`로 추론을 준비한 뒤, `predict()`에 모델·processor·지시·선택지 정의·사진판을 전달한다. `images=[image]`에 들어간 것은 여섯 장이 이어진 **사진판 한 장**이다. 코드의 `UNSLOTH_COMPILE_DISABLE`은 이 실험에서 사용한 컴파일 설정이다.
4. 반환값에서 `select`는 이번 선택 문제에 대한 결과다. 그 안의 `answer`가 A–F 또는 NONE이고, `probabilities`에는 각 선택지의 점수가 확률 형태로 들어 있다.

`processor`는 사진과 문장을 Qwen이 처리할 수 있는 입력으로 바꾼다. 그 입력을 Qwen이 읽고, head가 일곱 선택지의 점수를 만든다. 이때 `generate()`로 답변 문장을 토큰마다 이어 쓰는 대신 `predict()`가 선택 결과를 돌려준다.

이 컴퓨터에서 위 코드를 실행한 출력은 다음과 같았다.

```json
{{
  "type": "choice",
  "choice": "E",
  "confidence": 1.0,
  "probabilities": {{
    "A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0,
    "E": 1.0, "F": 0.0, "NONE": 0.0
  }},
  "answer": "E"
}}
```

C에서 왼쪽 B를 거쳐 아래 E로 이동한 결과다. 이번 Unsloth 버전은 출력 확률을 소수점 넷째 자리까지 반올림하므로 `1.0`과 `0.0`으로 표시되었다. 여러 문제를 연달아 풀 때는 모델을 한 번 불러온 뒤 이미지와 지시만 바꿔 `predict()`를 다시 호출하면 된다.

### 12.3 모델은 일곱 선택지를 어떻게 알까?

위 코드의 `questions()`는 선택지의 뜻을 모델 입력에 함께 넣기 위한 함수다. 예를 들어 다음 코드를 실행하면 각 문장이 무엇을 고르는지 볼 수 있다.

```python
import json
from schema import questions

print(json.dumps(questions()["select"]["criteria"], indent=2))
```

```json
{{
  "A": "Choose the top left photograph.",
  "B": "Choose the top middle photograph.",
  "C": "Choose the top right photograph.",
  "D": "Choose the bottom left photograph.",
  "E": "Choose the bottom middle photograph.",
  "F": "Choose the bottom right photograph.",
  "NONE": "No photograph satisfies the instruction."
}}
```

`questions()`에는 이 표와 함께 사진판의 배치, 한 칸 이동과 NONE에 관한 설명도 들어 있다. 이 정의는 **어떤 답을 고를 수 있는지** 알려 준다. 어느 칸이 정답인지는 넣지 않는다. 모델은 사진과 사용자의 지시를 읽고 일곱 선택지에 점수를 준다. [전체 선택지 정의](../schema.py) · [학습 중 실제 검증 코드](../train.py).

### 12.4 앱에서는 어떻게 실행할까?

공개 앱은 Mac의 MPS와 CPU에서도 실행할 수 있도록 Unsloth 대신 **Transformers·PEFT와 Clef head를 조합한 `DecisionEngine`**을 사용한다. 우리가 바로 위에서 본 Unsloth 추론은 학습·검증 코드에서 사용한 방식이고, 아래는 앱과 같은 방식을 직접 호출하는 예제다.

```python
{portable}
```

저장소에서 `python example_app_inference.py`로 실행한다. `device="auto"`는 사용할 수 있는 CUDA, MPS, CPU 순서로 장치를 고른다. 같은 Windows GPU에서 확인했을 때 이 코드도 **E**를 골랐다. [앱의 실제 추론 코드](../engine.py).

`DecisionEngine` 안에서는 사진·지시·선택지를 모델 입력으로 바꾸고, Qwen과 학습된 head를 거쳐 일곱 점수(logits)를 얻는다. 그다음 실제 코드에서 다음처럼 확률을 계산하고 가장 높은 선택지를 반환한다.

```python
# engine.py의 predict()에서 발췌
logits = self.logits(image, instruction)
probs = (logits / self.temperature).softmax(-1).cpu().tolist()
```

앱의 `self.temperature`에는 앞서 1,008문제로 구한 보정값이 반영된다. 위 Unsloth 예제는 모델 설정의 온도를 사용하고 별도의 `calibration.json`은 적용하지 않으므로, 두 코드의 선택이 같더라도 출력 확률은 다를 수 있다. 같은 가중치를 다시 읽은 검증 표본 24개에서는 Unsloth, 앱 CUDA, 앱 CPU가 모두 같은 선택을 했다. [재로딩 비교 기록](v2/reload-final.json).

'''
    marker = '## 12. 가중치의 출처와 다시 실행하기'
    if text.count(marker) != 1 or text.count('## 13. 실험 기록 살펴보기') != 1:
        raise RuntimeError('Article headings changed; inference section needs a new insertion point')
    text = text.replace('## 13. 실험 기록 살펴보기', '## 14. 실험 기록 살펴보기', 1)
    return text.replace(marker, section + '## 13. 가중치의 출처와 다시 실행하기', 1)
