"""A general, version-scoped guide to the Decision API used in the article."""


def api_guide():
    return '''### 12.4 이 예제를 벗어나서: Decision API의 입력과 출력

앞의 `questions()`는 우리 사진 선택 문제에 맞춰 만든 함수다. Unsloth에 원래 들어 있는 특별한 함수 이름은 아니다. 여기서 잠깐 사진 문제를 벗어나, `FastDecisionModel`에 어떤 정보를 줄 수 있는지 살펴보자. 아래 설명은 이 실험에 설치된 Unsloth **2026.10.10**의 Python 구현을 기준으로 한다. [Unsloth의 Decision 모델 구현](https://github.com/unslothai/unsloth/blob/main/unsloth/models/decision.py) · [질문 형식 처리 코드](https://github.com/unslothai/unsloth/blob/main/unsloth/models/_decision_common.py).

#### 먼저 모델을 준비한다: `from_pretrained()`

모델을 불러오는 함수의 주요 인자는 다음과 같다. 아래 기본값은 위에서 밝힌 버전의 `FastDecisionModel` 기준이다.

```python
model, processor = FastDecisionModel.from_pretrained(
    model_name="경로 또는 모델 ID",
    load_in_4bit=True,
    dtype=torch.bfloat16,
)
FastDecisionModel.for_inference(model)
```

| 인자 | 기본값 | 하는 일 |
|---|---|---|
| `model_name` | 필수 | 로컬 체크포인트 폴더 경로나 Hugging Face 모델 ID다. 우리 예제에서는 `"weights-v2"`다. |
| `subfolder` | `None` | 모델 저장소나 폴더 안의 특정 하위 폴더를 읽을 때 사용한다. |
| `max_seq_length` | `None` | 모델을 준비할 때 사용할 입력 길이 설정이다. 생략하면 해당 모델 로더의 기본값을 쓴다. |
| `dtype` | `None` | 계산에 사용할 데이터 타입을 지정한다. 예를 들어 지원 장치에서 `torch.bfloat16`을 쓸 수 있다. |
| `load_in_4bit` | `False` | 지원되는 모델을 4bit 양자화로 불러와 메모리를 절약한다. |
| `load_in_8bit` | `False` | 이 버전의 Decision 모델에서는 `True`를 지원하지 않는다. |
| `full_finetuning` | `False` | 이어서 학습할 때 전체 가중치 학습 여부를 정하는 설정이다. |
| `token` | `None` | 접근 권한이 필요한 Hugging Face 모델을 받을 때 사용할 인증 토큰이다. |
| `revision` | `None` | Hugging Face 모델의 특정 브랜치·태그·커밋을 지정한다. |
| `local_files_only` | `False` | 체크포인트를 찾을 때 로컬 파일과 캐시를 사용하도록 요청한다. 어댑터가 참조하는 기본 모델 파일도 별도로 준비해야 한다. |
| `use_gradient_checkpointing` | `"unsloth"` | 학습할 때 중간값 저장을 줄여 메모리 사용을 조정하는 설정이다. |
| `random_state` | `3407` | 모델 준비와 학습 과정에서 사용하는 난수 시드다. |
| `**kwargs` | — | 사용하는 모델 로더에 전달할 추가 설정이다. 예를 들어 일반 기반 모델에 `decision_head="clef"`를 지정할 수 있다. |

반환되는 `model`은 판단할 모델이고, 두 번째 값은 텍스트용 tokenizer 또는 이미지도 처리하는 processor다. 앞의 예제에서는 이미지가 필요하므로 변수 이름을 `processor`로 붙였다. `for_inference(model)`은 모델을 평가 모드로 준비한다. **이 단계는 모델을 준비하는 과정이고, 실제 질문은 다음 `predict()`에서 한다.**

이미 학습된 decision 체크포인트를 불러오는 경우에는 저장된 head를 읽는다. 반면 일반 기반 모델에서 새 head를 만드는 경우에는, 그 head가 원하는 판단을 하도록 데이터로 학습시키는 과정이 필요하다. 우리 `weights-v2`는 학습을 마친 head와 LoRA가 함께 저장된 체크포인트다.

#### 질문한다: `predict()`

```python
def predict(model, tokenizer, state, questions, *, images=None) -> dict:
    ...
```

위 줄은 인자의 위치를 보여 주는 **함수 시그니처**다. `*`는 설명용 표기이며, 실제 호출에서는 쓰지 않는다. `images`는 `images=[image]`처럼 이름을 붙여 전달한다.

| 인자 | 역할 | 사진 예제에서는 |
|---|---|---|
| `model` | `from_pretrained()`로 불러온 decision model이다. | Qwen과 학습된 LoRA·head를 담은 `model` |
| `tokenizer` | 모델 입력을 만드는 tokenizer다. 이미지가 있는 Clef 모델에서는 `from_pretrained()`가 돌려준 **processor**를 넣는다. | `processor` |
| `state` | 모델이 판단할 상황이나 문맥이다. 문자열을 쓸 수 있고, Clef에서는 JSON으로 표현할 수 있는 구조도 전달할 수 있다. | 사용자가 쓴 자연어 지시문 |
| `questions` | 무엇을 판단하고 어떤 종류의 답을 낼지 정하는 딕셔너리다. 한 번에 이름이 다른 질문 여러 개를 담을 수 있다. | `questions()`가 만든 `select` 질문 |
| `images` | 선택 사항. 이미지 입력을 지원하는 모델과 processor에 사진을 전달한다. PIL 이미지 목록을 사용할 수 있으며 목록의 순서가 유지된다. | `[image]`: 사진 여섯 장을 합친 사진판 한 장 |

텍스트만으로 판단한다면 `images`를 생략할 수 있다. 이미지를 쓰려면 해당 이미지 입력을 처리할 수 있는 모델과 processor가 필요하다. 이미지 여러 장을 API에 직접 넘길 수도 있지만, 우리 실험은 여섯 사진을 한 장의 격자로 합쳐 전달했다. [Unsloth 멀티 이미지 지원 관련 기록](https://github.com/unslothai/unsloth/issues/13461).

#### `questions` 안에는 무엇을 넣을까?

예를 들어 배송 주문의 처리 방법을 고른다고 해 보자. `state`가 현재 상황을, `questions`가 그 상황에 대해 모델이 답할 질문을 담는다.

```python
state = "아직 발송 전인 주문이며, 구매자가 배송지 변경을 요청했다."

questions = {
    "next_action": {
        "type": "choice",
        "instructions": "이 주문을 어떻게 처리할까?",
        "criteria": {
            "update": "배송지를 수정한다.",
            "contact": "추가 정보를 확인한다.",
            "leave": "주문을 변경하지 않는다.",
        },
    }
}

# model과 tokenizer는 앞에서 준비한 decision model과 tokenizer다.
answers = FastDecisionModel.predict(model, tokenizer, state, questions)
print(answers["next_action"]["answer"])
```

`next_action`은 우리가 정한 **질문 이름**이다. 반환값에서도 같은 이름으로 결과를 꺼낸다. `type`은 답의 종류, `instructions`는 질문, `criteria`는 고를 수 있는 답과 그 의미를 담는다. `instructions`를 비우면 Clef는 질문 이름을 설명으로 사용한다. `update`, `contact`, `leave` 역시 사용자가 정한 선택지 이름이다. 위 코드는 API의 형태를 보여 주는 예이며, 배송 주문을 판단하려면 그 목적에 맞는 모델과 학습·평가가 필요하다.

사진 문제에서는 같은 구조의 질문 이름을 `select`로 정하고 `criteria`에 A–F와 NONE을 넣었다. `questions`는 정답을 알려주는 표가 아니라 **답을 고를 틀**을 전달하는 것이다. 학습 데이터에서는 별도로 `gold`에 정답을 넣고, 추론할 때는 넣지 않는다.

#### 질문의 세 가지 종류와 반환값

이 버전의 API에서 질문의 `type`은 다음 세 가지다.

| `type` | 질문하는 방식 | `criteria` | 반환되는 `answer` |
|---|---|---|---|
| `choice` | 여러 선택지 중 하나를 고른다. | 선택지 이름과 설명의 딕셔너리, 또는 선택지 이름 목록 | 선택된 이름, 예: `"update"` |
| `noul` | 예·아니요를 판단한다. | 생략하거나 `true`·`false`의 뜻을 설명하는 딕셔너리 | `True` 또는 `False` |
| `score` | 순서가 있는 단계 중 어느 수준인지 판단한다. | 낮은 단계부터 순서대로 적은 설명 목록 | 가장 높은 확률을 받은 단계의 번호. 첫 단계는 `0` |

예를 들어 `"priority": {"type": "score", "instructions": "처리 우선순위는?", "criteria": ["낮음", "보통", "높음"]}`처럼 정의할 수 있다. 질문을 더 만들고 싶다면 `questions` 안에 이름과 정의를 추가하면 된다.

반환값은 질문 이름을 키로 하는 딕셔너리다. `choice`의 결과에는 선택한 `choice`, 그 선택의 `confidence`, 선택지별 `probabilities`와 편하게 꺼낼 수 있는 `answer`가 들어간다. `noul`은 참일 확률을 `noul`에, 최종 참·거짓을 `answer`에 담는다. `score`에서는 단계별 확률과 설명을 볼 수 있고, `score`는 단계 번호의 확률 가중 평균, `answer`는 가장 가능성 높은 단계 번호다. 이 실험의 A–F 문제는 세 종류 중 `choice`를 사용했다.

`state`와 `questions`는 역할이 다르다. `state`는 **지금 무슨 상황인가**, `questions`는 **그 상황에 대해 무엇을 어떤 답의 형식으로 물을 것인가**를 담는다. 앞에서 본 사진 앱은 이 구분을 이용해 자연어 지시와 일곱 선택지를 모델에 함께 전달한 것이다.

'''
