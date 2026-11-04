"""Explain the verified Decision API training example, with excerpts from its source."""
import hashlib
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _excerpt(source, start, end):
    return textwrap.dedent(source[source.index(start):source.index(end)]).rstrip()


def training_guide():
    folder = ROOT / 'examples/order_decision'
    source = (folder / 'train.py').read_text(encoding='utf-8')
    example = json.loads((folder / 'train.jsonl').read_text(encoding='utf-8').splitlines()[0])
    smoke = json.loads((folder / 'smoke-result.json').read_text(encoding='utf-8'))
    for name, digest in smoke['source_sha256'].items():
        normalized = (folder / name).read_text(encoding='utf-8').encode('utf-8')
        if hashlib.sha256(normalized).hexdigest() != digest:
            raise RuntimeError(f'Update the recorded smoke test after changing {name}')
    assert smoke['before_save_validation']['accuracy'] == 0.0
    assert smoke['after_reload_first_validation']['answer'] == 'leave'
    model_code = _excerpt(source, '    model, processor = FastDecisionModel.from_pretrained(',
                          '    train_items, train_report =')
    dataset_code = _excerpt(source, '    train_items, train_report =', '    training_args =')
    training_code = _excerpt(source, '    training_args = TrainingArguments(',
                             '    FastDecisionModel.for_inference(model)')
    saving_code = _excerpt(source, '    FastDecisionModel.for_inference(model)',
                           "    print('모델 저장:'")
    sample = json.dumps(example, ensure_ascii=False, indent=2)
    text = '''### 12.5 Decision 모델을 학습할 때는 무엇을 넣을까?

앞에서 `predict()`에 상황과 질문을 전달했다. 학습할 때도 입력은 같은 모양이다. 여기에 **그 상황에서 정답이 무엇인지**를 하나 더 넣어 주면 된다.

```text
학습: 상황(state) + 질문(questions) + 정답(gold) → 모델을 업데이트
추론: 상황(state) + 질문(questions)              → 모델이 답을 선택
```

사진판에만 적용되는 형식은 아니다. 이번에는 사진 없이 배송 문의의 처리 방법을 고르는 작은 예제로, 데이터가 모델 학습을 거쳐 추론까지 이어지는 모습을 살펴보자. 아래 코드는 Unsloth **2026.10.10**, Transformers **5.17.0**, Windows·NVIDIA CUDA 환경을 기준으로 한다. [Unsloth Decision 모델 구현](https://github.com/unslothai/unsloth/blob/main/unsloth/models/decision.py).

#### 1. 학습 데이터 한 행 만들기

주문이 아직 발송 전이고 새 주소를 받았다면 배송지를 수정한다고 가정하자. 아래는 실제 예제 파일의 첫 번째 학습 행이다.

```json
@@SAMPLE@@
```

`state`는 모델이 판단할 상황이다. `questions`는 질문의 이름, 형식과 선택지 설명이다. `gold`는 그 질문의 정답이다. 여기서는 질문 이름이 `next_action`이고, 정답 `update`가 `criteria`에 있는 선택지 이름과 일치한다. 사진 예제에서는 `state`에 자연어 지시를, `questions`에 A–F와 NONE을, `gold`에 정답 칸을 넣었다.

파일은 한 줄에 JSON 객체 하나씩 저장하는 **JSONL** 형식이다. 위 객체는 읽기 쉽도록 여러 줄로 펼쳐 썼다. 서로 다른 상황에 같은 질문을 붙이고, 각 상황에 맞는 정답을 지정한다. 질문이 여러 개면 `questions`와 `gold`에 같은 이름의 항목을 각각 넣는다.

이 예제는 학습 6행과 별도 검증 3행을 사용한다. [전체 학습 데이터](../examples/order_decision/train.jsonl) · [검증 데이터](../examples/order_decision/validation.jsonl). 두 파일은 API 사용 순서를 살펴보는 작은 예제다.

#### 2. 모델과 학습 데이터를 준비하기

먼저 데이터를 읽고 기반 모델에 새 decision head와 LoRA를 준비한다. 이어서 `build_dataset()`이 JSON 행을 모델이 학습할 수 있는 형태로 바꾼다. 아래에는 핵심 코드를 순서대로 나눠 적었다. 바로 실행하려면 [전체 학습 파일](../examples/order_decision/train.py)을 사용하면 된다.

```python
import json
import os
from pathlib import Path

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
from unsloth import FastDecisionModel, DecisionTrainer
import torch
from transformers import TrainingArguments

data_dir = Path("examples/order_decision")
output_dir = Path(".cache/order-decision-tutorial")
checkpoint = output_dir / "model"

def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]

train_rows = read_jsonl(data_dir / "train.jsonl")
validation_rows = read_jsonl(data_dir / "validation.jsonl")
```

```python
@@MODEL@@
```

여기서는 사진 실험의 v2 가중치가 아닌 **기본 Qwen3.5-4B**에서 출발한다. `decision_head="clef"`로 새 head를 만들고, `get_peft_model()`로 LoRA를 붙인다. 이미 LoRA가 들어 있는 체크포인트를 이어서 학습할 때는 새로운 LoRA를 또 붙이는 대신 저장된 것을 이어서 사용해야 한다.

이 코드에서 LoRA를 붙일 때 사용한 인자도 살펴보자.

| `get_peft_model()` 인자 | 역할 | 이 예제 |
|---|---|---|
| `model` | LoRA를 붙일 decision model | 방금 불러온 `model` |
| `r` | LoRA 행렬의 rank. 학습할 파라미터 수와 표현할 변화의 폭에 관여한다. | `16` |
| `lora_alpha` | LoRA 변화량의 배율을 정하는 값 | `16` |
| `lora_dropout` | LoRA 학습 시 적용할 dropout 비율 | `0` |
| `use_gradient_checkpointing` | 학습 중 메모리를 아끼기 위한 설정 | `"unsloth"` |
| `random_state` | 초기화 등에 사용할 난수 시드 | `3407` |

`from_pretrained()`의 `model_name`, `dtype`, `load_in_4bit` 등은 바로 앞 12.4절의 인자 표에서 다뤘다.

```python
@@DATASET@@
```

`FastDecisionModel.build_dataset(rows, tokenizer, model, validate=None)`의 인자는 다음과 같다.

| 인자 | 역할 |
|---|---|
| `rows` | `state`, `questions`, `gold`가 담긴 학습 또는 검증 행들 |
| `tokenizer` | 모델을 불러올 때 받은 tokenizer. 이미지가 있는 모델은 processor를 전달한다. |
| `model` | 입력 길이와 모델 종류를 확인할 decision model |
| `validate` | 선택 사항. 질문 이름과 정의를 받아 추가 검사를 하는 함수 |

반환값은 `(items, report)`다. `items`가 학습기에 들어갈 데이터이고, `report`에는 처리한 질문 수, 건너뛴 항목과 길이 초과 여부가 담긴다. 한 행에 질문이 여러 개면 `total`은 행 수가 아니라 질문 수로 늘어날 수 있다. 이 예제에서는 잘못 건너뛴 항목이나 잘린 입력이 있으면 학습 전에 확인하도록 했다.

#### 3. `DecisionTrainer`로 학습하기

```python
@@TRAINING@@
```

`DecisionTrainer`에 전달한 인자는 다음과 같다.

| 인자 | 역할 |
|---|---|
| `model` | 학습할 decision model |
| `args` | 배치 크기와 학습률 등이 담긴 `TrainingArguments` |
| `processing_class` | 모델 입력을 처리하는 tokenizer 또는 processor |
| `train_dataset` | `build_dataset()`으로 변환한 학습 데이터 |
| `eval_dataset` | 같은 방식으로 변환한 검증 데이터 |
| `head_learning_rate` | 선택용 head에 따로 적용할 학습률 |

여기서는 LoRA 쪽 `learning_rate`를 `0.0002`, head는 `0.00005`로 설정했다. 이 버전에서는 `head_learning_rate`를 생략하면 head에 `0.0001`을 사용한다. `trainer.train()`을 호출하면 학습이 시작된다.

`DecisionTrainer`에는 필요할 때 사용할 추가 인자도 있다. 이 예제에서는 지정하지 않았다.

| 선택 인자 | 역할 |
|---|---|
| `tokenizer` | `processing_class` 대신 tokenizer 또는 processor를 전달할 때 쓴다. |
| `label_smoothing` | 정답에 모든 가중치를 몰아주지 않고 일부를 다른 선택지로 나누는 설정이다. |
| `brier_weight` | 선택지 확률과 정답의 차이를 다루는 손실 항의 가중치다. |
| `ordinal_weight` | `score` 유형에서 선택한 단계와 정답 단계 사이의 거리를 반영하는 손실 항의 가중치다. |
| `kl_weight` | LoRA를 끈 기반 모델과 학습 시작 시점 head의 출력을 기준으로, 출력 분포의 차이를 줄이는 항의 가중치다. 이 버전에서는 Clef와 LoRA 조합에 적용한다. |
| `permute_fields` | 한 행에 질문이 여러 개일 때 학습 중 질문 순서를 섞을지 정한다. |
| `callbacks` | 학습 중 특정 step에서 검사하거나 기록하는 사용자 정의 동작을 전달한다. |
| `data_collator` | 예제들을 묶어 배치를 만드는 함수를 직접 지정한다. 생략하면 모델 종류에 맞는 collator를 사용한다. |

`TrainingArguments`의 주요 설정은 다음과 같다. [Hugging Face의 TrainingArguments 설명](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments).

| 설정 | 의미 | 이 예제 |
|---|---|---|
| `output_dir` | 학습 중 기록을 둘 폴더 | 지정한 출력 폴더 아래 `trainer` |
| `per_device_train_batch_size` | 장치 하나가 한 번에 처리할 예제 수 | `2` |
| `gradient_accumulation_steps` | 몇 번의 계산 결과를 모아 가중치를 한 번 갱신할지 | `1` |
| `max_steps` | 가중치를 갱신하는 횟수. 양수면 `num_train_epochs`보다 우선한다. | `2` |
| `learning_rate` | 학습 가능한 기반 모델 쪽 파라미터의 학습률 | `0.0002` |
| `bf16` | 지원 장치에서 bfloat16 연산을 쓸지 | `True` |
| `logging_steps` | 학습 상태를 기록할 간격 | 매 `1` step |
| `save_strategy`, `eval_strategy` | 학습 중 자동 저장과 자동 평가의 시점 | 둘 다 `"no"`; 아래에서 직접 검증·저장 |
| `report_to` | 외부 실험 추적 서비스로 기록을 보낼지 | `"none"` |

이 숫자들은 API의 기본값이 아니라 **이 예제에서 직접 지정한 값**이다. GPU 한 장에서 한 번에 2개씩, 누적 없이 2 step을 실행하므로 총 4회의 예제 제시가 이뤄진다. 여섯 행 전체를 반복 학습해 성능을 만들려는 설정이 아니라, 데이터 → 학습 → 저장 순서를 확인하는 설정이다.

#### 4. 검증하고 저장한 뒤 다시 불러오기

```python
@@SAVING@@
```

여기서는 자동 평가를 껐기 때문에 `FastDecisionModel.evaluate()`를 직접 호출해 별도 검증 행으로 정확도를 계산한다. 그다음 `model.save_pretrained()`로 학습 결과를, `processor.save_pretrained()`로 입력 처리에 필요한 파일을 저장한다. 이 LoRA 예제의 저장 폴더에는 어댑터와 decision head가 들어가고, 기반 Qwen 가중치는 별도로 불러온다.

`evaluate(model, tokenizer, items, batch_size=None)`에는 모델, 입력을 처리할 tokenizer 또는 processor, `build_dataset()`이 만든 검증 항목을 넣는다. `batch_size`는 한 번에 평가할 항목 수를 선택적으로 지정한다. 반환된 결과에서 `accuracy`와 `loss` 등을 확인할 수 있다.

이제 새 프로세스에서 저장 폴더를 `from_pretrained()`에 넘기고, 학습에 쓰지 않은 상황과 같은 형식의 질문으로 추론한다. **추론 호출에는 `gold`를 전달하지 않는다.** 아래는 [별도 추론 파일](../examples/order_decision/predict.py) 전체다.

```python
@@PREDICTION@@
```

전체 예제는 저장소 폴더에서 다음처럼 실행한다. 앞 절의 학습용 의존성과 CUDA용 PyTorch가 준비돼 있어야 한다. 기본 모델은 처음 사용할 때 내려받는다.

```bash
python examples/order_decision/train.py
python examples/order_decision/predict.py
```

기본 저장 위치는 `.cache/order-decision-tutorial`이다. 같은 이름의 저장된 모델이 있으면 두 명령에 `--output-dir`로 다른 폴더를 지정할 수 있다. 실제로 이 컴퓨터에서 학습하고 저장하기 전 검증했을 때 **3문제 중 정답은 0개**였다. 저장 후 다시 읽은 모델도 첫 검증 문제에서 기대 답인 `update` 대신 `leave`를 골랐다. 2 step짜리 시연이므로, 실제 서비스를 만들 때는 충분한 예제를 준비하고 학습량과 별도 평가셋을 설계해야 한다. [실행 검사 기록](../examples/order_decision/smoke-result.json).

#### 이미지가 들어가는 학습 행은?

같은 구조에 `images`만 더하면 된다. 다음은 이미지 경로와 정답을 예시로 적은 Python 코드다. 실제 사진에 맞는 정답을 별도로 준비해 넣는다.

```python
from PIL import Image

with Image.open("photos/parcel.jpg") as source:
    photo = source.convert("RGB")

row = {
    "state": "택배 사진을 보고 겉면의 파손 여부를 확인한다.",
    "images": [photo],
    "questions": {
        "damaged": {
            "type": "noul",
            "instructions": "택배 상자가 파손되었는가?",
        }
    },
    "gold": {"damaged": True},  # 이 예시 사진의 정답이라고 가정
}
items, report = FastDecisionModel.build_dataset([row], processor, model)
```

이때 `model`과 `processor`는 이미지를 처리할 수 있어야 한다. JSONL에 PIL 객체를 그대로 저장할 수는 없으므로 파일에는 이미지 경로를 적어 두고, 데이터를 읽을 때 이미지를 여는 방식으로 구성할 수 있다. 앞의 사진 실험도 사진판·지시·선택지·정답으로 학습하고, 추론할 때는 같은 사진판·지시·선택지만 전달했다. [이 프로젝트의 실제 데이터 변환 코드](../train.py).

'''
    prediction = (folder / 'predict.py').read_text(encoding='utf-8').strip()
    for key, value in {'@@SAMPLE@@': sample, '@@MODEL@@': model_code,
                       '@@DATASET@@': dataset_code, '@@TRAINING@@': training_code,
                       '@@SAVING@@': saving_code, '@@PREDICTION@@': prediction}.items():
        assert text.count(key) == 1
        text = text.replace(key, value)
    return text
