"""공개 v2 모델을 Unsloth의 CUDA 추론 방식으로 실행한다."""
import json
import os
from pathlib import Path

os.environ.setdefault('UNSLOTH_COMPILE_DISABLE', '1')

from PIL import Image
from unsloth import FastDecisionModel
import torch

from schema import questions


def main():
    # 사진 여섯 장이 A–F 순서로 합쳐진 이미지다.
    image_path = Path('docs/figures/example-canonical.png')
    with Image.open(image_path) as source:
        image = source.convert('RGB')

    instruction = '고양이에서 왼쪽으로 한 칸, 그다음 아래로 한 칸 이동한 사진을 골라줘.'
    model, processor = FastDecisionModel.from_pretrained(
        'weights-v2', load_in_4bit=True, dtype=torch.bfloat16
    )
    FastDecisionModel.for_inference(model)

    result = FastDecisionModel.predict(
        model, processor, instruction, questions(), images=[image]
    )['select']
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
