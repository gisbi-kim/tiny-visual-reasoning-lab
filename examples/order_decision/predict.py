"""저장한 모델을 다시 읽고 검증 예제의 상황과 질문으로 추론한다."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('UNSLOTH_COMPILE_DISABLE', '1')

from unsloth import FastDecisionModel
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='.cache/order-decision-tutorial')
    args = parser.parse_args()
    checkpoint = Path(args.output_dir) / 'model'

    data_path = Path(__file__).resolve().parent / 'validation.jsonl'
    with data_path.open(encoding='utf-8') as file:
        example = json.loads(next(file))

    model, processor = FastDecisionModel.from_pretrained(
        str(checkpoint), load_in_4bit=True, dtype=torch.bfloat16
    )
    FastDecisionModel.for_inference(model)
    answers = FastDecisionModel.predict(
        model, processor, example['state'], example['questions']
    )
    print('상황:', example['state'])
    print('모델 선택:', answers['next_action']['answer'])
    print('선택지별 확률:', answers['next_action']['probabilities'])


if __name__ == '__main__':
    main()
