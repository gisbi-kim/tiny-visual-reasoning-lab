"""작은 예제로 Decision API의 학습·검증·저장 순서를 확인한다."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('UNSLOTH_COMPILE_DISABLE', '1')

from unsloth import DecisionTrainer, FastDecisionModel
import torch
from transformers import TrainingArguments


def read_jsonl(path):
    with path.open(encoding='utf-8') as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='.cache/order-decision-tutorial')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    checkpoint = output_dir / 'model'
    if checkpoint.exists():
        raise FileExistsError(f'다른 --output-dir을 지정해 주세요: {checkpoint}')

    data_dir = Path(__file__).resolve().parent
    train_rows = read_jsonl(data_dir / 'train.jsonl')
    validation_rows = read_jsonl(data_dir / 'validation.jsonl')

    model, processor = FastDecisionModel.from_pretrained(
        model_name='Qwen/Qwen3.5-4B',
        revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        decision_head='clef',
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    model = FastDecisionModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0,
        use_gradient_checkpointing='unsloth', random_state=3407,
    )

    train_items, train_report = FastDecisionModel.build_dataset(train_rows, processor, model)
    valid_items, valid_report = FastDecisionModel.build_dataset(validation_rows, processor, model)
    print('학습 데이터 검사:', train_report)
    print('검증 데이터 검사:', valid_report)
    for report in (train_report, valid_report):
        if report['skipped'] or report['truncated']:
            raise ValueError(f'데이터를 확인해 주세요: {report}')

    training_args = TrainingArguments(
        output_dir=str(output_dir / 'trainer'),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=2,  # 실행 흐름을 확인하기 위한 짧은 시연
        learning_rate=0.0002,
        bf16=True,
        logging_steps=1,
        save_strategy='no',
        eval_strategy='no',
        optim='adamw_torch',
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to='none',
        seed=3407,
    )
    FastDecisionModel.for_training(model)
    trainer = DecisionTrainer(
        model=model,
        args=training_args,
        processing_class=processor,
        train_dataset=train_items,
        eval_dataset=valid_items,
        head_learning_rate=0.00005,
    )
    trainer.train()

    FastDecisionModel.for_inference(model)
    print('검증 결과:', FastDecisionModel.evaluate(model, processor, valid_items))
    model.save_pretrained(str(checkpoint))
    processor.save_pretrained(str(checkpoint))
    print('모델 저장:', checkpoint)


if __name__ == '__main__':
    main()
