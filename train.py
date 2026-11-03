"""Train the actual multimodal Unsloth Clef model with a wall-clock budget."""
import os
os.environ.setdefault('UNSLOTH_COMPILE_DISABLE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('PYTHONUTF8','1')
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import time
from PIL import Image
from schema import model_input

ROOT = Path(__file__).resolve().parent

def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    temp.replace(path)

def read_rows(name):
    return [json.loads(s) for s in (ROOT/'data'/f'{name}.jsonl').read_text(encoding='utf-8').splitlines()]

def encode(api,model,processor,rows):
    records = []
    for row in rows:
        record = model_input(str(ROOT/'data'/row['image']), row['instruction'])
        record['gold'] = {'select':row['action']}
        records.append(record)
    items,report = api.build_dataset(records,processor,model)
    print('DATASET',report,flush=True)
    if len(items)!=len(rows) or report.get('skipped') or report.get('truncated'):
        raise RuntimeError(f'Dataset must encode without loss: {report}')
    return items

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',default='main')
    parser.add_argument('--steps',type=int,default=6000)
    parser.add_argument('--seconds',type=int,default=9000)
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--accum',type=int,default=1)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--head-lr',type=float,default=1e-4)
    parser.add_argument('--validate-every',type=int,default=100)
    parser.add_argument('--four-bit',action='store_true')
    parser.add_argument('--overfit',action='store_true')
    parser.add_argument('--val-count',type=int,default=140)
    parser.add_argument('--revision',default='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
    args = parser.parse_args()
    from unsloth import FastDecisionModel, DecisionTrainer
    import torch
    from transformers import TrainingArguments, TrainerCallback
    run = ROOT/'runs'/args.run
    if (run/'adapter').exists(): raise RuntimeError('Use a new run name to preserve previous weights')
    run.mkdir(parents=True,exist_ok=True)
    def status(phase,**extra):
        dump(run/'status.json',dict(phase=phase,updated_at=datetime.now(timezone.utc).isoformat(),**extra))
    status('loading')
    base = 'Qwen/Qwen3.5-4B'
    revision = args.revision
    model,processor = FastDecisionModel.from_pretrained(model_name=base,revision=revision,
        max_seq_length=1024,load_in_4bit=args.four_bit,dtype=torch.bfloat16,
        random_state=3407,use_gradient_checkpointing='unsloth')
    if not hasattr(processor,'image_processor'): raise RuntimeError('Vision processor is required')
    model = FastDecisionModel.get_peft_model(model,r=16,lora_alpha=16,lora_dropout=0,
        random_state=3407,use_gradient_checkpointing='unsloth')
    model.decision_config['base_revision'] = revision
    train_rows = read_rows('train')
    if args.limit: train_rows = train_rows[:args.limit]
    val_rows = train_rows if args.overfit else read_rows('validation_select')[:args.val_count]
    # PEFT accounts for packed Params4bit logical elements; raw tensor.numel()
    # would undercount a quantized backbone.
    encoder_trainable,encoder_total=model.encoder.get_nb_trainable_parameters()
    head_total=sum(p.numel() for p in model.head.parameters())
    quantization=getattr(model.encoder.config,'quantization_config',None)
    if quantization is not None:
        dump(run/'quantization.json',quantization.to_dict() if hasattr(quantization,'to_dict') else quantization)
    config = dict(**vars(args),base_model=base,base_revision=revision,
                  train_rows=len(train_rows),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  total_parameters=encoder_total+head_total,head_parameters=head_total,
                  lora_trainable_parameters=encoder_trainable,parameter_count_note='Logical parameter count via PEFT, including packed 4-bit backbone elements.',
                  gpu=torch.cuda.get_device_name(),gpu_vram=torch.cuda.get_device_properties(0).total_memory,
                  versions={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','unsloth','unsloth_zoo')},
                  max_length=1024,lora_rank=16,lora_alpha=16,seed=3407,
                  trainable_names=[n for n,p in model.named_parameters() if p.requires_grad])
    dump(run/'config.json',config)
    (run/'requirements.lock.txt').write_text(subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    status('encoding')
    items = encode(FastDecisionModel,model,processor,train_rows)
    eval_items = encode(FastDecisionModel,model,processor,val_rows)
    dump(run/'encoding.json',{'train_count':len(items),'max_tokens':max(len(i['input_ids']) for i in items),
        'min_tokens':min(len(i['input_ids']) for i in items),'skipped':0,'truncated':0})
    validation_history=[]
    best = -1.
    start = None
    def save_checkpoint(destination):
        model.save_pretrained(str(destination))
        processor.save_pretrained(str(destination))
        if quantization is not None:dump(destination/'quantization.json',quantization.to_dict() if hasattr(quantization,'to_dict') else quantization)
    def validate(step):
        nonlocal best
        FastDecisionModel.for_inference(model)
        predictions=[]
        for row in val_rows:
            with Image.open(ROOT/'data'/row['image']) as im:
                result = FastDecisionModel.predict(model,processor,row['instruction'],model_input(im,'x')['questions'],images=[im.convert('RGB')])['select']
            predictions.append(dict(gold=row['action'],**result))
        accuracy = sum(p['gold']==p['answer'] for p in predictions)/len(predictions)
        validation_history.append(dict(step=step,accuracy=accuracy,predictions=predictions))
        dump(run/'validation.json',validation_history)
        print('VALIDATION',step,accuracy,flush=True)
        if accuracy > best:
            best = accuracy
            save_checkpoint(run/'best')
        FastDecisionModel.for_training(model)
        return accuracy
    class Progress(TrainerCallback):
        def on_train_begin(self,arguments,state,control,**kwargs):
            nonlocal start
            start=time.monotonic()
        def on_log(self,arguments,state,control,logs=None,**kwargs):
            elapsed = time.monotonic()-start if start else 0
            status('training',step=state.global_step,max_steps=state.max_steps,elapsed_seconds=elapsed,logs=logs,
                   peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30)
            dump(run/'history.json',state.log_history)
        def on_step_end(self,arguments,state,control,**kwargs):
            if args.validate_every and state.global_step % args.validate_every == 0:
                status('validation',step=state.global_step)
                validate(state.global_step)
            if time.monotonic()-start >= args.seconds:
                control.should_training_stop=True
                print('Reached training wall-clock budget',flush=True)
    FastDecisionModel.for_training(model)
    trainer = DecisionTrainer(model=model,processing_class=processor,train_dataset=items,eval_dataset=eval_items,
        head_learning_rate=args.head_lr,callbacks=[Progress()],
        args=TrainingArguments(output_dir=str(run/'checkpoints'),per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,max_steps=args.steps,learning_rate=args.lr,
            warmup_steps=min(60,max(1,args.steps//10)),lr_scheduler_type='cosine',weight_decay=.01,
            bf16=True,fp16=False,logging_steps=1 if args.steps<30 else 10,eval_strategy='no',
            save_strategy='steps',save_steps=200,save_total_limit=2,report_to='none',seed=3407,
            dataloader_num_workers=0,dataloader_pin_memory=False,optim='adamw_torch',disable_tqdm=True))
    torch.cuda.reset_peak_memory_stats()
    status('training',step=0)
    result=trainer.train()
    dump(run/'metrics.json',dict(**result.metrics,peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30,
                               peak_reserved_gb=torch.cuda.max_memory_reserved()/2**30))
    validate(trainer.state.global_step)
    save_checkpoint(run/'adapter')
    dump(run/'files.json',{p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                          for p in (run/'adapter').iterdir() if p.is_file()})
    status('trained',step=trainer.state.global_step,best_validation=best)
    print('TRAINED',run,flush=True)

if __name__=='__main__': main()
