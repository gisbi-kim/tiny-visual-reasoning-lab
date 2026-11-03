"""Portable real Clef inference on CPU/CUDA/MPS. No oracle or label access."""
import json
import math
import os
from pathlib import Path
import time
import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from clef_reference import JointSchemaHead, ClefModel, encode_record, collate_records
from schema import ACTIONS, model_input, questions

def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize(device)
    elif device.type=='mps': torch.mps.synchronize()

class DecisionEngine:
    def __init__(self, checkpoint, device='auto'):
        if device=='auto':
            device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable. Try --device cpu.')
        if device=='mps' and not torch.backends.mps.is_available(): raise RuntimeError('MPS is unavailable. Try --device cpu.')
        self.device=torch.device(device)
        self.cpu_fallback_enabled=device=='mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1'
        self.dtype=torch.bfloat16 if device=='cuda' else torch.float32
        if device=='cpu': torch.set_num_threads(min(8,torch.get_num_threads()))
        path=Path(checkpoint)
        schema_file=path/'inference_schema.json'
        if schema_file.exists():
            saved_schema=json.loads(schema_file.read_text(encoding='utf-8'))
            if saved_schema['actions']!=list(ACTIONS) or saved_schema['questions']!=questions():
                raise RuntimeError('Saved action schema does not match this application version')
        self.config=json.loads((path/'unsloth_decision_config.json').read_text())
        self.quantized=self.device.type=='cuda' and self.config.get('load_in_4bit',False)
        kwargs={}
        if self.quantized:
            profile=json.loads((path/'quantization.json').read_text())
            kwargs={'quantization_config':BitsAndBytesConfig.from_dict(profile),'device_map':{'':str(self.device)}}
        base=AutoModelForImageTextToText.from_pretrained(self.config['base_model'],
            revision=self.config['base_revision'],dtype=self.dtype,attn_implementation='eager',**kwargs)
        self.encoder=PeftModel.from_pretrained(base,str(path),is_trainable=False)
        saved=load_file(str(path/'adapter_model.safetensors'))
        actual=get_peft_model_state_dict(self.encoder)
        if set(saved)!=set(actual): raise RuntimeError('LoRA keys do not match saved adapter')
        if any(not torch.equal(saved[k].to(actual[k].dtype),actual[k].cpu()) for k in saved):
            raise RuntimeError('Loaded LoRA values do not match saved adapter')
        del saved,actual
        head=JointSchemaHead(**json.loads((path/'joint_head_config.json').read_text()))
        head.load_state_dict(load_file(str(path/'joint_head.safetensors')),strict=True)
        if not self.quantized:self.encoder.to(device=self.device,dtype=self.dtype)
        head.to(device=self.device,dtype=torch.float32)
        head.register_forward_pre_hook(lambda module,args:(args[0].float(),*args[1:]))
        self.model=ClefModel(self.encoder,head).eval()
        self.processor=AutoProcessor.from_pretrained(str(path))
        if not hasattr(self.processor,'image_processor'): raise RuntimeError('Vision processor missing')
        temps=self.config.get('temperature',[1.,1.,1.])
        if self.config.get('temperature_by_options'):
            raise RuntimeError('Per-option calibration requires a matching portable implementation')
        self.temperature=float(self.config.get('head_temperature',1))*float(temps[0])
        calibration=path/'calibration.json'
        self.calibration=json.loads(calibration.read_text()) if calibration.exists() else {'temperature':1.0}
        self.temperature*=self.calibration['temperature']
        if not 0.0001<self.temperature<10000: raise RuntimeError('Invalid temperature')

    @torch.inference_mode()
    def logits(self,image,instruction):
        record=encode_record(self.processor.tokenizer,model_input(image.convert('RGB'),instruction),
            max_length=16384,processor=self.processor)
        if len(record.input_ids)>int(self.config.get('max_len',1024)):
            raise ValueError('Instruction/image exceeds the trained context length. Please use a shorter instruction.')
        batch=collate_records([record],self.processor.tokenizer.pad_token_id,self.device)
        for key,value in (batch.get('media') or {}).items():
            if torch.is_tensor(value) and value.is_floating_point(): batch['media'][key]=value.to(self.dtype)
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
            raw=self.model(batch)[0][0].float()
        mapping=dict(zip(record.questions[0].option_ids,raw))
        return torch.stack([mapping[a] for a in ACTIONS])

    def predict(self,image,instruction):
        synchronize(self.device)
        start=time.perf_counter()
        logits=self.logits(image,instruction)
        probs=(logits/self.temperature).softmax(-1).cpu().tolist()
        synchronize(self.device)
        duration=time.perf_counter()-start
        if not all(math.isfinite(p) for p in probs): raise RuntimeError('Nonfinite model output')
        return {'action':ACTIONS[max(range(len(ACTIONS)),key=lambda i:probs[i])],
                'probabilities':dict(zip(ACTIONS,probs)),'latency_ms':duration*1000,
                'device':str(self.device),'dtype':str(self.dtype),
                'quantized_base':self.quantized,
                'cpu_fallback_enabled':self.cpu_fallback_enabled}
