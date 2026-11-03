All 12 supported task families and all 144 authored Korean/English forms are now trained, including those held out in v1. V2 continues the v1 adapter/head for 148.1 minutes on one RTX 4070 Ti SUPER.

On the exact original 1,120 examples per suite:

| Original v1 evaluation | v1 | v2 | Change |
|---|---:|---:|---:|
| Held-out wording | 79.29% | 95.71% | +16.43 pp |
| Held-out composition | 67.77% | 95.71% | +27.95 pp |
| Both | 54.82% | 95.62% | +40.80 pp |

These wording/composition types are included in v2 training; test photographs remain excluded. The fixed freeform demo regressed from 12/14 to 11/14, so improvement on authored templates does not establish general natural-language improvement. Native/CUDA/CPU reload predictions agree on the 24-example validation probe. The run completed 4,704 steps (one epoch); validation selected step 3,600 for release.

On the same 2,016 photo-held-out examples: v1 84.52% → v2 95.78%. This tests trained instruction types on new photographs, not novel-task generalization. Detailed family scores, failures, provenance and limitations are in docs/V2_TRAINING.md.

The archive contains trained LoRA, the Clef head, processor and calibration. Qwen/Qwen3.5-4B at revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a downloads separately from Hugging Face; Tiny ImageNet downloads separately from Stanford. V1 remains available as model-v1.

Download: `python download_model.py --version v2 --output weights-v2`
Run: `python app.py --checkpoint weights-v2 --device cpu` (or cuda).
