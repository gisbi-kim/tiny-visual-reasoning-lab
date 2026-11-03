Qwen3.5-4B + trained LoRA + Clef selection head for six-photograph Tiny ImageNet grids.

- Local training: RTX 4070 Ti SUPER 16 GB, 4,400 steps, 135.7 minutes.
- Released checkpoint: step 2,400, selected using validation only.
- Dataset: 31,360 bilingual instructions, 40 object classes, 4,024 scenes. Source-photo splits and exact RGB deduplication audited.
- Held-out accuracy (1,120 examples each): new photos 93.48%; new wording 79.29%; new compositions 67.77%; both 54.82%. Novel wording/composition generalization remains limited.

The ZIP contains the trained LoRA, decision head, processor, quantization profile and calibration. It does **not** include the approximately 9.34 GB Qwen base model or Tiny ImageNet photos. The base downloads separately from Hugging Face at a pinned revision; photos download from Stanford.

Run `python download_model.py` to download and verify the ZIP and every extracted file against `docs/release.json`. Follow the repository README for the local app and data preparation. CPU and CUDA are checked; physical Mac MPS is unverified.

Code, complete metrics, failures, data-generation methodology and training provenance are in the repository. This is a bounded visual selection experiment, not an unrestricted reasoning assistant.
