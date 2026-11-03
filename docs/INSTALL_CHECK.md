# Public v1 installation and app check

Verified on 2026-11-03, Windows / RTX 4070 Ti SUPER, using public commit `4e7ebb3b87968040bf9cb7960109199eed0dbe0c` and the `model-v1` release.

1. Cloned the public GitHub repository into a new directory.
2. Downloaded the 174,182,037-byte release ZIP anonymously and verified its SHA-256 and all 15 extracted files.
3. Downloaded the original Stanford dataset again and verified its recorded SHA-256, then reconstructed 400 local demo crops.
4. Used an isolated, initially empty Hugging Face cache with implicit HF token use disabled. Downloaded the four base-model files required for inference (9,319,907,453 bytes); every file matched the SHA-256 in the original pinned-snapshot manifest.
5. Loaded the released weights and executed the 14 prewritten free-form cases. All 14 predictions matched the pre-release run: 12 correct, two incorrect.

The download/setup/inference sequence took **353.95 seconds (5 minutes 54 seconds)**; cloning occurred immediately beforehand and is excluded. This depends on the network and existing hardware and is not a promised startup time. **The installed Python environment was reused**: this did not test a fresh dependency installation or physical Mac MPS. Detailed evidence is in [install-check.json](install-check.json).

The verified public weights were then installed in the main project's `weights/` directory. The actual CUDA Flask app was started on port **7861** and tested through HTTP:

- Root page, health endpoint, canonical photo grid and random photo grid responded.
- Raw natural-language requests returned seven probabilities and actual model decisions.
- Seven prewritten queries on the illustrative board produced six correct decisions. This is an interface smoke test, not an independent accuracy benchmark.
- The query “동물 사진이 3장 이상이면 바나나를, 아니면 배낭을 골라줘.” should select **B**, but the model selected **F**. The example was retained; it was not replaced with a successful one.
- Empty instructions were rejected with HTTP 400. Inline JavaScript passed `node --check`.
- The existing MNIST app on port 7860 still returned HTTP 200.

HTTP responses, instructions and probabilities are in [api-smoke.json](api-smoke.json). Physical browser layout was not inspected; the checks above cover JavaScript syntax and real HTTP inference. Source images are local and are not included in these reports.
