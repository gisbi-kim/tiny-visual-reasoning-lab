# Data card

This is a bounded visual instruction-following benchmark built from Tiny ImageNet photographs. It is not a benchmark of unrestricted language reasoning, nor a dataset of chain-of-thought explanations. Targets are discrete choices; symbolic traces are retained only for audit.

## Construction and provenance

The original Stanford archive is recorded with its byte size and SHA-256 in `dataset-download.json`. The 40 selected synsets, bilingual aliases and semantic groups are in `catalog.json`. A fixed manual review examined five training crops per candidate class, before any final-test evaluation. This is sample-based class curation, not exhaustive human annotation.

Images with annotated objects smaller than 20 pixels along either dimension or 512 pixels in area are rejected. The filter removed 2,664 small-object photos and three exact RGB duplicates across the selected sources. Source bounding boxes are converted from inclusive coordinates to PIL's exclusive coordinates. Bounding-box crops preserve aspect ratio, fit in a 112×112 region and occupy six 128×128 tiles. Final inputs are 384×256 RGB. No class name, answer letter or oracle trace is drawn into the image supplied to the model. The UI's A–F labels are a browser overlay.

Each board contains six distinct object classes. Semantic counting refers to tiles' annotated main objects, not every visible object. Some groups are intentionally narrow in this subset: `boat` currently includes lifeboat, and `kitchenware` includes teapot. A teddy bear is a toy, not an animal. Foods such as bell pepper are counted as food; fruit follows the explicit task catalog rather than botanical taxonomy.

Each generated program must resolve to a unique tile or NONE. Ambiguous programs are discarded. Missing anchors, missing requested targets and steps outside the grid yield NONE. No wrapping is allowed. Consecutive moves are perpendicular, so they cannot cancel each other. Counts and strict/non-strict comparison rules are represented explicitly in programs.

## Language diversity

Each of 12 task families has six authored semantic forms in each language. The ten training families use forms 0–3; forms 4–5 are held out. Lexical aliases, polite/conversational framing, imperative/question wording and defaults/negated conditions add variation. A seeded renderer generates text from the program without seeing its answer. 29,350 distinct training strings do not imply 29,350 distinct semantic structures. No paid API or language-model paraphrase generation was used.

Two entirely held-out composition families combine existing primitives in new orders. Four test suites separate photo generalization, wording generalization, composition generalization, and their combination. A small Korean copula fix affected 23 held-out wording rows before final evaluation; training/validation/calibration rows were unchanged (`language-review.json`).

The actual 31,360 training rows are balanced across family × language × action. Conditional task families are additionally balanced across true/false outcomes. There are 3,148 accepted minimal contrast pairs when each language is counted separately. Pair members use the same authored form and lexical seed, change a direction/threshold/target/category comparison, and have different answers. Other instructions may share a scene without being a minimal pair.

## Splits

For each class, original training photos are deterministically shuffled after filtering: 40 for validation, 20 for calibration, the remainder for training. Labeled original validation photos supply the final test pool. Original unlabeled test photos are not used.

IDs and exact full-image RGB hashes are disjoint across these four source pools. The four final test suites share the final-test source pool, so their results are correlated. A source photograph can appear in multiple boards within its pool. Reported scene-cluster bootstrap intervals do not fully account for that dependence. Near-duplicate photographs and possible Qwen pretraining exposure are not audited.

The checkpoint-selection subset contains 140 balanced validation rows; temperature calibration uses a separate 560-row split. Final test labels do not select model weights or hyperparameters. Selection accuracy on the 140-row subset is not final-test accuracy.

## Automated checks and limits

All generated answers are recomputed from their programs. All instructions must reproduce exactly from their stored form and random seed. All source-pool and held-out-form/family assertions must pass. Unit tests cover program semantics, grid edges, ambiguity, horizontal transformations and the exclusion of labels/programs from inference input. An encoder probe also checks that pixels and raw language affect the actual processor inputs.

Remaining limitations include small and sometimes ambiguous photographs, finite templates, synonym/group boundaries, limited subjects, repeated source photos across scenes, and synthetic layouts. Better benchmark scores alone do not establish general reasoning ability or fluent handling of every user phrasing.

## Availability

Code, class metadata, aggregate audits and model metrics are published. Photos and rendered boards are downloaded/generated locally from the original Stanford archive; neither is bundled in the Git repository or trained-weight release. Image rights remain with their original owners. Code/model licensing does not grant rights to the photographs.
