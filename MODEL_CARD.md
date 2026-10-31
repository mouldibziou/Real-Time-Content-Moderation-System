# Flagged model card

## Purpose

Flagged provides multilabel toxicity signals and moderator-facing explanations for user-submitted text. It supports human review; predictions are not definitive judgments about a person or conversation.

| Property | Packaged release |
| --- | --- |
| Release identifier | `v2.0.0` (project release, not an upstream revision) |
| Base architecture | `distilbert-base-uncased`, six transformer layers |
| Task | Six independent binary labels with sigmoid probabilities |
| Classification | FP32 ONNX Runtime on CPU; deployed through NVIDIA Triton |
| Explanation | Captum Layer Integrated Gradients targeting a selected label logit; 32 steps by default |
| Serving limit | 512 tokens including special tokens; longer input is truncated |

## Predictions and thresholds

Predict a label when its probability is greater than or equal to its threshold. Multiple labels can apply; probabilities do not sum to one.

| Output index | Label | Threshold |
| --- | --- | ---: |
| 0 | `toxic` | 0.74 |
| 1 | `severe_toxic` | 0.45 |
| 2 | `obscene` | 0.74 |
| 3 | `threat` | 0.59 |
| 4 | `insult` | 0.61 |
| 5 | `identity_hate` | 0.76 |

The [threshold notebook](Finetuning/transformer/notebooks/threshold_tuning.ipynb) selects per-label thresholds on development data using **F2**, with tie-breaking by recall, precision, and proximity to 0.50. Macro F1 is a reported evaluation metric, not the threshold-selection objective.

## Research and evaluation

The research sources include [data exploration and preprocessing](Finetuning/eda/), a [TF-IDF / logistic regression baseline](Finetuning/baseline/src/README.md), and [encoder comparison and fine-tuning](Finetuning/transformer/notebooks/README.md). The recorded experiment sequence uses four fine-tuning epochs followed by two additional epochs for the strongest candidates.

| Historical reported result | Value |
| --- | ---: |
| Baseline development macro F1 | 0.64 |
| DistilBERT macro F1 before final threshold tuning | 0.8959 |
| DistilBERT macro precision after tuning | 0.9131 |
| DistilBERT macro recall after tuning | 0.9181 |
| DistilBERT macro F1 after tuning | 0.9148 |

**These are historical development results, not independently reproduced held-out test scores.** The threshold notebook leaves unseen test-set evaluation as follow-up work. Runtime smoke tests check functional behavior rather than predictive accuracy.

Complete experiment reproduction still requires the original data/split fingerprints, run outputs, training-code revision, and immutable upstream model/tokenizer revision. The repository does not contain a complete release-linked evaluation bundle. The training script uses a 510-token limit for this architecture; serving uses 512, which should be retained when evaluating the deployed configuration.

## Artifact identity and availability

Model binaries and datasets are kept outside Git. Contact the [project author](README.md#contact) for matching serving artifacts. Required paths are listed in the [setup instructions](README.md#backend).

[`docker/model-manifest.json`](docker/model-manifest.json) records SHA-256 hashes and sizes for the PyTorch weights, model configuration, tokenizer files, and ONNX export. Docker builds run [`docker/verify_artifacts.py`](docker/verify_artifacts.py) against those five files.

Runtime loading additionally validates label order, output count, tokenizer compatibility, and token limits. These checks establish structural compatibility; artifact hashes identify files but do not prove training lineage. `/v1/model_info` reports serving metadata and the packaged release name, not a weight fingerprint.

## Limitations and review

- Limited conversational context and input truncation can change the interpretation of a comment.
- Rare labels require particular care in evaluation. Development threshold tuning does not establish generalization to new communities or distributions.
- Token attributions show contributions to a model output; they do not establish the author's intent or a causal explanation of harmful behavior.
- Public-demo corrections are untrusted feedback. Review them before using them in offline retraining; no automatic model update occurs.

For a new release, keep the weights, tokenizer, label mapping, thresholds, artifact manifest, and evaluation evidence together. Evaluate the actual serving configuration on an untouched test set and distinguish it from development-time model and threshold selection.
