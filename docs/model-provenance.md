# Model identity and threshold provenance

## Serving artifact

The current default release is **DistilBERT v2.0.0**, loaded locally from `models/distilbert/v2.0.0`. This release name comes from the repository directory; it is not an upstream model revision.

The [artifact manifest](model-artifact-manifest.json) records SHA-256 hashes and sizes of the weights, model configuration, tokenizer files, and trainer-state record present when this document was created. These fingerprints identify the exact local files. Startup does not currently verify them, and hashes alone do not prove how the weights were trained.

| Property | Current value and evidence |
| --- | --- |
| Model class | `DistilBertForSequenceClassification`, from the saved configuration |
| Architecture | `distilbert`, six layers, six output labels |
| Task | Multilabel classification; sigmoid probabilities and independent thresholds |
| Base model | Training notes identify `distilbert-base-uncased`; an immutable upstream revision is not recorded here |
| Tokenizer | Local fast tokenizer; loaded from the same artifact directory |
| Runtime token limit | 512, including special tokens |
| Explanation | Layer Integrated Gradients, 32 steps by default, targeting the selected label's logit |

Sources: [saved model configuration](../models/distilbert/v2.0.0/config.json), [runtime settings](../app/settings.py), [resource loading](../app/resources.py), and [training notes](../Finetuning/transformer/notebooks/README.md).

The runtime validates the saved label order and output count before constructing the services. It also checks tokenizer compatibility and the configured token limit. These structural checks do not establish model quality or artifact provenance.

`model_info()` currently exposes the architecture and serving settings, not a release identifier or artifact fingerprint. The manifest supplies offline identity; adding a public version field is a separate implementation decision.

## Label order and thresholds

The classifier predicts a label when `sigmoid(logit) >= threshold`.

| Output index | Label | Threshold |
| --- | --- | --- |
| 0 | `toxic` | 0.74 |
| 1 | `severe_toxic` | 0.45 |
| 2 | `obscene` | 0.74 |
| 3 | `threat` | 0.59 |
| 4 | `insult` | 0.61 |
| 5 | `identity_hate` | 0.76 |

This vector agrees across [runtime settings](../app/settings.py), the [earlier prediction script](../Finetuning/transformer/src/predict.py), and the training notes' section on threshold tuning after epoch 6.

The [threshold-tuning notebook](../Finetuning/transformer/notebooks/threshold_tuning.ipynb) implements a sweep on development data. It chooses each label's threshold by **F2**, breaking ties by recall, precision, and then proximity to 0.50. The reported macro F1 is an evaluation result, not the threshold-selection objective.

The notebook describes an initial four-epoch run followed by two additional epochs. The local `trainer_state.json` records epoch `2.0`, global step `3144`, and best metric `0.8959233362453131`. This is consistent with the reported final untuned score, but does not independently prove the connection to the initial four-epoch run. The recorded checkpoint path is a historical training-environment path, not a runtime dependency.

## Evidence limits

The training notes report post-tuning DistilBERT macro precision `0.9131`, recall `0.9181`, and F1 `0.9148`. These are historical reported results, not scores reproduced by the runtime tests. The threshold notebook states that unseen test-set evaluation remains to be performed; do not present these figures as verified held-out test performance.

The repository review did not locate exported `best_thresholds.json` or `threshold_sweep.csv` files tying this exact weight fingerprint to the selected thresholds. The source notebook documents the method and chosen values, but complete reproducibility still needs the original run outputs, dataset/split fingerprints, training-code revision, and upstream model/tokenizer revision. These gaps must remain explicit rather than be filled with inferred identifiers.

The training script uses `max_position_embeddings - 2` (510 for this model), while the serving configuration intentionally uses 512. The saved model supports 512 positions. Evaluation of the final serving configuration should retain this distinction, especially for long inputs.

The real-model smoke test checks loading, prediction, explanation, and subsequent prediction consistency. It does not establish predictive accuracy, worst-case memory use, or the target throughput of 50 requests per second.

## Updating the release

For a future release, keep the weights, tokenizer, label mapping, thresholds, and evaluation evidence together. Regenerate the manifest whenever any recorded file changes, and retain the previous manifest with its release history. Record evaluation results for the actual serving configuration and distinguish development threshold selection from held-out testing.

No model binaries or training-state binaries are added to Git by this documentation change.
