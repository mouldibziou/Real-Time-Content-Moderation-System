# Transformer Model Selection and Fine-Tuning

This notebook folder compares multiple pretrained transformer encoders for multilabel toxic comment classification, starting with a feature extraction setup and later proceeding to full fine-tuning of the strongest candidates.

## Evaluated Models

* RoBERTa-base
* BERTweet-base
* HateBERT
* DistilBERT-base-uncased
* BERT-large-uncased

## Initial Feature Extraction Methodology

To run quick and fair experiments on a subset of the full dataset, we first used a feature extraction pipeline:

1. Run each transformer in inference mode.
2. Extract the encoder `[CLS]` representation.
3. Train a `OneVsRestLogisticRegression` classifier on top of the extracted features.

This setup was optimized for efficiency with:

* Dynamic/optimized batch size based on available compute.
* A controlled maximum sequence length to reduce outlier-driven memory and latency issues.

## Feature Extraction Results

From the model-selection experiment results:

* Best micro F1: DistilBERT-base-uncased (`0.64`)
* Best macro F1: HateBERT (`0.50`)
* Lowest macro/micro F1 in this comparison: RoBERTa-base (`0.36` / `0.59`)

## Initial Interpretation

* Performance differences were moderate across models, especially on frequent labels.
* Rare labels remained difficult for all tested encoders, similar to baseline behavior.
* RoBERTa underperformed in this specific feature extraction setting. This is unusual given its general strength and suggested that full fine-tuning could significantly change the ranking.

## Fine-Tuning Candidate Selection

Based on the initial score/computation trade-offs:

* Keep for fine-tuning:

  * DistilBERT-base-uncased
  * HateBERT
  * RoBERTa-base

* Drop:

  * BERT-large-uncased due to its higher compute cost and weaker overall trade-off.
  * BERTweet-base because it appeared less aligned with this dataset domain than expected.

## Initial Fine-Tuning Results

The three selected candidates — DistilBERT, RoBERTa-base, and HateBERT — were fine-tuned for 4 epochs each.

The two strongest candidates after this stage were:

* HateBERT: macro F1 = `0.8540`
* DistilBERT: macro F1 = `0.8445`

These two models were selected for further experimentation.

### Threshold Fine-Tuning After Epoch 4

After fine-tuning the decision threshold for each toxicity label, the macro F1 scores improved to:

* HateBERT: `0.8666`
* DistilBERT: `0.8786`

Selected thresholds:

* DistilBERT: `[0.60, 0.40, 0.69, 0.68, 0.67, 0.48]`
* HateBERT: `[0.66, 0.53, 0.57, 0.69, 0.47, 0.55]`

At this point, DistilBERT overtook HateBERT after threshold optimization.

## Extended Fine-Tuning

The two selected models were then fine-tuned for two additional epochs using the same training techniques.

This resulted in:

* Lower training loss
* Lower validation loss
* Improved evaluation metrics

Before threshold optimization, the 6th-epoch models achieved:

* HateBERT: macro F1 = `0.8888`
* DistilBERT: macro F1 = `0.8959`

### Threshold Fine-Tuning After Epoch 6

Threshold fine-tuning was repeated on the final 6th-epoch checkpoints.

Final macro F1 scores:

* HateBERT: `0.9011`
* DistilBERT: `0.9148`

Selected thresholds:

* DistilBERT: `[0.74, 0.45, 0.74, 0.59, 0.61, 0.76]`
* HateBERT: `[0.58, 0.49, 0.76, 0.46, 0.76, 0.72]`

## Final Results

### DistilBERT

* Precision: `0.9131`
* Recall: `0.9181`
* Macro F1: `0.9148`

### HateBERT

* Precision: `0.8879`
* Recall: `0.9168`
* Macro F1: `0.9011`

The final results clearly favor DistilBERT.

## Final Model Decision

**DistilBERT-base-uncased is selected as the primary model.**

It currently provides the strongest overall performance while remaining comparatively lightweight and efficient.

HateBERT will remain the secondary candidate and can be useful for:

* A/B testing
* Benchmarking future model versions
* Ensemble experiments
* Alternative production configurations

The model selection may be revisited after further fine-tuning or experimentation with additional transformer architectures.

## Comparison With Baseline

Baseline test performance from the classical machine learning pipeline:

* Micro F1: `0.69`
* Macro F1: `0.64`

During the initial feature extraction stage, transformer performance was lower by approximately:

* `~0.05` on micro F1
* `~0.14` on macro F1

However, full end-to-end fine-tuning significantly changed the results.

The final DistilBERT model reached a macro F1 of:

`0.9148`

compared with the baseline macro F1 of:

`0.64`

This represents an absolute improvement of approximately:

`+0.2748 macro F1`

The experiments therefore confirm that feature extraction alone significantly underestimated the potential of the transformer models for this task.
