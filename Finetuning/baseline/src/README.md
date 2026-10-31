# Baseline Model — TF-IDF + Logistic Regression
## Overview
Multilabel toxic comment classifier with 6 labels: `toxic`, `severe_toxic`, `obscene`, 
`threat`, `insult`, `identity_hate`. A manual One-vs-Rest Logistic Regression is used, 
fitting one binary classifier per label independently.

This is the historical research baseline, not the deployed classifier. See the
[model card](../../../MODEL_CARD.md) for the serving release and evaluation scope.

---

## Results

| Split | Micro F1 | Macro F1 |
|-------|----------|----------|
| Train | 0.74     | 0.64     |
| Dev   | 0.69     | 0.64     |
| Gap   | 0.05     | 0.00     |

These are recorded development results. Similar train/dev macro F1 alone does not
establish the absence of underfitting or overfitting. The rare labels
`severe_toxic`, `threat`, and `identity_hate` have limited positive support, so
micro and macro averages should be read alongside per-label scores.

---

## Architecture

**Vectorizer — TF-IDF**
```python
TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.8,
    sublinear_tf=True
)
```

**Classifier — Manual One-vs-Rest Logistic Regression**
```python
LogisticRegression(
    class_weight={0: 1, 1: 2},
    max_iter=1000,
    C=1.0  # default
)
```
A uniform weight of `{0:1, 1:2}` is applied across all labels. Heavier weights 
like `{0:1, 1:3}` were tested but caused precision collapse on rare labels — the 
model began flagging a large proportion of sentences as toxic.

**Thresholds**
```python
thresholds = [0.5, 0.4, 0.5, 0.4, 0.5, 0.4]
# order: toxic, severe_toxic, obscene, threat, insult, identity_hate
```
Lower thresholds (0.4) are applied to rare labels to recover recall, as the model 
has limited signal on those labels due to low positive support.

---

## Known Limitations

- **Rare label metrics are unreliable** — `severe_toxic` (62 dev positives), 
  `threat` (46), and `identity_hate` (110) have insufficient dev support for 
  statistically stable evaluation. Treat their scores with caution.
- **Limited frequent-label performance** — recorded F1 for `toxic`, `obscene`, and
  `insult` is approximately 0.68–0.73; this motivated comparison with contextual encoders.
- **TF-IDF loses context** — character n-grams partially capture misspellings and 
  obfuscation (e.g. "st*pid") but cannot capture word order, sarcasm, or semantic 
  meaning.

---

## Next Steps
The next experiment compared contextual transformer encoders and end-to-end
fine-tuning. See the [transformer experiment notes](../../transformer/notebooks/README.md)
for the recorded results.
