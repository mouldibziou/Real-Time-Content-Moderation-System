# Final model
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
import os
from datetime import datetime
from pathlib import Path
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score



VERSION = "v1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "models" / "baseline" / VERSION
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Config
col_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
thresholds = np.array([0.5, 0.4, 0.5, 0.4, 0.5, 0.4])
balanced_weights = {0:1, 1:2}

# Importing Data
train = pd.read_csv(DATA_DIR / "train.csv")
dev = pd.read_csv(DATA_DIR / "dev.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

x_train = train['comment_text'].to_list()
x_dev = dev['comment_text'].to_list()
x_test = test['comment_text'].to_list()

y_train = train[col_labels].to_numpy()
y_dev = dev[col_labels].to_numpy()
y_test = test[col_labels].to_numpy()

# Vectorization
vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(1,3),
    min_df=3,
    max_df=0.8,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(x_train)
X_dev = vectorizer.transform(x_dev)
X_test = vectorizer.transform(x_test)

# Train
models = [
    LogisticRegression(class_weight=balanced_weights, max_iter=1000).fit(X_train, y_train[:, i])
    for i in range(len(col_labels))
]

# Evaluate
def evaluate(X, y, split_name):
    y_proba = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    y_hat = (y_proba >= thresholds).astype(int)
    print(f"\n{'='*60}\n{split_name}\n{'='*60}")
    print(classification_report(y, y_hat, target_names=col_labels, zero_division=0))
    return f1_score(y, y_hat, average="macro", zero_division=0)

train_macro_f1 = evaluate(X_train, y_train, "Training Set")
dev_macro_f1   = evaluate(X_dev,   y_dev,   "Dev Set")
test_macro_f1 = evaluate(X_test, y_test, "Test Set")

# Save
# models
for i, (model, label) in enumerate(zip(models, col_labels)):
    joblib.dump(model, f"{OUTPUT_DIR}/{label}_model.pkl")

# vectorizer
joblib.dump(vectorizer, f"{OUTPUT_DIR}/tfidf_vectorizer.pkl")

# thresholds
np.save(f"{OUTPUT_DIR}/threshold.npy", thresholds)

# Metadata
tfidf_params = vectorizer.get_params()
if "dtype" in tfidf_params:
    tfidf_params["dtype"] = str(tfidf_params["dtype"])

metadata = {
    "version": VERSION,
    "date": datetime.now().isoformat(),
    "labels": col_labels,
    "thresholds": thresholds.tolist(),
    "class_weight": balanced_weights,
    "logistic_regression": {"max_iter": 1000, "C": 1.0},
    "tfidf": tfidf_params,
    "metrics": {
        "train_macro_f1": round(train_macro_f1, 4),
        "dev_macro_f1":   round(dev_macro_f1, 4),
    }
}
with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nModel saved to {OUTPUT_DIR}/")
