import joblib
import numpy as np
import os
from pathlib import Path

VERSION = "v1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "models" / "baseline" / VERSION

col_labels    = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
vectorizer    = joblib.load(f"{OUTPUT_DIR}/tfidf_vectorizer.pkl")
models        = [joblib.load(os.path.join(OUTPUT_DIR, f"{label}_model.pkl")) for label in col_labels]
thresholds    = np.load(f"{OUTPUT_DIR}/threshold.npy")

def predict(texts):
    X = vectorizer.transform(texts)
    y_proba = np.column_stack([m.predict_proba(X)[:, 1] for m in models])
    y_hat   = (y_proba >= thresholds).astype(int)
    return y_hat, y_proba

# Insert the text to see the predictions of the baseline model.

texts = [
    "I will kill you",
    "Have a great day!",
    "You are so stupid and ugly",
]

y_hat, y_proba = predict(texts)

for i, text in enumerate(texts):
    predicted_labels = [col_labels[j] for j in range(len(col_labels)) if y_hat[i, j] == 1]
    
    print(f"\nText: {text}")
    print(f"Predicted labels : {predicted_labels if predicted_labels else 'none'}")


# Free RAM

del vectorizer, models, thresholds, texts, y_hat, y_proba
