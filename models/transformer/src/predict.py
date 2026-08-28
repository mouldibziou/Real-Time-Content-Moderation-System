from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_RESULTS_DIR = PROJECT_ROOT / "models" / "transformer" / "results"

MODEL_PATHS = {
    "DistilBERT": MODEL_RESULTS_DIR / "DistilBERT" / "Model6thEpoch",
    "HateBERT": MODEL_RESULTS_DIR / "HateBERT" / "Model6thEpoch",
}

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

THRESHOLD_VALUES = {
    "DistilBERT": torch.tensor([0.74, 0.45, 0.74, 0.59, 0.61, 0.76]),
    "HateBERT": torch.tensor([0.58, 0.49, 0.76, 0.46, 0.76, 0.72])
}

input_sentence = ""

while input_sentence != "stop":

    input_sentence = input("Write the sentence to classify: ")
    input_sentence = input_sentence.strip()

    if input_sentence == "stop":
        break

    print("\nAvailable models:")
    print("1. DistilBERT")
    print("2. HateBERT")

    choice = input("Select a model (1/2): ").strip()

    if choice == "1":
        selected_model = "DistilBERT"
    elif choice == "2":
        selected_model = "HateBERT"
    else:
        raise ValueError("Invalid model selection.")

    model_path = MODEL_PATHS[selected_model]
    thresholds = THRESHOLD_VALUES[selected_model]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    tokenized_text = tokenizer(
        input_sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        )

    model.eval()

    with torch.no_grad():
        outputs = model(**tokenized_text)
        logits = outputs.logits

        probs = torch.sigmoid(outputs.logits)[0]

        predictions = probs >= thresholds

    result = {
        label: int(pred.item())
        for label, pred in zip(LABEL_COLS, predictions)
    }

    print(result)

gc.collect()
del tokenizer, model, result, tokenized_text, input_sentence
