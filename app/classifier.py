import torch
from dataclasses import dataclass

@dataclass(frozen=True)
class LabelPrediction:
    probability: float
    predicted: bool

@dataclass(frozen=True)
class ClassificationResult:
    labels: dict[str, LabelPrediction]

class ToxicityClassifier:
    def __init__(
            self,
            model,
            tokenizer,
            thresholds,
            labels,
            device,
            max_length
    ):
        self.model = model
        self.labels = labels
        self.device = device
        self.max_length=max_length

        self.tokenizer = tokenizer
        self.thresholds = thresholds
        self.thresholds = torch.tensor(
            self.thresholds,
            device=self.device
        )

    def predict(self, text:str) -> ClassificationResult:
        """Classifies a unique input sequence"""

        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**tokenized_text)
            logits = outputs.logits

            probs = torch.sigmoid(logits)[0]

            predictions = probs >= self.thresholds

        return ClassificationResult(
            labels={
                label: LabelPrediction(
                    probability=float(probability.item()),
                    predicted=bool(predicted.item()),
                )
                for label, probability, predicted in zip(
                    self.labels,
                    probs,
                    predictions,
                )
            }
        )

    def predict_batch(self, texts:list[str]) -> list[ClassificationResult]:
        """Classifies a batch of input sequences,
        returns the multi-label result as a list of dictionaries"""

        tokenized_texts = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**tokenized_texts)
            logits = outputs.logits

            probs = torch.sigmoid(logits)

            predictions = probs >= self.thresholds

        results = [
            ClassificationResult(
                labels={
                    label: LabelPrediction(
                        probability=float(probability.item()),
                        predicted=bool(predicted.item()),
                    )
                    for label, probability, predicted in zip(
                        self.labels,
                        probability_row,
                        prediction_row,
                    )
                }
            )
            for probability_row, prediction_row in zip(probs, predictions)
        ]

        return results