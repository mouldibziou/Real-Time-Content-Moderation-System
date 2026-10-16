import numpy as np
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
    """
    needs the model in inference mode, to get the output logits.
    """
    def __init__(
            self,
            model,
            tokenizer,
            thresholds,
            labels,
            device,
            max_length,
            inference_batch_size: int = 4,
    ):
        if type(inference_batch_size) is not int or inference_batch_size < 1:
            raise ValueError("Inference batch size must be a positive integer")
        self.inference_batch_size = inference_batch_size
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

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        )
        # NOTE: the saved tokenizer returns token_type_ids, which DistilBERT
        # does not accept. Only forward the inputs the model supports.
        tokenized_text = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

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
        # Bound model activations independently of the public request batch size.
        results = []
        for offset in range(0, len(texts), self.inference_batch_size):
            results.extend(self._predict_batch(texts[offset:offset + self.inference_batch_size]))
        return results

    def _predict_batch(self, texts:list[str]) -> list[ClassificationResult]:
        """Classifies a batch of input sequences,
        returns the multi-label result as a list of dictionaries"""

        tokenized_texts = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        )
        # NOTE: see predict(): drop token_type_ids for DistilBERT compatibility.
        tokenized_texts = {
            "input_ids": tokenized_texts["input_ids"].to(self.device),
            "attention_mask": tokenized_texts["attention_mask"].to(self.device),
        }

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

class ToxicityOnnxClassifier:
    """
    ONNX Runtime counterpart of ToxicityClassifier.

    Same predict/predict_batch interface and ClassificationResult output,
    but inference runs through an onnxruntime InferenceSession instead of
    torch. The torch model is still required elsewhere for explanations,
    which need gradients that ONNX cannot provide.
    """
    def __init__(
            self,
            session,
            tokenizer,
            thresholds,
            labels,
            max_length,
            inference_batch_size: int = 4,
    ):
        if type(inference_batch_size) is not int or inference_batch_size < 1:
            raise ValueError("Inference batch size must be a positive integer")
        self.inference_batch_size = inference_batch_size
        self.session = session
        self.labels = labels
        self.max_length = max_length

        self.tokenizer = tokenizer
        self.thresholds = np.asarray(thresholds, dtype=np.float64).reshape(1, -1)

    def _logits(self, texts:list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=self.max_length
        )
        feed = {
            "input_ids": np.asarray(encoded["input_ids"], dtype=np.int64),
            "attention_mask": np.asarray(encoded["attention_mask"], dtype=np.int64),
        }
        return np.asarray(self.session.run(None, feed)[0], dtype=np.float64)

    @staticmethod
    def _to_results(texts:list[str], labels, probs:np.ndarray, predictions:np.ndarray) -> list[ClassificationResult]:
        return [
            ClassificationResult(
                labels={
                    label: LabelPrediction(
                        probability=float(probability),
                        predicted=bool(predicted),
                    )
                    for label, probability, predicted in zip(
                        labels,
                        probability_row,
                        prediction_row,
                    )
                }
            )
            for probability_row, prediction_row in zip(probs.tolist(), predictions.tolist())
        ]

    def predict(self, text:str) -> ClassificationResult:
        """Classifies a unique input sequence"""
        logits = self._logits([text])
        probs = 1.0 / (1.0 + np.exp(-logits))
        predictions = probs >= self.thresholds
        return self._to_results([text], self.labels, probs, predictions)[0]

    def predict_batch(self, texts:list[str]) -> list[ClassificationResult]:
        results = []
        for offset in range(0, len(texts), self.inference_batch_size):
            results.extend(self._predict_batch(texts[offset:offset + self.inference_batch_size]))
        return results

    def _predict_batch(self, texts:list[str]) -> list[ClassificationResult]:
        """Classifies a batch of input sequences,
        returns the multi-label result as a list of dictionaries"""
        logits = self._logits(list(texts))
        probs = 1.0 / (1.0 + np.exp(-logits))
        predictions = probs >= self.thresholds
        return self._to_results(list(texts), self.labels, probs, predictions)
