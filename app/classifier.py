import torch

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

    def predict(self, text:str):
        """Classifies a unique input sequence,
        returns the multi-label result as a JSON object"""

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

        result = {
            label: int(pred.item())
            for label, pred in zip(self.labels, predictions)
        }

        return result

    def predict_batch(self, texts:list[str]):
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
            {label: int(pred.item())
             for label, pred in zip(self.labels, row)}
            for row in predictions
        ]

        return results