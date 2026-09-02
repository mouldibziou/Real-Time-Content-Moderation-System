from types import SimpleNamespace
from unittest import TestCase

import torch
from transformers import BatchEncoding

from app.explainer import ToxicityExplainer


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, **kwargs) -> BatchEncoding:
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[101, 5, 6, 102, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
                "token_type_ids": torch.tensor([[0, 0, 0, 0, 0]]),
                "special_tokens_mask": torch.tensor([[1, 0, 0, 1, 1]]),
            }
        )

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [
            {0: "[PAD]", 5: "alpha", 6: "beta", 101: "[CLS]", 102: "[SEP]"}[token_id]
            for token_id in token_ids
        ]


class FakeSequenceClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(103, 2)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[5] = torch.tensor([2.0, -1.0])
            self.embedding.weight[6] = torch.tensor([-3.0, 4.0])

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        embeddings = self.embedding(input_ids)
        pooled = (embeddings * attention_mask.unsqueeze(-1)).sum(dim=1)
        return SimpleNamespace(logits=pooled)


class ToxicityExplainerTests(TestCase):
    def setUp(self) -> None:
        self.explainer = ToxicityExplainer(
            model=FakeSequenceClassifier(),
            tokenizer=FakeTokenizer(),
            labels=("first", "second"),
            device=torch.device("cpu"),
            max_length=8,
            n_steps=8,
        )

    def test_explain_targets_label_and_hides_special_tokens(self) -> None:
        explanation = self.explainer.explain("ignored", "first")

        self.assertEqual(explanation.label, "first")
        self.assertEqual([span.text for span in explanation.spans], ["alpha", "beta"])
        self.assertAlmostEqual(explanation.spans[0].score, 2.0, places=5)
        self.assertAlmostEqual(explanation.spans[1].score, -3.0, places=5)
        self.assertAlmostEqual(explanation.spans[0].normalized_importance, 2 / 3)
        self.assertAlmostEqual(explanation.spans[1].normalized_importance, 1.0)
        self.assertAlmostEqual(explanation.convergence_delta, 0.0, places=5)

        second_label = self.explainer.explain("ignored", "second")
        self.assertAlmostEqual(second_label.spans[0].score, -1.0, places=5)
        self.assertAlmostEqual(second_label.spans[1].score, 4.0, places=5)

    def test_explain_rejects_unknown_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown label"):
            self.explainer.explain("ignored", "missing")
