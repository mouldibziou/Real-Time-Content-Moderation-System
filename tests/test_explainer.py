from types import SimpleNamespace
from unittest import TestCase

import torch
from transformers import BatchEncoding

from app.explainer import ToxicityExplainer


class FakeTokenizer:
    pad_token_id = 0
    is_fast = True

    def __call__(self, text: str, **kwargs) -> BatchEncoding:
        batch_size = len(text) if isinstance(text, list) else 1
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[101, 5, 6, 102, 0]]).repeat(batch_size, 1),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]).repeat(batch_size, 1),
                "token_type_ids": torch.tensor([[0, 0, 0, 0, 0]]).repeat(batch_size, 1),
                "special_tokens_mask": torch.tensor([[1, 0, 0, 1, 1]]).repeat(batch_size, 1),
                "offset_mapping": torch.tensor([[0, 0], [0, 2], [2, 12], [0, 0], [0, 0]]).repeat(batch_size, 1, 1),
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
        explanation = self.explainer.explain("unacceptable", "first")

        self.assertEqual(explanation.label, "first")
        self.assertEqual([span.text for span in explanation.spans], ["unacceptable"])
        self.assertEqual((explanation.spans[0].start, explanation.spans[0].end), (0, 12))
        self.assertAlmostEqual(explanation.spans[0].score, -1.0, places=5)
        self.assertAlmostEqual(explanation.spans[0].normalized_importance, 1.0)
        self.assertAlmostEqual(explanation.convergence_delta, 0.0, places=5)

        second_label = self.explainer.explain("unacceptable", "second")
        self.assertAlmostEqual(second_label.spans[0].score, 3.0, places=5)

    def test_explain_rejects_unknown_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown label"):
            self.explainer.explain("ignored", "missing")

    def test_explain_batch_returns_one_result_per_text(self) -> None:
        explanations = self.explainer.explain_batch(
            ["unacceptable", "unacceptable"],
            "first",
        )

        self.assertEqual(len(explanations), 2)
        for explanation in explanations:
            self.assertEqual(explanation.label, "first")
            self.assertEqual([span.text for span in explanation.spans], ["unacceptable"])
