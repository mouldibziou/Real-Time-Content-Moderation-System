from dataclasses import dataclass
import torch
from captum.attr import LayerIntegratedGradients

@dataclass(frozen=True)
class AttributedSpan:
    text: str
    score: float
    normalized_importance: float | None

@dataclass(frozen=True)
class AttributionResult:
    label: str
    spans: tuple[AttributedSpan, ...]
    convergence_delta: float | None = None

class ToxicityExplainer:
    def __init__(
            self,
            model,
            tokenizer,
            labels: tuple[str, ...],
            device: torch.device,
            max_length: int,
            n_steps: int = 32,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        self.device = device
        self.max_length = max_length

        self._label_to_index = {
            label: index
            for index, label in enumerate(labels)
        }

        self._integrated_gradients = LayerIntegratedGradients(
            self._forward,
            self.model.get_input_embeddings(),
        )

        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        self.n_steps = n_steps

    def explain(self, text: str, label: str) -> AttributionResult:
        return self.explain_batch([text], label)[0]

    def explain_batch(
        self,
        texts: list[str],
        label: str,
    ) -> list[AttributionResult]:
        if label not in self._label_to_index:
            raise ValueError(f"Unknown label: {label}")

        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id")

        if not texts:
            return []

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get("token_type_ids")

        special_tokens_mask = encoded["special_tokens_mask"].bool()
        baseline_ids = input_ids.clone()
        baseline_ids[~special_tokens_mask] = self.tokenizer.pad_token_id

        attributions, convergence_delta = self._integrated_gradients.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask, token_type_ids),
            target=self._label_to_index[label],
            n_steps=self.n_steps,
            return_convergence_delta=True,
        )

        token_scores_by_text = attributions.sum(dim=-1)
        display_mask_by_text = attention_mask.bool() & ~special_tokens_mask
        results = []
        for input_id_row, token_scores, display_mask, delta in zip(
            input_ids,
            token_scores_by_text,
            display_mask_by_text,
            convergence_delta,
        ):
            displayed_scores = token_scores[display_mask]
            max_absolute_score = (
                displayed_scores.abs().max().item()
                if displayed_scores.numel() > 0
                else 0.0
            )
            tokens = self.tokenizer.convert_ids_to_tokens(input_id_row.tolist())

            spans = []
            for token, score, should_display in zip(
                tokens,
                token_scores,
                display_mask,
            ):
                if not should_display.item():
                    continue

                signed_score = score.item()
                normalized_importance = (
                    abs(signed_score) / max_absolute_score
                    if max_absolute_score > 0.0
                    else 0.0
                )
                spans.append(
                    AttributedSpan(
                        text=token,
                        score=signed_score,
                        normalized_importance=normalized_importance,
                    )
                )

            results.append(
                AttributionResult(
                    label=label,
                    spans=tuple(spans),
                    convergence_delta=delta.item(),
                )
            )

        return results

    
    def _forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        
        return self.model(**model_inputs).logits
