from dataclasses import dataclass
import inspect
import torch
from captum.attr import LayerIntegratedGradients

@dataclass(frozen=True)
class AttributedSpan:
    text: str
    start: int
    end: int
    score: float
    normalized_importance: float | None

@dataclass(frozen=True)
class AttributionResult:
    label: str
    spans: tuple[AttributedSpan, ...]
    convergence_delta: float | None = None

def _model_supports_token_type_ids(model) -> bool:
    """Detect whether the model's forward() accepts token_type_ids.

    Falls back to True (previous behavior) when introspection fails,
    e.g. for heavily mocked models in tests.
    """
    try:
        return "token_type_ids" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return True

class ToxicityExplainer:
    """
    need gradient tracking, (not training mode). The service should run in eval mode.
    """
    def __init__(
            self,
            model,
            tokenizer,
            labels: tuple[str, ...],
            device: torch.device,
            max_length: int,
            n_steps: int = 32,
    ) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "ToxicityExplainer requires a fast tokenizer for offset-based spans"
            )

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

        # The saved DistilBERT tokenizer emits token_type_ids, which
        # DistilBERT does not accept. Only forward them to models whose
        # forward() declares the parameter (e.g. BERT-based models).
        self._supports_token_type_ids = _model_supports_token_type_ids(model)

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
            return_offsets_mapping=True,
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get("token_type_ids")
        offset_mapping = encoded["offset_mapping"]

        special_tokens_mask = encoded["special_tokens_mask"].bool()
        baseline_ids = input_ids.clone()
        baseline_ids[~special_tokens_mask] = self.tokenizer.pad_token_id

        attributions, convergence_delta = self._integrated_gradients.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask, token_type_ids),
            target=self._label_to_index[label],
            n_steps=self.n_steps,
            # Evaluate integration steps incrementally to bound worker memory.
            internal_batch_size=max(len(texts), 8),
            return_convergence_delta=True,
        )

        token_scores_by_text = attributions.sum(dim=-1)
        display_mask_by_text = attention_mask.bool() & ~special_tokens_mask
        results = []
        for text, token_scores, display_mask, offsets, delta in zip(
            texts,
            token_scores_by_text,
            display_mask_by_text,
            offset_mapping,
            convergence_delta,
        ):
            results.append(
                AttributionResult(
                    label=label,
                    spans=self._aggregate_spans(
                        text=text,
                        token_scores=token_scores,
                        display_mask=display_mask,
                        offsets=offsets,
                    ),
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

        if token_type_ids is not None and self._supports_token_type_ids:
            model_inputs["token_type_ids"] = token_type_ids
        
        return self.model(**model_inputs).logits

    def _aggregate_spans(
        self,
        text: str,
        token_scores: torch.Tensor,
        display_mask: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[AttributedSpan, ...]:
        raw_spans: list[tuple[int, int, float]] = []
        current_start: int | None = None
        current_end: int | None = None
        current_score = 0.0

        for score, should_display, offset in zip(
            token_scores,
            display_mask,
            offsets,
        ):
            if not should_display.item():
                continue

            start, end = (int(value) for value in offset.tolist())
            if end <= start:
                continue

            if current_start is None:
                current_start = start
                current_end = end
                current_score = score.item()
            elif start <= current_end:
                current_end = max(current_end, end)
                current_score += score.item()
            else:
                raw_spans.append((current_start, current_end, current_score))
                current_start = start
                current_end = end
                current_score = score.item()

        if current_start is not None and current_end is not None:
            raw_spans.append((current_start, current_end, current_score))

        max_absolute_score = max(
            (abs(score) for _, _, score in raw_spans),
            default=0.0,
        )
        return tuple(
            AttributedSpan(
                text=text[start:end],
                start=start,
                end=end,
                score=score,
                normalized_importance=(
                    abs(score) / max_absolute_score
                    if max_absolute_score > 0.0
                    else 0.0
                ),
            )
            for start, end, score in raw_spans
        )
