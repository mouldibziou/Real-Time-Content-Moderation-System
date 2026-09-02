# Future implementations

## Mixed-label batched explanations

Add an explicit production API for explaining a batch in which each text may target a different label.

The API should accept paired inputs, for example `[(text, label), ...]`, validate every label, and return results in input order. Internally, group requests by label and execute one batched Integrated Gradients call per label group. This preserves the current attribution semantics: each IG calculation targets exactly one output logit.

Before implementation, decide request limits, batching and memory policy, error behavior for invalid labels, and whether the runtime exposes the capability only for flagged labels or also for arbitrary labels.
