export const LABELS = [
  "toxic",
  "severe_toxic",
  "obscene",
  "threat",
  "insult",
  "identity_hate",
] as const;
export type Label = (typeof LABELS)[number];
export const LABEL_NAMES: Record<Label, string> = {
  toxic: "Toxic",
  severe_toxic: "Severe toxicity",
  obscene: "Obscene",
  threat: "Threat",
  insult: "Insult",
  identity_hate: "Identity hate",
};
export interface Prediction {
  probability: number;
  predicted: boolean;
}
export interface Classification {
  text: string;
  labels: Record<Label, Prediction>;
  classification_id: string;
}
export interface ModelInfo {
  model_type: string;
  device: string;
  labels: Label[];
  thresholds: number[];
  max_length: number;
  explanation_steps: number;
  inference_backend: string;
  model_version: string;
  max_text_chars: number;
  max_prediction_batch_size: number;
}
export interface Span {
  text: string;
  start: number;
  end: number;
  score: number;
  normalized_importance: number | null;
}
export interface Explanation {
  text: string;
  label: Label;
  spans: Span[];
  convergence_delta: number | null;
}
export interface Job {
  job_id: string;
  classification_id: string;
  label: Label;
  status: "queued" | "running" | "done" | "failed";
  error: string | null;
  result?: Explanation | null;
}
export interface Correction {
  id: string;
  classification_id: string;
  corrected_labels: Record<Label, boolean>;
  moderator_id: string;
  flagged_at: string;
  created_at: string;
}
export const PRESETS = [
  {
    name: "Helpful",
    kind: "clean",
    text: "Thanks for sharing this. Your explanation really helped me understand the topic.",
  },
  {
    name: "Disagreement",
    kind: "borderline",
    text: "I completely disagree with this argument. It ignores the evidence and makes no sense.",
  },
  {
    name: "Insult",
    kind: "insult",
    text: "You are an idiot. Nobody wants to hear your stupid opinions.",
  },
  {
    name: "Threat",
    kind: "threat",
    text: "I am going to find you and hurt you. You will regret saying that.",
  },
];
export const charCount = (text: string) => Array.from(text).length;
export function defaultLabel(result: Classification): Label {
  return [...LABELS].sort(
    (a, b) =>
      Number(result.labels[b].predicted) - Number(result.labels[a].predicted) ||
      result.labels[b].probability - result.labels[a].probability,
  )[0];
}
export function attributionParts(explanation: Explanation) {
  const chars = Array.from(explanation.text);
  let cursor = 0;
  const parts: { text: string; span?: Span }[] = [];
  for (const span of [...explanation.spans].sort((a, b) => a.start - b.start)) {
    if (
      span.start < cursor ||
      span.end <= span.start ||
      span.end > chars.length
    )
      continue;
    if (span.start > cursor)
      parts.push({ text: chars.slice(cursor, span.start).join("") });
    parts.push({ text: chars.slice(span.start, span.end).join(""), span });
    cursor = span.end;
  }
  if (cursor < chars.length) parts.push({ text: chars.slice(cursor).join("") });
  return parts;
}
