import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";
import { Demo } from "./Demo";
import { LABELS, PRESETS, attributionParts, charCount } from "./types";
import type { Classification, Explanation, Job, ModelInfo } from "./types";
const info: ModelInfo = {
  model_type: "distilbert",
  model_version: "v2.0.0",
  labels: [...LABELS],
  thresholds: [0.74, 0.45, 0.74, 0.59, 0.61, 0.76],
  device: "cpu",
  max_length: 512,
  explanation_steps: 32,
  inference_backend: "onnx",
  max_text_chars: 10000,
  max_prediction_batch_size: 64,
};
const result = (
  text = PRESETS[2].text,
  id = "classification-1",
): Classification => ({
  classification_id: id,
  text,
  labels: Object.fromEntries(
    LABELS.map((l) => [
      l,
      { probability: l === "insult" ? 0.93 : 0.12, predicted: l === "insult" },
    ]),
  ) as Classification["labels"],
});
const explanation: Explanation = {
  text: "🙂 stupid idea",
  label: "insult",
  spans: [
    { text: "stupid", start: 2, end: 8, score: 1, normalized_importance: 1 },
    {
      text: "idea",
      start: 9,
      end: 13,
      score: -0.2,
      normalized_importance: 0.2,
    },
  ],
  convergence_delta: 0.001,
};
const job: Job = {
  job_id: "job-1",
  classification_id: "classification-1",
  label: "insult",
  status: "queued",
  error: null,
};
beforeEach(() => {
  vi.spyOn(api, "info").mockResolvedValue(info);
  vi.spyOn(api, "ready").mockResolvedValue({ is_ready: true });
  vi.spyOn(api, "classify").mockResolvedValue(result());
  vi.spyOn(api, "explain").mockResolvedValue(explanation);
  vi.spyOn(api, "correct").mockImplementation(async (id, labels) => ({
    id: "event-1",
    classification_id: id,
    corrected_labels: labels,
    moderator_id: "public-demo",
    flagged_at: "2026-10-19T00:00:00Z",
    created_at: "2026-10-19T00:00:00Z",
  }));
});
async function classify() {
  await screen.findByRole("button", { name: "Classify text" });
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Classify text" })).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: "Classify text" }));
  await screen.findByText("Flagged for review");
}
describe("moderation workspace", () => {
  it("uses API decisions and metadata thresholds rather than rounding probabilities", async () => {
    const clean = result();
    LABELS.forEach(
      (l) => (clean.labels[l] = { probability: 0.999, predicted: false }),
    );
    vi.mocked(api.classify).mockResolvedValue(clean);
    render(<Demo />);
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Classify text" }),
      ).toBeEnabled(),
    );
    fireEvent.click(screen.getByRole("button", { name: "Classify text" }));
    await screen.findByText("No labels triggered");
    expect(screen.getAllByText("THRESHOLD 74%")).toHaveLength(2);
    expect(screen.getAllByRole("meter")).toHaveLength(6);
    expect(screen.queryByText("FLAGGED")).not.toBeInTheDocument();
  });
  it("shows cached Unicode attribution and saves a complete human review", async () => {
    vi.mocked(api.classify).mockResolvedValue(result(explanation.text));
    render(<Demo />);
    await classify();
    fireEvent.click(screen.getByRole("button", { name: "Explain prediction" }));
    await screen.findByText("↺ Cached explanation");
    expect(
      screen.getByLabelText("Text with contribution highlights").textContent,
    ).toBe(explanation.text);
    fireEvent.click(screen.getByRole("switch", { name: "Threat" }));
    fireEvent.click(screen.getByRole("button", { name: "Save my review" }));
    await screen.findByText("Review recorded in the audit trail.");
    expect(api.correct).toHaveBeenCalledWith(
      "classification-1",
      expect.objectContaining({ threat: true, insult: true }),
      expect.any(AbortSignal),
    );
    expect(Object.keys(vi.mocked(api.correct).mock.calls[0][1])).toHaveLength(
      6,
    );
    expect(screen.getByText("EVENT event-1")).toBeInTheDocument();
  });
  it("polls real queued and running states until completion", async () => {
    vi.mocked(api.explain).mockResolvedValue(job);
    vi.spyOn(api, "job")
      .mockResolvedValueOnce({ ...job, status: "running" })
      .mockResolvedValueOnce({ ...job, status: "done", result: explanation });
    render(<Demo />);
    await classify();
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole("button", { name: "Explain prediction" }));
    await act(async () => {});
    expect(screen.getByText("QUEUED")).toBeInTheDocument();
    await act(() => vi.advanceTimersByTimeAsync(1500));
    expect(screen.getByText("RUNNING")).toBeInTheDocument();
    await act(() => vi.advanceTimersByTimeAsync(1500));
    expect(
      screen.getByLabelText("Text with contribution highlights"),
    ).toBeInTheDocument();
    expect(api.job).toHaveBeenCalledTimes(2);
  });
  it("stops watching after two minutes and can refresh the same job", async () => {
    vi.mocked(api.explain).mockResolvedValue(job);
    vi.spyOn(api, "job").mockResolvedValue(job);
    render(<Demo />);
    await classify();
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole("button", { name: "Explain prediction" }));
    await act(async () => {});
    await act(() => vi.advanceTimersByTimeAsync(120000));
    expect(screen.getByText("WATCH PAUSED")).toBeInTheDocument();
    vi.mocked(api.job).mockResolvedValue({
      ...job,
      status: "done",
      result: explanation,
    });
    fireEvent.click(screen.getByRole("button", { name: "Refresh job status" }));
    await act(async () => {});
    expect(
      screen.getByLabelText("Text with contribution highlights"),
    ).toBeInTheDocument();
    expect(api.explain).toHaveBeenCalledTimes(1);
  });
  it("describes a failed job without promising a restart", async () => {
    vi.mocked(api.explain).mockResolvedValue({ ...job, status: "failed" });
    render(<Demo />);
    await classify();
    fireEvent.click(screen.getByRole("button", { name: "Explain prediction" }));
    await screen.findByText(/server cannot restart this job/);
    expect(
      screen.queryByRole("button", { name: "Refresh job status" }),
    ).not.toBeInTheDocument();
  });
  it("ignores a stale classification after text changes", async () => {
    let resolve!: (value: Classification) => void;
    vi.mocked(api.classify).mockImplementation(
      () => new Promise((r) => (resolve = r)),
    );
    render(<Demo />);
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Classify text" }),
      ).toBeEnabled(),
    );
    fireEvent.click(screen.getByRole("button", { name: "Classify text" }));
    fireEvent.change(screen.getByLabelText("Comment to classify"), {
      target: { value: "Changed comment" },
    });
    await act(async () => resolve(result()));
    expect(screen.queryByText("Flagged for review")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Comment to classify")).toHaveValue(
      "Changed comment",
    );
  });
  it("cancels an explanation when a different comment is selected", async () => {
    vi.mocked(api.explain).mockResolvedValue(job);
    vi.spyOn(api, "job").mockResolvedValue(job);
    render(<Demo />);
    await classify();
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole("button", { name: "Explain prediction" }));
    await act(async () => {});
    fireEvent.click(screen.getByRole("button", { name: "Helpful" }));
    await act(() => vi.advanceTimersByTimeAsync(5000));
    expect(api.job).not.toHaveBeenCalled();
    expect(screen.queryByText("QUEUED")).not.toBeInTheDocument();
  });
  it("preserves batch order and opens an existing result for inspection", async () => {
    vi.spyOn(api, "batch").mockResolvedValue({
      results: [
        result("First comment", "one"),
        result("Second comment", "two"),
      ],
    });
    render(<Demo />);
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Classify text" }),
      ).toBeEnabled(),
    );
    fireEvent.click(screen.getByText("Try a batch."));
    fireEvent.click(screen.getByRole("button", { name: "Classify batch" }));
    await screen.findByText("First comment");
    const rows = screen.getAllByRole("button", { name: "Inspect" });
    expect(rows).toHaveLength(2);
    Element.prototype.scrollIntoView = vi.fn();
    fireEvent.click(rows[1]);
    await screen.findByText("Flagged for review");
    expect(screen.getByLabelText("Comment to classify")).toHaveValue(
      "Second comment",
    );
    expect(api.classify).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Explain prediction" }));
    await waitFor(() =>
      expect(api.explain).toHaveBeenCalledWith(
        "two",
        "insult",
        expect.any(AbortSignal),
      ),
    );
  });
  it("retains text on server failure and rejects empty input", async () => {
    vi.mocked(api.classify).mockRejectedValue(
      new Error("The API is at capacity."),
    );
    render(<Demo />);
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Classify text" }),
      ).toBeEnabled(),
    );
    fireEvent.click(screen.getByRole("button", { name: "Classify text" }));
    await screen.findByText("The API is at capacity.");
    expect(screen.getByLabelText("Comment to classify")).toHaveValue(
      PRESETS[0].text,
    );
    fireEvent.change(screen.getByLabelText("Comment to classify"), {
      target: { value: "  " },
    });
    fireEvent.click(screen.getByRole("button", { name: "Classify text" }));
    await screen.findByText("Enter some text before classifying.");
    expect(api.classify).toHaveBeenCalledTimes(1);
  });
  it("retains correction edits when save fails and permits retry", async () => {
    vi.mocked(api.correct).mockRejectedValueOnce(
      new Error("Storage unavailable"),
    );
    render(<Demo />);
    await classify();
    fireEvent.click(screen.getByRole("switch", { name: "Threat" }));
    fireEvent.click(screen.getByRole("button", { name: "Save my review" }));
    await screen.findByText("Storage unavailable");
    expect(screen.getByRole("switch", { name: "Threat" })).toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: "Save my review" }));
    await screen.findByText("Review recorded in the audit trail.");
  });
  it("keeps classification disabled until model metadata is available", async () => {
    vi.mocked(api.info).mockRejectedValueOnce(new Error("Metadata blocked"));
    render(<Demo />);
    await screen.findByText("Model information is unavailable.");
    expect(
      screen.getByRole("button", { name: "Classify text" }),
    ).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Reconnect" }));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: "Classify text" }),
      ).toBeEnabled(),
    );
  });
});
it("preserves Python code-point offsets and whitespace around emoji", () => {
  expect(
    attributionParts(explanation)
      .map((p) => p.text)
      .join(""),
  ).toBe(explanation.text);
  expect(attributionParts(explanation).find((p) => p.span)?.text).toBe(
    "stupid",
  );
  expect(charCount("🙂x")).toBe(2);
});
