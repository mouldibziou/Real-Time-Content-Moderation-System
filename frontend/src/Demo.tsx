import { useCallback, useEffect, useRef, useState } from "react";
import { api, delay } from "./api";
import { Arrow, Flag, Footer, Header, Status } from "./components";
import {
  LABELS,
  LABEL_NAMES,
  PRESETS,
  attributionParts,
  charCount,
  defaultLabel,
} from "./types";
import type {
  Classification,
  Correction,
  Explanation,
  Job,
  Label,
  ModelInfo,
} from "./types";
const message = (error: unknown) =>
  error instanceof Error
    ? error.message
    : "Something went wrong. Please try again.";
const aborted = (error: unknown) =>
  error instanceof DOMException && error.name === "AbortError";
export function Demo() {
  const [info, setInfo] = useState<ModelInfo | null>(null),
    [infoError, setInfoError] = useState(""),
    [loadingInfo, setLoadingInfo] = useState(true);
  const [text, setText] = useState(PRESETS[0].text),
    [result, setResult] = useState<Classification | null>(null),
    [elapsed, setElapsed] = useState<number | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const [batchText, setBatchText] = useState(
      PRESETS.map((p) => p.text).join("\n"),
    ),
    [batchResults, setBatchResults] = useState<Classification[]>([]),
    [batchBusy, setBatchBusy] = useState(false),
    [batchError, setBatchError] = useState("");
  const [label, setLabel] = useState<Label>("toxic"),
    [explanation, setExplanation] = useState<Explanation | null>(null),
    [job, setJob] = useState<Job | null>(null),
    [explaining, setExplaining] = useState(false),
    [explainError, setExplainError] = useState(""),
    [cached, setCached] = useState(false),
    [paused, setPaused] = useState(false);
  const [corrections, setCorrections] = useState<Record<Label, boolean>>(
      Object.fromEntries(LABELS.map((l) => [l, false])) as Record<
        Label,
        boolean
      >,
    ),
    [correction, setCorrection] = useState<Correction | null>(null),
    [correcting, setCorrecting] = useState(false),
    [correctionError, setCorrectionError] = useState("");
  const classifyController = useRef<AbortController | null>(null),
    explainController = useRef<AbortController | null>(null),
    correctionController = useRef<AbortController | null>(null),
    batchController = useRef<AbortController | null>(null),
    infoController = useRef<AbortController | null>(null);
  const loadInfo = useCallback(async () => {
    infoController.current?.abort();
    const controller = new AbortController();
    infoController.current = controller;
    setLoadingInfo(true);
    setInfoError("");
    try {
      const next = await api.info(controller.signal);
      if (controller.signal.aborted) return;
      if (
        next.labels.length !== 6 ||
        !LABELS.every((l) => next.labels.includes(l)) ||
        next.thresholds.length !== 6
      )
        throw new Error("The API returned an unsupported label configuration.");
      setInfo(next);
    } catch (e) {
      if (!aborted(e)) setInfoError(message(e));
    } finally {
      if (!controller.signal.aborted) setLoadingInfo(false);
    }
  }, []);
  useEffect(() => {
    void loadInfo();
    return () => {
      [
        classifyController,
        explainController,
        correctionController,
        batchController,
        infoController,
      ].forEach((ref) => ref.current?.abort());
    };
  }, [loadInfo]);
  function resetInspection() {
    explainController.current?.abort();
    correctionController.current?.abort();
    setExplanation(null);
    setJob(null);
    setExplainError("");
    setExplaining(false);
    setPaused(false);
    setCached(false);
    setCorrection(null);
    setCorrectionError("");
    setCorrecting(false);
  }
  function inspect(next: Classification, timing: number | null) {
    setError("");
    classifyController.current?.abort();
    setBusy(false);
    resetInspection();
    setResult(next);
    setLabel(defaultLabel(next));
    setCorrections(
      Object.fromEntries(
        LABELS.map((l) => [l, next.labels[l].predicted]),
      ) as Record<Label, boolean>,
    );
    setElapsed(timing);
  }
  function editText(next: string) {
    classifyController.current?.abort();
    setBusy(false);
    setText(next);
    setError("");
    setResult(null);
    setElapsed(null);
    resetInspection();
  }
  function validate(value: string) {
    if (!value.trim()) return "Enter some text before classifying.";
    if (charCount(value) > (info?.max_text_chars ?? 10000))
      return `Each comment must be at most ${info?.max_text_chars ?? 10000} characters.`;
    return "";
  }
  async function classify(event: React.FormEvent) {
    event.preventDefault();
    const invalid = validate(text);
    if (invalid) {
      setError(invalid);
      return;
    }
    classifyController.current?.abort();
    resetInspection();
    setResult(null);
    const controller = new AbortController();
    classifyController.current = controller;
    setBusy(true);
    setError("");
    const start = performance.now();
    try {
      const next = await api.classify(text, controller.signal);
      if (!controller.signal.aborted) {
        const timing = performance.now() - start;
        inspect(next, timing);
      }
    } catch (e) {
      if (!aborted(e)) setError(message(e));
    } finally {
      if (classifyController.current === controller) setBusy(false);
    }
  }
  async function classifyBatch(event: React.FormEvent) {
    event.preventDefault();
    const texts = batchText.split(/\r?\n/).filter((t) => t.trim());
    if (!texts.length) {
      setBatchError("Add at least one comment, one per line.");
      return;
    }
    if (texts.length > (info?.max_prediction_batch_size ?? 64)) {
      setBatchError(
        `Use at most ${info?.max_prediction_batch_size ?? 64} comments per batch.`,
      );
      return;
    }
    const invalid = texts.map(validate).find(Boolean);
    if (invalid) {
      setBatchError(invalid);
      return;
    }
    batchController.current?.abort();
    const controller = new AbortController();
    batchController.current = controller;
    setBatchBusy(true);
    setBatchError("");
    try {
      const response = await api.batch(texts, controller.signal);
      if (!controller.signal.aborted) setBatchResults(response.results);
    } catch (e) {
      if (!aborted(e)) setBatchError(message(e));
    } finally {
      if (!controller.signal.aborted) setBatchBusy(false);
    }
  }
  async function watchJob(initial: Job, controller: AbortController) {
    const start = Date.now();
    let current = initial;
    while (!controller.signal.aborted) {
      setJob(current);
      if (current.status === "failed") {
        setExplainError(
          "The explanation job failed. The server cannot restart this job from the demo. You can still inspect the classification or try different text.",
        );
        return;
      }
      if (current.status === "done") {
        if (current.result) setExplanation(current.result);
        else
          setExplainError(
            "The job completed without a readable explanation. Refresh its status to check again.",
          );
        return;
      }
      if (Date.now() - start >= 120000) {
        setPaused(true);
        return;
      }
      await delay(1500, controller.signal);
      current = await api.job(current.job_id, controller.signal);
    }
  }
  async function explain(resume = false) {
    if (!result) return;
    explainController.current?.abort();
    const controller = new AbortController();
    explainController.current = controller;
    setExplaining(true);
    setExplainError("");
    setPaused(false);
    try {
      if (resume && job) {
        await watchJob(
          await api.job(job.job_id, controller.signal),
          controller,
        );
      } else {
        setExplanation(null);
        setCached(false);
        const response = await api.explain(
          result.classification_id,
          label,
          controller.signal,
        );
        if (controller.signal.aborted) return;
        if ("spans" in response) {
          setExplanation(response);
          setCached(true);
          setJob(null);
        } else await watchJob(response, controller);
      }
    } catch (e) {
      if (!aborted(e)) setExplainError(message(e));
    } finally {
      if (!controller.signal.aborted) setExplaining(false);
    }
  }
  async function submitCorrection(event: React.FormEvent) {
    event.preventDefault();
    if (!result) return;
    correctionController.current?.abort();
    const controller = new AbortController();
    correctionController.current = controller;
    setCorrecting(true);
    setCorrectionError("");
    try {
      const saved = await api.correct(
        result.classification_id,
        corrections,
        controller.signal,
      );
      if (!controller.signal.aborted) setCorrection(saved);
    } catch (e) {
      if (!aborted(e)) setCorrectionError(message(e));
    } finally {
      if (!controller.signal.aborted) setCorrecting(false);
    }
  }
  const flagged = result && LABELS.some((l) => result.labels[l].predicted);
  const count = charCount(text);
  return (
    <>
      <Header demo />
      <main tabIndex={-1} id="main-content" className="demo-main wrap">
        <div className="demo-intro">
          <div>
            <a href="/" className="back-link mono">
              ← THE PROJECT
            </a>
            <h1>
              A closer look
              <br />
              at your content<span>.</span>
            </h1>
            <p>Classify the text. Explore the signals. Make the final call.</p>
          </div>
          <div className="demo-live">
            <Status />
            <span className="mono">LIVE API · NO SIGN-UP REQUIRED</span>
          </div>
        </div>
        {infoError && (
          <div className="service-notice" role="alert">
            <div>
              <strong>Model information is unavailable.</strong>
              <p>
                {infoError} Classification becomes available when the model
                configuration loads.
              </p>
            </div>
            <button
              className="button outline small"
              onClick={() => void loadInfo()}
              disabled={loadingInfo}
            >
              {loadingInfo ? "Connecting…" : "Reconnect"}
              <Arrow />
            </button>
          </div>
        )}
        <div className="workspace">
          <section className="input-panel" aria-labelledby="input-heading">
            <div className="panel-heading">
              <span className="mono">01 / SUBMIT</span>
              <span className="mono">TEXT INPUT</span>
            </div>
            <div className="input-body">
              <h2 id="input-heading">What’s being said?</h2>
              <p>Start with an example, or bring your own text.</p>
              <div
                className="preset-buttons"
                role="group"
                aria-label="Example comments"
              >
                {PRESETS.map((p) => (
                  <button
                    key={p.name}
                    className={text === p.text ? "selected" : ""}
                    aria-pressed={text === p.text}
                    onClick={() => editText(p.text)}
                  >
                    <span className={`preset-dot ${p.kind}`} />
                    {p.name}
                  </button>
                ))}
              </div>
              <form onSubmit={classify}>
                <label className="sr-only" htmlFor="comment">
                  Comment to classify
                </label>
                <textarea
                  id="comment"
                  placeholder="Write a comment to classify…"
                  value={text}
                  onChange={(e) => editText(e.target.value)}
                  aria-describedby="text-limits storage-notice"
                />
                <div id="text-limits" className="input-limit mono">
                  <span>{info?.max_length ?? 512}-TOKEN MODEL LIMIT</span>
                  <span
                    className={
                      count > (info?.max_text_chars ?? 10000)
                        ? "over-limit"
                        : ""
                    }
                  >
                    {count.toLocaleString()} /{" "}
                    {(info?.max_text_chars ?? 10000).toLocaleString()}
                  </span>
                </div>
                {error && (
                  <p className="error-message" role="alert">
                    {error}
                  </p>
                )}
                <button
                  className="button dark classify-button"
                  disabled={!info || busy}
                  type="submit"
                >
                  {busy ? (
                    <>
                      <span className="spinner" />
                      Classifying…
                    </>
                  ) : (
                    <>
                      Classify text
                      <Arrow />
                    </>
                  )}
                </button>
              </form>
              <p id="storage-notice" className="storage-note">
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 20 20"
                  fill="none"
                  aria-hidden="true"
                >
                  <circle cx="10" cy="10" r="7" stroke="currentColor" />
                  <path d="M10 9v5m0-9v1" stroke="currentColor" />
                </svg>
                Submitted text and corrections are stored in the project
                database. Avoid personal or confidential information. Text
                beyond {info?.max_length ?? 512} tokens is truncated for
                inference.
              </p>
            </div>
            <div className="input-footer mono">
              <span className="green-square" />
              SIX LABELS · INDEPENDENT SCORES
            </div>
          </section>
          <section
            className="results-panel"
            aria-labelledby="result-heading"
            aria-busy={busy}
          >
            <div className="panel-heading">
              <span className="mono">02 / UNDERSTAND</span>
              <span className="mono">CLASSIFICATION</span>
            </div>
            {!result ? (
              <div className="empty-result" aria-live="polite">
                <div className={`empty-orbit ${busy ? "is-busy" : ""}`}>
                  <span />
                  <Flag />
                  <span />
                </div>
                <h2 id="result-heading">
                  {busy ? "Finding the signals." : "Every comment has context."}
                </h2>
                <p>
                  {busy
                    ? "Scoring six categories and saving the result."
                    : "Run a classification to see six scores, the thresholds, and what deserves a closer look."}
                </p>
                <div className="empty-labels">
                  {LABELS.map((l) => (
                    <span key={l}>{LABEL_NAMES[l]}</span>
                  ))}
                </div>
                <span className="mono empty-caption">
                  YOUR TEXT → SIX SIGNALS → A CLEARER DECISION
                </span>
              </div>
            ) : (
              <div className="results-body" aria-live="polite">
                <div className="result-verdict">
                  <span
                    className={`verdict-icon ${flagged ? "flagged" : "clear"}`}
                  >
                    {flagged ? <Flag /> : "✓"}
                  </span>
                  <div>
                    <span className="mono eyebrow">MODEL PREDICTION</span>
                    <h2 id="result-heading">
                      {flagged ? "Flagged for review" : "No labels triggered"}
                    </h2>
                  </div>
                </div>
                <p className="score-intro">
                  Each score is independent. Multiple labels can apply.
                </p>
                <div className="score-list">
                  {LABELS.map((l) => {
                    const prediction = result.labels[l],
                      threshold = info?.thresholds[info.labels.indexOf(l)];
                    return (
                      <div
                        className={`score-row ${prediction.predicted ? "positive" : ""}`}
                        key={l}
                      >
                        <div className="score-label">
                          <span>
                            {LABEL_NAMES[l]}{" "}
                            {prediction.predicted && (
                              <span className="flag-badge">FLAGGED</span>
                            )}
                          </span>
                          <strong>
                            {(prediction.probability * 100).toFixed(1)}
                            <small>%</small>
                          </strong>
                        </div>
                        <div
                          className="score-track"
                          role="meter"
                          aria-label={`${LABEL_NAMES[l]} probability`}
                          aria-valuemin={0}
                          aria-valuemax={100}
                          aria-valuenow={Number(
                            (prediction.probability * 100).toFixed(1),
                          )}
                        >
                          <span
                            className="score-fill"
                            style={{
                              width: `${prediction.probability * 100}%`,
                            }}
                          />
                          {threshold !== undefined && (
                            <i
                              style={{ left: `${threshold * 100}%` }}
                              title={`${LABEL_NAMES[l]} threshold: ${Math.round(threshold * 100)}%`}
                            />
                          )}
                        </div>
                        <span className="threshold-caption mono">
                          THRESHOLD{" "}
                          {threshold === undefined
                            ? "UNAVAILABLE"
                            : `${Math.round(threshold * 100)}%`}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <div className="result-metadata mono">
                  <span>
                    {info?.model_type} {info?.model_version}
                  </span>
                  <span>
                    {info?.device.toUpperCase()} /{" "}
                    {info?.inference_backend.toUpperCase()}
                  </span>
                  {elapsed !== null && (
                    <span>{Math.round(elapsed)} ms REQUEST ROUND TRIP</span>
                  )}
                </div>
              </div>
            )}
          </section>
        </div>
        <section
          className={`explanation-section ${!result ? "inactive" : ""}`}
          aria-labelledby="explain-heading"
        >
          <div className="explanation-heading">
            <div>
              <span className="mono eyebrow">03 / LOOK BENEATH THE SCORE</span>
              <h2 id="explain-heading">Why this flag?</h2>
              <p>See which words support or oppose a selected label.</p>
            </div>
            <div className="explain-actions">
              <label htmlFor="explain-label" className="sr-only">
                Label to explain
              </label>
              <select
                id="explain-label"
                value={label}
                disabled={!result || explaining}
                onChange={(e) => {
                  explainController.current?.abort();
                  setLabel(e.target.value as Label);
                  setExplanation(null);
                  setJob(null);
                  setExplainError("");
                  setPaused(false);
                  setCached(false);
                }}
              >
                {LABELS.map((l) => (
                  <option key={l} value={l}>
                    {LABEL_NAMES[l]}
                  </option>
                ))}
              </select>
              <button
                className="button outline"
                disabled={!result || explaining}
                onClick={() => void explain(false)}
              >
                {explaining ? (
                  <>
                    <span className="spinner" />
                    Explaining…
                  </>
                ) : (
                  <>
                    Explain prediction
                    <Arrow />
                  </>
                )}
              </button>
            </div>
          </div>
          {!result ? (
            <div className="explain-placeholder mono">
              CLASSIFY TEXT TO UNLOCK ITS EXPLANATION <span>↗</span>
            </div>
          ) : (
            <>
              <blockquote className="classified-text">{result.text}</blockquote>
              {job && (
                <div className="job-lifecycle" aria-live="polite">
                  <span className="mono">EXPLANATION JOB</span>
                  {["queued", "running", "done"].map((s) => (
                    <span key={s} className={job.status === s ? "current" : ""}>
                      <i />
                      {s}
                    </span>
                  ))}
                  <strong className="mono">
                    {job.status === "failed"
                      ? "FAILED"
                      : paused
                        ? "WATCH PAUSED"
                        : job.status.toUpperCase()}
                  </strong>
                </div>
              )}
              {explainError && (
                <p className="error-message" role="alert">
                  {explainError}
                </p>
              )}
              {(paused || (explainError && job && job.status !== "failed")) && (
                <button
                  className="button outline small"
                  disabled={explaining}
                  onClick={() => void explain(true)}
                >
                  Refresh job status
                  <Arrow />
                </button>
              )}
              {explanation ? (
                <div className="attribution-view">
                  <div className="attribution-header">
                    <span className="mono">
                      WORD CONTRIBUTIONS /{" "}
                      {LABEL_NAMES[explanation.label].toUpperCase()}
                    </span>
                    {cached && (
                      <span className="cached-badge">↺ Cached explanation</span>
                    )}
                  </div>
                  <div
                    className="token-heatmap"
                    aria-label="Text with contribution highlights"
                  >
                    {attributionParts(explanation).map((part, i) =>
                      part.span ? (
                        <mark
                          key={i}
                          className={
                            part.span.score >= 0 ? "supports" : "opposes"
                          }
                          style={{
                            backgroundColor:
                              part.span.score === 0
                                ? "transparent"
                                : `rgba(${part.span.score >= 0 ? "185, 104, 76" : "76, 119, 155"}, ${0.09 + (part.span.normalized_importance ?? 0) * 0.55})`,
                          }}
                          title={`${part.span.score >= 0 ? "Supports" : "Opposes"} ${LABEL_NAMES[label]}; relative importance ${Math.round((part.span.normalized_importance ?? 0) * 100)}%`}
                        >
                          {part.text}
                        </mark>
                      ) : (
                        <span key={i}>{part.text}</span>
                      ),
                    )}
                  </div>
                  <div className="attribution-legend mono">
                    <span>
                      <i className="support-key" />
                      SUPPORTS LABEL
                    </span>
                    <span>
                      <i className="oppose-key" />
                      OPPOSES LABEL
                    </span>
                    <span>DARKER = STRONGER CONTRIBUTION</span>
                  </div>
                  <div className="contributors">
                    <span className="mono">STRONGEST CONTRIBUTORS</span>
                    {[...explanation.spans]
                      .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
                      .slice(0, 5)
                      .map((span, i) => (
                        <span
                          key={i}
                          className={span.score >= 0 ? "supports" : "opposes"}
                        >
                          {span.text}
                          <small>
                            {span.score >= 0 ? "+" : "−"}
                            {Math.round(
                              (span.normalized_importance ?? 0) * 100,
                            )}
                            %
                          </small>
                        </span>
                      ))}
                  </div>
                  <p className="attribution-note">
                    Contributions explain this model’s score for one label.
                    Percentages show relative attribution strength, not toxicity
                    probability.
                  </p>
                </div>
              ) : !explaining && !explainError && !paused ? (
                <div className="explain-placeholder">
                  Explanations run in a separate worker. Request one when you
                  want a closer look.<span>↗</span>
                </div>
              ) : null}
            </>
          )}
        </section>
        {result && (
          <section className="correction-section">
            <div>
              <span className="mono eyebrow">
                04 / KEEP THE HUMAN IN THE LOOP
              </span>
              <h2>Your judgment matters.</h2>
              <p>
                Agree with the model, or adjust the labels. Your feedback is
                saved as a public-demo audit event for possible future
                retraining. It does not retrain the model automatically.
              </p>
              <span className="mono moderator-label">
                MODERATOR / PUBLIC-DEMO
              </span>
            </div>
            <form onSubmit={submitCorrection}>
              <div className="correction-toggles">
                {LABELS.map((l) => (
                  <label key={l}>
                    <span>{LABEL_NAMES[l]}</span>
                    <input
                      type="checkbox"
                      role="switch"
                      checked={corrections[l]}
                      onChange={(e) => {
                        setCorrections({
                          ...corrections,
                          [l]: e.target.checked,
                        });
                        setCorrection(null);
                      }}
                      disabled={correcting}
                    />
                    <span className="toggle-visual" />
                  </label>
                ))}
              </div>
              {correctionError && (
                <p className="error-message" role="alert">
                  {correctionError}
                </p>
              )}
              <button
                className="button dark"
                type="submit"
                disabled={correcting || Boolean(correction)}
              >
                {correcting
                  ? "Saving correction…"
                  : correction
                    ? "Correction saved ✓"
                    : "Save my review"}
                {!correction && <Arrow />}
              </button>
              {correction && (
                <div className="correction-receipt" role="status">
                  <strong>Review recorded in the audit trail.</strong>
                  <span className="mono">EVENT {correction.id}</span>
                  <span>
                    {new Date(correction.created_at).toLocaleString()}
                  </span>
                  <p>
                    {LABELS.filter((l) => correction.corrected_labels[l])
                      .map((l) => LABEL_NAMES[l])
                      .join(", ") || "No labels selected"}
                  </p>
                </div>
              )}
            </form>
          </section>
        )}
        <details className="batch-section">
          <summary>
            <div>
              <span className="mono eyebrow">
                ONE REQUEST. MULTIPLE COMMENTS.
              </span>
              <h2>Try a batch.</h2>
              <p>
                Same six signals. Up to {info?.max_prediction_batch_size ?? 64}{" "}
                comments at once.
              </p>
            </div>
            <span className="expand-icon">+</span>
          </summary>
          <div className="batch-content">
            <form onSubmit={classifyBatch}>
              <label htmlFor="batch-text" className="mono">
                COMMENTS / ONE PER LINE
              </label>
              <textarea
                id="batch-text"
                value={batchText}
                onChange={(e) => {
                  batchController.current?.abort();
                  setBatchBusy(false);
                  setBatchText(e.target.value);
                  setBatchResults([]);
                  setBatchError("");
                }}
                rows={7}
              />
              <p className="storage-note">
                Batch comments are also stored. Each comment has the same
                character and inference token limits.
              </p>
              {batchError && (
                <p className="error-message" role="alert">
                  {batchError}
                </p>
              )}
              <button
                type="submit"
                className="button outline"
                disabled={!info || batchBusy}
              >
                {batchBusy ? "Classifying batch…" : "Classify batch"}
                <Arrow />
              </button>
            </form>
            {batchResults.length > 0 && (
              <div className="batch-results" aria-live="polite">
                <span className="mono">RESULTS / INPUT ORDER PRESERVED</span>
                {batchResults.map((r, i) => (
                  <div key={`${r.classification_id}-${i}`}>
                    <span className="mono">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <div>
                      <p>{r.text}</p>
                      <div className="batch-badges">
                        {LABELS.filter((l) => r.labels[l].predicted).map(
                          (l) => (
                            <span key={l}>{LABEL_NAMES[l]}</span>
                          ),
                        )}
                        {!LABELS.some((l) => r.labels[l].predicted) && (
                          <span className="clean-badge">
                            No labels triggered
                          </span>
                        )}
                      </div>
                    </div>
                    <button
                      className="text-link"
                      onClick={() => {
                        setText(r.text);
                        inspect(r, null);
                        document
                          .getElementById("result-heading")
                          ?.scrollIntoView({
                            behavior: matchMedia(
                              "(prefers-reduced-motion: reduce)",
                            ).matches
                              ? "instant"
                              : "smooth",
                            block: "center",
                          });
                      }}
                    >
                      Inspect
                      <Arrow />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </details>
        {result && (
          <details className="raw-section">
            <summary className="mono">
              ADVANCED / RESPONSE JSON<span>+</span>
            </summary>
            <pre>
              {JSON.stringify(
                {
                  classification: result,
                  ...(explanation ? { explanation } : {}),
                  ...(job ? { job } : {}),
                  ...(correction ? { correction } : {}),
                },
                null,
                2,
              )}
            </pre>
          </details>
        )}
        <div className="demo-bottom">
          <span className="mono">THE FULL PICTURE, BEYOND THE PREDICTION.</span>
          <a href="/#architecture" className="text-link">
            Explore the architecture
            <Arrow diagonal />
          </a>
        </div>
      </main>
      <Footer />
    </>
  );
}
