import { useState } from "react";
import {
  Arrow,
  Footer,
  Header,
  SectionLabel,
  Status,
  useReveal,
} from "./components";
import { Architecture } from "./Architecture";
const repo =
  "https://github.com/mouldibziou/Real-Time-Content-Moderation-System";
function UseCaseHighlights() {
  return (
    <div className="platform-highlights wrap" aria-label="Moderation use case examples">
      <span className="mono">MODERATION USE CASES</span>
      <ul>
        <li>
          <svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
            <path d="M14 3h3a5 5 0 0 0 4 4v3a8 8 0 0 1-4-1.2V16a6 6 0 1 1-6-6h1v3h-1a3 3 0 1 0 3 3Z" />
          </svg>
          TikTok comments
        </li>
        <li>
          <svg viewBox="0 0 24 24" aria-hidden="true" fill="currentColor">
            <path d="M12 3C6.5 3 2 6.9 2 11.8c0 2.7 1.4 5.2 3.7 6.8V22l3.2-1.8c1 .3 2 .4 3.1.4 5.5 0 10-3.9 10-8.8S17.5 3 12 3Z" />
            <path d="m6.4 14.8 4.7-5 2.8 2.1 3.7-2.7-4.7 5-2.8-2.1Z" fill="var(--paper)" />
          </svg>
          Messenger chats
        </li>
        <li>
          <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20.5 11.7a8.5 8.5 0 0 1-12.7 7.4L3 20.5l1.4-4.7a8.5 8.5 0 1 1 16.1-4.1Z" />
            <path d="m8.4 7.4 1.4 2.8-1.1 1.1a8.1 8.1 0 0 0 4 4l1.1-1.1 2.8 1.4c-.2 1.4-1.2 2-2.5 1.7-4.1-1-7-3.9-8-8-.3-1.3.3-2.3 1.7-2.5Z" transform="translate(1 -1) scale(.95)" />
          </svg>
          WhatsApp groups
        </li>
        <li>
          <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="4" width="18" height="16" rx="2" />
            <path d="M3 9h18M10 9v11M6 6.5h.01M8.5 6.5h.01M13 12h5M13 15h5M13 18h3" />
          </svg>
          Content platforms
        </li>
      </ul>
    </div>
  );
}
function HeroSketch() {
  return (
    <div
      className="hero-sketch"
      aria-label="Illustrative sketch: submitted text passes through a classifier and produces six independent scores"
    >
      <div className="sketch-top mono">
        <span>TEXT IN. CLARITY OUT.</span>
        <span>FIG. 01 ↗</span>
      </div>
      <svg
        viewBox="0 0 600 525"
        role="img"
        aria-label="Example moderation signal flow"
      >
        <defs>
          <pattern
            id="heroGrid"
            width="26"
            height="26"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M26 0H0V26"
              fill="none"
              stroke="#d0dbe1"
              strokeWidth=".55"
            />
          </pattern>
          <marker
            id="heroArrow"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto"
          >
            <path
              d="m1 1 7 4-7 4"
              fill="none"
              stroke="#597b95"
              strokeWidth="1.2"
            />
          </marker>
        </defs>
        <rect width="600" height="525" fill="url(#heroGrid)" />
        <g className="hero-input">
          <path
            d="m49 61 291-4 2 100-294 4Z"
            fill="#fbfcf8"
            stroke="#7890a2"
            strokeWidth="1.2"
          />
          <text x="69" y="88" className="diagram-micro">
            INCOMING COMMENT / 001
          </text>
          <text x="69" y="120" className="comment-svg">
            “Nobody wants your
          </text>
          <text x="69" y="143" className="comment-svg">
            stupid opinions.”
          </text>
          <path
            d="M69 147h142"
            stroke="#c8856b"
            strokeWidth="7"
            opacity=".27"
          />
        </g>
        <path
          d="M342 110h92q21 0 21 21v49"
          fill="none"
          stroke="#597b95"
          strokeDasharray="4 5"
          markerEnd="url(#heroArrow)"
        />
        <text x="365" y="97" className="diagram-micro">
          / CLASSIFY
        </text>
        <g className="engine-layer">
          <path
            d="m142 245 158-82 159 81-158 84Z"
            fill="#dce8ed"
            stroke="#597b95"
            strokeWidth="1.3"
          />
          <path
            d="m142 245 1 39 158 84 158-85v-39m-158 84v40"
            fill="#edf3f3"
            stroke="#597b95"
            strokeWidth="1.3"
          />
          <path
            d="m142 284 159 84 158-85"
            fill="none"
            stroke="#597b95"
            strokeWidth="1.3"
          />
          <path
            d="m178 244 122-63 122 62-121 64Z"
            fill="none"
            stroke="#597b95"
            strokeDasharray="3 4"
            opacity=".6"
          />
          <path
            d="M286 267v-52m0 2 33 16-10 8 10 18-33-15"
            fill="none"
            stroke="#3d617b"
            strokeWidth="2"
          />
          <text
            x="289"
            y="340"
            className="diagram-micro"
            transform="rotate(28 289 340)"
          >
            DISTILBERT
          </text>
        </g>
        <path
          d="M300 368v32"
          fill="none"
          stroke="#597b95"
          markerEnd="url(#heroArrow)"
        />
        <g className="hero-output">
          <path
            d="m114 409 374-2 2 79-376 2Z"
            fill="#fafbf6"
            stroke="#7890a2"
            strokeWidth="1.2"
          />
          <text x="135" y="432" className="diagram-micro">
            SIX LABELS. ONE CLEARER PICTURE.
          </text>
          {[0.88, 0.16, 0.4, 0.08, 0.93, 0.04].map((v, i) => (
            <g key={i}>
              <rect
                x={135 + i * 58}
                y="445"
                width="42"
                height="5"
                fill="#e1e5de"
              />
              <rect
                x={135 + i * 58}
                y="445"
                width={v * 42}
                height="5"
                fill={i === 0 || i === 4 ? "#b76a50" : "#7d9bab"}
              />
              <text x={135 + i * 58} y="471" className="diagram-micro">
                {["TOX", "SEV", "OBS", "THR", "INS", "IDH"][i]}
              </text>
            </g>
          ))}
        </g>
        <text
          x="40"
          y="343"
          className="sketch-annotation"
          transform="rotate(-8 40 343)"
        >
          a little context
        </text>
        <text
          x="47"
          y="364"
          className="sketch-annotation"
          transform="rotate(-8 47 364)"
        >
          goes a long way.
        </text>
        <path
          d="M93 314q-17-40 29-59m-13 1 13-1-4 12"
          fill="none"
          stroke="#7890a2"
          strokeWidth="1.3"
        />
        <circle className="signal-dot" cx="455" cy="159" r="4" fill="#a4bd7b" />
      </svg>
      <div className="sketch-bottom mono">
        <span className="tiny-cross">+</span>ILLUSTRATIVE SIGNAL · NOT A LIVE
        PREDICTION<span className="tiny-cross">+</span>
      </div>
    </div>
  );
}
function Journey() {
  const [step, setStep] = useState(0);
  return (
    <div className="journey" data-reveal>
      <div
        className="journey-tabs"
        role="group"
        aria-label="Moderation journey"
      >
        {[
          ["Classify", "Find the signal."],
          ["Explain", "See what contributed."],
          ["Correct", "Keep a human in the loop."],
        ].map(([title, sub], i) => (
          <button
            aria-pressed={step === i}
            onClick={() => setStep(i)}
            key={title}
          >
            <span className="mono">0{i + 1} /</span>
            <h3>
              {title}
              <Arrow />
            </h3>
            <p>{sub}</p>
            <span className="journey-tab-line" />
          </button>
        ))}
      </div>
      <div className="journey-visual">
        <div className="journey-comment">
          <span className="mono eyebrow">AN ILLUSTRATIVE COMMENT</span>
          <p>
            “Nobody wants your{" "}
            <mark className={step === 1 ? "emphasis" : ""}>
              stupid opinions
            </mark>
            .”
          </p>
          <div className="comment-context">
            <span className="avatar">U</span>Submitted in a community discussion
          </div>
        </div>
        <div className="journey-connector">
          <span />
          <Arrow />
          <span />
        </div>
        <div className="journey-result" aria-live="polite">
          {step === 0 ? (
            <>
              <span className="mono eyebrow">
                PREDICTION / SIX INDEPENDENT LABELS
              </span>
              <div className="example-label">
                <span>Insult</span>
                <strong>93%</strong>
                <i style={{ width: "93%" }} />
              </div>
              <div className="example-label">
                <span>Toxic</span>
                <strong>88%</strong>
                <i style={{ width: "88%" }} />
              </div>
              <small>Illustrative scores · multiple labels can apply</small>
            </>
          ) : step === 1 ? (
            <>
              <span className="mono eyebrow">
                INTEGRATED GRADIENTS / INSULT
              </span>
              <h3>The words behind the flag.</h3>
              <p>
                Token contributions show which parts of the text support or
                oppose the selected label.
              </p>
              <span className="contributor-example">
                stupid <span>stronger contribution ↗</span>
              </span>
            </>
          ) : (
            <>
              <span className="mono eyebrow">
                HUMAN REVIEW / CORRECTION EVENT
              </span>
              <h3>A prediction is a starting point.</h3>
              <p>
                Reviewers can correct all six labels. Each decision is saved as
                an audit event for future retraining.
              </p>
              <span className="saved-example">
                <span />
                Correction recorded
              </span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
export function Landing() {
  useReveal();
  return (
    <>
      <Header />
      <main tabIndex={-1} id="main-content">
        <section className="hero wrap">
          <div className="hero-copy">
            <div className="eyebrow mono">
              <span className="green-square" />
              EXPLAINABLE CONTENT MODERATION
            </div>
            <h1>
              Flag harmful
              <br />
              content.
              <br />
              <span className="soft-word">Understand</span>
              <br />
              <span className="why-word">
                why
                <svg viewBox="0 0 160 13" aria-hidden="true">
                  <path d="M2 9Q70 0 157 5M8 12Q87 5 148 8" />
                </svg>
              </span>
              .
            </h1>
            <p className="hero-description">
              Good conversations deserve better protection. Flagged identifies
              harmful text across six categories—and makes the reasoning
              visible.
            </p>
            <div className="hero-actions">
              <a href="/demo" className="button dark">
                Try the demo
                <Arrow />
              </a>
              <a href="#architecture" className="text-link">
                Explore the system <span>↓</span>
              </a>
            </div>
            <div className="hero-foot mono">
              <span>NO SIGN-UP. JUST TEXT.</span>
              <Status />
            </div>
          </div>
          <HeroSketch />
        </section>
        <div className="context-strip wrap">
          <span className="mono">
            BUILT FOR THE PLACES
            <br />
            WHERE CONVERSATIONS HAPPEN
          </span>
          <div>
            <span className="context-icon">☷</span>Comment sections
          </div>
          <div>
            <span className="context-icon">↳</span>Community forums
          </div>
          <div>
            <span className="context-icon">◫</span>Chat applications
          </div>
        </div>
        <UseCaseHighlights />
        <section id="how-it-works" className="section wrap">
          <SectionLabel number="01">FROM TEXT TO UNDERSTANDING</SectionLabel>
          <div className="section-heading" data-reveal>
            <h2>
              More than a flag.
              <br />
              <span className="muted">A reason to act.</span>
            </h2>
            <p>
              Speed matters. So does context. Flagged connects real-time
              classification with explainable predictions and human
              review—without slowing the conversation.
            </p>
          </div>
          <Journey />
        </section>
        <section id="architecture" className="architecture-section">
          <div className="wrap">
            <SectionLabel number="02">AN END-TO-END SYSTEM</SectionLabel>
            <div className="section-heading" data-reveal>
              <h2>
                Small footprint.
                <br />
                <span className="muted">Considered architecture.</span>
              </h2>
              <p>
                A model is only the beginning. Durable storage, background
                explanations, and observability make the pipeline a service you
                can build on.
              </p>
            </div>
            <Architecture />
          </div>
        </section>
        <section id="performance" className="section performance wrap">
          <SectionLabel number="03">MEASURED, NOT ASSUMED</SectionLabel>
          <div className="section-heading" data-reveal>
            <h2>
              Built for the
              <br />
              <span className="muted">pace of a conversation.</span>
            </h2>
            <p>
              CPU inference. Dynamic batching. A measured classification
              workload of 50 requests per second.
            </p>
          </div>
          <div className="metrics" data-reveal>
            <div>
              <span className="mono">P95 CLASSIFICATION LATENCY</span>
              <strong>
                84<span>ms</span>
              </strong>
              <p>95% of requests within 84 milliseconds.</p>
              <div className="metric-spark">
                {[
                  28, 39, 30, 44, 35, 49, 39, 29, 52, 43, 33, 42, 58, 46, 40,
                  34, 48, 63, 50, 41, 54, 45, 57, 38, 47, 33, 51, 40,
                ].map((h, i) => (
                  <i key={i} style={{ height: h + "%" }} />
                ))}
              </div>
            </div>
            <div>
              <span className="mono">SUCCESSFUL THROUGHPUT</span>
              <strong>
                49.99<span>/s</span>
              </strong>
              <p>Classifications served per second.</p>
              <div className="throughput-line">
                <svg viewBox="0 0 250 45" aria-hidden="true">
                  <path
                    d="m0 30 14-4 13 4 12-8 13 3 12-7 14 4 13-6 12 5 13-3 12 2 13-5 13 1 13 2 13-1 13 2 12-1 15 1"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                  />
                </svg>
              </div>
            </div>
            <div>
              <span className="mono">MEASURED REQUEST ERRORS</span>
              <strong>
                0<span>%</span>
              </strong>
              <p>Across the 180-second measured run.</p>
              <div className="zero-line">
                <span />
                NO DROPPED ITERATIONS
              </div>
            </div>
          </div>
          <div className="benchmark-caption">
            <span className="mono">
              CLASSIFICATION ONLY · 180 SECONDS · OFFERED LOAD: 50/s
            </span>
            <p>
              Supplied benchmark, recorded before the commit-before-response
              update. These figures exclude explanations and must be remeasured
              for the updated API.
            </p>
          </div>
          <details className="benchmark-details">
            <summary>
              Look closer at the benchmark<span>+</span>
            </summary>
            <dl>
              {[
                ["Mean latency", "43.9 ms"],
                ["Median latency", "34 ms"],
                ["p99 latency", "185 ms"],
                ["Average sampled host CPU", "66.9%"],
                ["Dropped load-generator iterations", "0"],
              ].map(([k, v]) => (
                <div key={k}>
                  <dt>{k}</dt>
                  <dd>{v}</dd>
                </div>
              ))}
            </dl>
            <p>
              The small chart graphics above are illustrative. The numerical
              measurements are from the supplied benchmark.
            </p>
          </details>
        </section>
        <section className="model-section">
          <div className="wrap model-layout">
            <div data-reveal>
              <SectionLabel number="04">THE MODEL, IN CONTEXT</SectionLabel>
              <h2>
                Six signals.
                <br />
                Room for nuance.
              </h2>
              <p>
                Harm rarely fits into a single category. A fine-tuned DistilBERT
                model scores six labels independently, so a comment can trigger
                several at once.
              </p>
              <p>
                Per-label thresholds are tuned with F2 to put more weight on
                recall. The goal: miss less harmful content while keeping
                decisions open to human review.
              </p>
              <a href="/demo" className="text-link">
                See the signals for yourself
                <Arrow />
              </a>
            </div>
            <div className="model-spec" data-reveal>
              <div className="model-spec-title">
                <span className="mono">MODEL CARD / 001</span>
                <span className="version-pill">v2.0.0</span>
              </div>
              <h3>DistilBERT</h3>
              <p className="mono">FINE-TUNED · MULTILABEL CLASSIFIER</p>
              <div className="label-cloud">
                {[
                  "Toxic",
                  "Severe toxicity",
                  "Obscene",
                  "Threat",
                  "Insult",
                  "Identity hate",
                ].map((l, i) => (
                  <span key={l}>
                    <small>0{i + 1}</small>
                    {l}
                  </span>
                ))}
              </div>
              <dl>
                <div>
                  <dt>Inference</dt>
                  <dd>FP32 ONNX · CPU</dd>
                </div>
                <div>
                  <dt>Serving</dt>
                  <dd>FastAPI + NVIDIA Triton</dd>
                </div>
                <div>
                  <dt>Explanations</dt>
                  <dd>Integrated Gradients</dd>
                </div>
                <div>
                  <dt>Learning loop</dt>
                  <dd>Saved human corrections</dd>
                </div>
              </dl>
              <span className="model-footnote mono">
                NO QUANTIZATION · NO GENERATIVE MODEL
              </span>
            </div>
          </div>
        </section>
        <section className="section wrap developer-section">
          <SectionLabel number="05">MAKE IT PART OF YOUR STACK</SectionLabel>
          <div className="developer-layout">
            <div data-reveal>
              <h2>
                A simple request.
                <br />
                <span className="muted">A better-informed decision.</span>
              </h2>
              <p>
                Send text. Get six scores and a saved classification ID. Request
                an explanation when a decision needs a closer look.
              </p>
              <div className="hero-actions">
                <a href="/demo" className="button dark">
                  Try it with your text
                  <Arrow />
                </a>
                <a
                  href={repo}
                  target="_blank"
                  rel="noreferrer"
                  className="text-link"
                >
                  View source
                  <Arrow diagonal />
                </a>
              </div>
            </div>
            <div className="code-window" data-reveal>
              <div className="code-title mono">
                <span>01 / CLASSIFY A COMMENT</span>
                <span>HTTP</span>
              </div>
              <pre>
                <code>
                  <span className="code-green">POST</span>{" "}
                  https://flagged.duckdns.org/v1/classify{"\n"}
                  <span className="code-muted">
                    Content-Type: application/json
                  </span>
                  {"\n\n"}
                  {"{"}
                  {"\n"} <span className="code-green">"text"</span>: "Thanks for
                  helping me understand."{"\n"}
                  {"}"}
                </code>
              </pre>
              <div className="code-footer mono">
                <span />
                SIX SCORES + ONE CANONICAL ID
              </div>
            </div>
          </div>
        </section>
        <section className="closing wrap">
          <span className="mono">LESS HARM. MORE UNDERSTANDING.</span>
          <h2>
            Keep the conversation
            <br />
            worth having.
          </h2>
          <a href="/demo" className="button dark">
            Explore the live demo
            <Arrow />
          </a>
          <div className="closing-sketch" aria-hidden="true">
            ↳
          </div>
        </section>
        <section id="contact" className="contact-section wrap" aria-labelledby="contact-heading">
          <SectionLabel number="06">LET'S BUILD BETTER CONVERSATIONS</SectionLabel>
          <div className="contact-layout" data-reveal>
            <div className="contact-copy">
              <h2 id="contact-heading">
                Bring Flagged<br />
                <span className="muted">to your platform.</span>
              </h2>
              <p>
                Want to use the backend API directly? Get in touch to discuss
                your use case, integration, and API access.
              </p>
            </div>
            <div className="contact-links">
              <a className="contact-link" href="mailto:bzioumohamedmouldi@gmail.com">
                <span className="contact-mark" aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="5" width="18" height="14" rx="1" />
                    <path d="m3 6 9 7 9-7" />
                  </svg>
                </span>
                <span className="contact-link-copy">
                  <span className="mono">EMAIL / SAY HELLO</span>
                  <span>bzioumohamedmouldi@gmail.com</span>
                </span>
                <Arrow diagonal />
              </a>
              <a className="contact-link" href="https://www.linkedin.com/in/mohamedmouldibziou/" target="_blank" rel="noreferrer">
                <span className="contact-mark" aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="currentColor">
                    <rect x="3" y="9" width="4" height="12" rx=".5" />
                    <circle cx="5" cy="5" r="2" />
                    <path d="M10 9h4v1.5a4.7 4.7 0 0 1 3.8-1.8c3.2 0 4.2 2 4.2 5.2V21h-4v-6.3c0-1.6-.3-2.8-1.9-2.8-1.7 0-2.1 1.3-2.1 2.8V21h-4Z" />
                  </svg>
                </span>
                <span className="contact-link-copy">
                  <span className="mono">LINKEDIN / LET'S CONNECT</span>
                  <span>Discuss your next integration</span>
                </span>
                <Arrow diagonal />
              </a>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
}
