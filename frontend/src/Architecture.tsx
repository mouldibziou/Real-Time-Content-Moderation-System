import { useState } from "react";
type Pathway = "classification" | "explanation" | "monitoring";
const notes: Record<Pathway, { title: string; text: string }> = {
  classification: {
    title: "A fast answer. A durable record.",
    text: "The browser calls FastAPI through Caddy. Triton scores the text on CPU, and PostgreSQL commits the result before its canonical ID returns.",
  },
  explanation: {
    title: "Deeper reasoning, off the critical path.",
    text: "FastAPI creates a durable job. Redis wakes the worker; PyTorch and Captum compute Integrated Gradients, then save the result for polling and reuse.",
  },
  monitoring: {
    title: "Every layer, observed.",
    text: "Prometheus scrapes the API, inference server, and explanation worker. Grafana turns those measurements into operational dashboards.",
  },
};
export function Architecture() {
  const [active, setActive] = useState<Pathway>("classification");
  const nodes = [
    {
      x: 30,
      y: 95,
      w: 150,
      h: 72,
      a: ["classification", "explanation"],
      title: "Browser",
      sub: "THE VISITOR",
    },
    {
      x: 30,
      y: 225,
      w: 150,
      h: 72,
      a: ["classification"],
      title: "Cloudflare Pages",
      sub: "STATIC FRONTEND",
    },
    {
      x: 250,
      y: 95,
      w: 136,
      h: 72,
      a: ["classification", "explanation"],
      title: "Caddy",
      sub: "HTTPS / ROUTING",
    },
    {
      x: 442,
      y: 95,
      w: 156,
      h: 72,
      a: ["classification", "explanation", "monitoring"],
      title: "FastAPI",
      sub: "VALIDATE · PERSIST",
    },
    {
      x: 666,
      y: 95,
      w: 180,
      h: 72,
      a: ["classification", "monitoring"],
      title: "NVIDIA Triton",
      sub: "FP32 ONNX · CPU",
    },
    {
      x: 442,
      y: 270,
      w: 156,
      h: 72,
      a: ["explanation"],
      title: "Managed Redis",
      sub: "JOB NOTIFICATION",
    },
    {
      x: 666,
      y: 270,
      w: 180,
      h: 72,
      a: ["explanation", "monitoring"],
      title: "Explain worker",
      sub: "PYTORCH + CAPTUM",
    },
    {
      x: 442,
      y: 440,
      w: 156,
      h: 72,
      a: ["classification", "explanation"],
      title: "PostgreSQL",
      sub: "MANAGED · DURABLE",
    },
    {
      x: 258,
      y: 440,
      w: 125,
      h: 72,
      a: [],
      title: "Migrations",
      sub: "DEPLOYMENT ONLY",
    },
    {
      x: 666,
      y: 440,
      w: 180,
      h: 72,
      a: ["monitoring"],
      title: "Prometheus",
      sub: "SCRAPE / MEASURE",
    },
    {
      x: 666,
      y: 560,
      w: 180,
      h: 62,
      a: ["monitoring"],
      title: "Grafana",
      sub: "OBSERVE / EXPLORE",
    },
  ];
  const edge = (d: string, paths: Pathway[], key: string) => (
    <path
      key={key}
      d={d}
      className={`arch-edge ${paths.includes(active) ? "active" : ""} ${active === "monitoring" ? "dotted" : ""}`}
      markerEnd="url(#arch-arrow)"
    />
  );
  return (
    <div className="architecture-interactive">
      <div
        className="path-controls"
        role="group"
        aria-label="Architecture pathway"
      >
        {(["classification", "explanation", "monitoring"] as const).map(
          (path, i) => (
            <button
              key={path}
              aria-pressed={active === path}
              onClick={() => setActive(path)}
            >
              <span className="mono">0{i + 1}</span>
              {path}
              <span className="path-dot" />
            </button>
          ),
        )}
      </div>
      <div className="architecture-layout">
        <div className="architecture-canvas">
          <svg
            role="img"
            aria-label={`${active} architecture pathway. ${notes[active].text}`}
            viewBox="0 0 900 660"
          >
            <defs>
              <pattern
                id="arch-grid"
                width="22"
                height="22"
                patternUnits="userSpaceOnUse"
              >
                <circle cx="1" cy="1" r=".7" fill="#cad4d9" />
              </pattern>
              <marker
                id="arch-arrow"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path
                  d="M1 1 9 5 1 9"
                  fill="none"
                  stroke="context-stroke"
                  strokeWidth="1.5"
                />
              </marker>
            </defs>
            <rect width="900" height="660" fill="url(#arch-grid)" />
            <path
              className="aws-boundary"
              d="M226 43 875 45 872 641 620 639 620 244 420 244 420 639 225 638Z"
            />
            <text x="246" y="30" className="diagram-micro">
              AWS EC2 · STOCKHOLM · 2 vCPU / 8 GiB
            </text>
            <text x="435" y="420" className="diagram-micro">
              MANAGED STORAGE
            </text>
            {edge("M180 131H250", ["classification", "explanation"], "a")}
            {edge("M105 225V168", ["classification"], "b")}
            {edge("M386 131H442", ["classification", "explanation"], "c")}
            {edge("M598 131H666", ["classification"], "d")}
            {edge("M520 167V270", ["explanation"], "e")}
            {edge("M598 306H666", ["explanation"], "f")}
            {edge("M442 151H410V476H442", ["classification", "explanation"], "h")}
            {edge("M756 342V385H572V440", ["explanation"], "i")}
            {edge("M383 476H442", [], "j")}
            {edge("M814 167V440", ["monitoring"], "k")}
            {edge("M847 306H861V476H846", ["monitoring"], "l")}
            {edge("M598 151H632V462H666", ["monitoring"], "m")}
            {edge("M756 512V560", ["monitoring"], "n")}
            {nodes.map((n) => (
              <g
                key={n.title}
                className={`arch-node ${n.a.includes(active) ? "active" : ""}`}
              >
                <rect x={n.x} y={n.y} width={n.w} height={n.h} rx="3" />
                <path
                  d={`M${n.x + 3} ${n.y + n.h + 3}H${n.x + n.w + 3}V${n.y + 3}`}
                  className="sketch-echo"
                />
                <text x={n.x + 13} y={n.y + 29} className="diagram-title">
                  {n.title}
                </text>
                <text x={n.x + 13} y={n.y + 49} className="diagram-micro">
                  {n.sub}
                </text>
              </g>
            ))}
            <text x="34" y="617" className="diagram-micro">
              FIG. 02 / SYSTEM MAP
            </text>
          </svg>
        </div>
        <div className="mobile-system-map" aria-label="System architecture">
          <div className="map-node">
            Browser <small>THE VISITOR</small>
          </div>
          <span className="map-arrow">↕</span>
          <div className="map-node">
            Cloudflare Pages <small>FRONTEND</small>
          </div>
          <span className="map-arrow caption">↓ BROWSER CALLS AWS OVER HTTPS</span>
          <div className="map-node">
            Caddy → FastAPI <small>API</small>
          </div>
          <span className="map-arrow">↕</span>
          <div className="map-node">
            NVIDIA Triton <small>ONNX · CPU</small>
          </div>
          <div className="map-storage">
            <span className="mono">MANAGED POSTGRESQL</span>
            <p>
              Classification results, corrections, and explanation jobs.
              Committed before the API responds.
            </p>
          </div>
          <div className="map-node">
            Redis → Explain worker <small>ASYNC JOBS</small>
          </div>
          <div className="map-node">
            Prometheus → Grafana <small>MONITORING</small>
          </div>
          <p className="mono">MIGRATION CONTAINER · DEPLOYMENT ONLY</p>
        </div>
        <aside className="architecture-note" aria-live="polite">
          <span className="mono eyebrow">THE {active.toUpperCase()} PATH</span>
          <h3>{notes[active].title}</h3>
          <p>{notes[active].text}</p>
          <div className="arch-note-bottom">
            <span className="hand-arrow">↳</span>
            <p>
              One system.
              <br />
              Three connected stories.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
