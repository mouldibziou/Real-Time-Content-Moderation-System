import { useEffect, useState } from "react";
import { api } from "./api";
export function Arrow({ diagonal = false }: { diagonal?: boolean }) {
  return (
    <svg
      aria-hidden="true"
      width="18"
      height="18"
      viewBox="0 0 20 20"
      fill="none"
    >
      <path
        d={diagonal ? "M5 15 15 5M5 5h10v10" : "M3 10h13m-5-5 5 5-5 5"}
        stroke="currentColor"
        strokeWidth="1.5"
      />
    </svg>
  );
}
export function Flag({ className = "" }: { className?: string }) {
  return (
    <svg
      className={className}
      aria-hidden="true"
      viewBox="0 0 32 32"
      fill="none"
    >
      <path
        d="M7 28V4m0 1h19l-5 7 5 7H7"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
      />
    </svg>
  );
}
export function Status() {
  const [status, setStatus] = useState<"checking" | "ready" | "offline">(
    "checking",
  );
  useEffect(() => {
    const controller = new AbortController();
    const update = () => {
      if (document.visibilityState === "hidden") return;
      api
        .ready(controller.signal)
        .then((r) => setStatus(r.is_ready ? "ready" : "offline"))
        .catch(() => {
          if (!controller.signal.aborted) setStatus("offline");
        });
    };
    update();
    const interval = setInterval(update, 30000);
    document.addEventListener("visibilitychange", update);
    return () => {
      controller.abort();
      clearInterval(interval);
      document.removeEventListener("visibilitychange", update);
    };
  }, []);
  return (
    <span
      className={`service-status ${status}`}
      title="Live API readiness check; not a historical uptime measurement"
    >
      <i />
      {status === "checking"
        ? "Checking API"
        : status === "ready"
          ? "API operational"
          : "API unavailable"}
    </span>
  );
}
export function Header({ demo = false }: { demo?: boolean }) {
  return (
    <header className="header">
      <div className="header-inner">
        <a href="/" className="wordmark" aria-label="Flagged home">
          <Flag />
          flagged<span className="wordmark-dot">.</span>
        </a>
        <nav aria-label="Main navigation">
          <a href="/#how-it-works">How it works</a>
          <a href="/#architecture">Architecture</a>
          <a href="/#performance">Performance</a>
        </nav>
        <a
          className={`button small ${demo ? "outline" : "dark"}`}
          href={demo ? "/" : "/demo"}
        >
          {demo ? "Explore the project" : "Try the demo"}
          <Arrow diagonal={!demo} />
        </a>
      </div>
    </header>
  );
}
export function Footer() {
  return (
    <footer className="footer wrap">
      <a href="/" className="wordmark">
        <Flag />
        flagged.
      </a>
      <p>Better signals. More thoughtful moderation.</p>
      <a
        href="https://github.com/mouldibziou/Real-Time-Content-Moderation-System"
        target="_blank"
        rel="noreferrer"
      >
        Explore the source <Arrow diagonal />
      </a>
      <span className="mono">BUILT WITH INTENT · FLAGGED</span>
    </footer>
  );
}
export function SectionLabel({
  number,
  children,
}: {
  number: string;
  children: React.ReactNode;
}) {
  return (
    <div className="section-label mono">
      <span>[ {number} ]</span>
      {children}
    </div>
  );
}
export function useReveal() {
  useEffect(() => {
    const items = document.querySelectorAll("[data-reveal]");
    if (
      matchMedia("(prefers-reduced-motion: reduce)").matches ||
      typeof IntersectionObserver === "undefined"
    )
      return;
    const observer = new IntersectionObserver(
      (entries) =>
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("revealed");
            observer.unobserve(entry.target);
          }
        }),
      { threshold: 0.1 },
    );
    items.forEach((item) => {
      item.classList.add("reveal-ready");
      observer.observe(item);
    });
    return () => observer.disconnect();
  }, []);
}
