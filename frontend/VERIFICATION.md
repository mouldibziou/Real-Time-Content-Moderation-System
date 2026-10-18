# Verification — 19 October 2026

- Frontend: 20 tests passed; TypeScript and Vite production build passed.
- Complete backend suite in the existing Linux API image: 247 tests, 24 intentionally skipped, no failures. The macOS run encounters seven pre-existing Linux CPU-affinity test assumptions; the Linux run covers them.
- Disposable PostgreSQL database-enabled suite: 22 tests passed, including concurrent duplicate classifications, overlapping batches, immediate detail/explanation/correction, and rollback on single/batch persistence failures.
- Real local DistilBERT inference and the actual PyTorch/Captum worker were exercised through the browser with disposable PostgreSQL and Redis. Queued completion, subsequent cached explanation, saved review receipt, and four-item batch inspection worked. No production records were submitted.
- Browser: desktop and 320/375/390 px mobile layouts, no horizontal overflow, keyboard skip target, metadata loading, and reduced-motion rendering checked.
- Pinned Caddy image: configuration validates; local integration returns 200 for readiness/metadata/detail/job reads, 202 for explanation submission, 201 for correction, and correct CORS preflight. Global history, metrics, and docs return 404 at ingress.
- Production output uses https://flagged.duckdns.org, includes the /demo rewrite, local font files, and both font licenses. The temporary local API override was removed after QA.
- AWS and Cloudflare were not deployed. The old performance benchmark remains explicitly historical; no new production capacity claim was measured.

See README.md for local preview, deployment order, and the explicit performance recheck. Temporary QA services are stopped after verification.
