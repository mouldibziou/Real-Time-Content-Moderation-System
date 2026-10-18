# Flagged frontend

A static React + TypeScript moderation experience, built with Vite. `/` tells the project story; `/demo` connects directly to the API. No authentication, frontend server, or Cloudflare Functions are required.

## Local development

Use Node 24 and pnpm 11.25.0 (declared in `package.json`). With Corepack installed, `corepack enable` makes the package manager available.

```sh
cd frontend
pnpm install --frozen-lockfile
cp .env.example .env.local
pnpm dev
```

Open the printed local URL. All `VITE_` variables are public build-time configuration. The default API is `https://flagged.duckdns.org`.

The API changes in this repository must be rolled out before the public demo can work. The currently hosted API may reject browser preflights or block metadata/explanation routes until that rollout. The interface reports this as unavailable; it does not invent predictions.

For a local backend, set `.env.local` to `VITE_API_BASE_URL=http://127.0.0.1:8001`, and set the API process environment to:

```text
TOXICITY_CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
```

Use your existing local PostgreSQL/Redis development setup and model artifacts, then run these from the repository root in separate terminals:

```sh
venv/bin/python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8001
venv/bin/python -m app.worker
```

Set `DATABASE_URL`, `REDIS_URL`, and any model settings in those terminals as described in the backend docs. Keep development data separate from the production database. The worker must be running for uncached explanations to finish.

## Validation

```sh
pnpm test
pnpm build
pnpm preview
```

Tests exercise API errors, independent scores, metadata-derived thresholds, cached and asynchronous explanations, Unicode offsets, polling cancellation and timeout, batch order, and correction persistence/retry. The production build is emitted to `dist/`.

Backend browser-contract tests:

```sh
venv/bin/python -m unittest tests.test_frontend_contract
```

The database tests additionally require `TOXICITY_DB_TESTS=1` and a migrated disposable PostgreSQL database whose name ends in `_tests`. They test concurrent duplicate requests, overlapping batches, immediate follow-up operations, and transaction rollback. Never point them at production.

## Cloudflare Pages

Create a Pages project connected to this repository:

| Setting | Value |
| --- | --- |
| Root directory | `frontend` |
| Framework | React / Vite |
| Build command | `pnpm build` |
| Output directory | `dist` |
| Node version | `24` |
| pnpm version | `11.25.0` |
| `VITE_API_BASE_URL` | `https://flagged.duckdns.org` |

Install with the checked-in pnpm lockfile. Set `PNPM_VERSION=11.25.0` and `NODE_VERSION=24` in Pages if the build image does not infer them. Fonts and graphics are bundled locally. There is no tracking or analytics integration.

`public/_redirects` explicitly maps `/demo` to the SPA entry document. The Pages SPA fallback handles ordinary client navigation. Do not add a top-level `404.html` that overrides this behavior. If you change the API origin, update `connect-src` in `public/_headers` as well as the environment variable.

## AWS API rollout

The code intentionally changes successful classification responses to wait for the PostgreSQL commit. Repeated text returns the existing canonical ID, so explanations, corrections, and cache reuse work immediately. A failed database commit now returns 503. No schema migration is introduced by this change.

1. Choose the final Pages hostname. Set the API environment variable to exact permitted origins, for example:

   ```text
   TOXICITY_CORS_ORIGINS=https://YOUR-PROJECT.pages.dev,https://YOUR-CUSTOM-DOMAIN
   ```

   Remove the example custom domain if unused. Origins have no trailing slash. Preview deployments require adding their exact origin explicitly. CORS is not authentication or a user quota. The public corrections are marked `public-demo` and should be reviewed before any training use.

2. Build and publish a new API image from this checkout with the project's existing image registry/tag procedure. For the managed-services topology, retain the existing `compose.cloud.yaml` + `compose.triton.yaml` + `compose.https.yaml` stack. The Cloudflare frontend is built separately and excluded from backend Docker build context. The worker, Triton, and database schema do not need an application change for this frontend.

3. On EC2, place the updated Compose files and `docker/caddy/Caddyfile` alongside the deployment. Keep the existing `.env.cloud`, managed-service URLs, certificates, and image prefix. Set its new `IMAGE_TAG` and `TOXICITY_CORS_ORIGINS`, then run:

   ```sh
   docker compose --env-file .env.cloud -f compose.cloud.yaml -f compose.triton.yaml -f compose.https.yaml config --quiet
   docker compose --env-file .env.cloud -f compose.cloud.yaml -f compose.triton.yaml -f compose.https.yaml pull api
   docker compose --env-file .env.cloud -f compose.cloud.yaml -f compose.triton.yaml -f compose.https.yaml up -d --no-build --no-deps api
   docker compose --env-file .env.cloud -f compose.cloud.yaml -f compose.triton.yaml -f compose.https.yaml exec caddy caddy validate --config /etc/caddy/Caddyfile
   docker compose --env-file .env.cloud -f compose.cloud.yaml -f compose.triton.yaml -f compose.https.yaml exec caddy caddy reload --config /etc/caddy/Caddyfile
   ```

   If this installation uses a differently named environment file, use that existing file consistently. Do not use the local-database EC2 overlay for the managed PostgreSQL/Redis topology.

4. Verify readiness, metadata, and the exact browser origin:

   ```sh
   curl --fail https://flagged.duckdns.org/v1/health/ready
   curl --fail https://flagged.duckdns.org/v1/model_info
   curl -i -X OPTIONS https://flagged.duckdns.org/v1/classify -H 'Origin: https://YOUR-PROJECT.pages.dev' -H 'Access-Control-Request-Method: POST' -H 'Access-Control-Request-Headers: content-type'
   ```

   Expect readiness `true`, version/limits in metadata, and an origin-specific CORS response. `/v1/classifications` (the global feed), `/metrics/`, `/docs`, and monitoring interfaces remain outside the Caddy allowlist. Only individual result IDs are used by the demo.

5. Deploy the Pages build and exercise single classification → explanation → saved review, repeated examples, and batch inspection from its actual hostname. Confirm `/demo` loads after a hard refresh. Public submissions are persisted; there is no automatic retraining or deletion interface in this frontend.

## Performance recheck

The landing-page numbers are the supplied classification-only 180-second measurement **before** committing persistence in the response path. They are static historical measurements, not current live guarantees. Readiness is an independent live probe. In-demo timing is browser request round trip.

After the API rollout, remeasure on the same EC2 resources, managed services, inference settings, client location/network path, and original text distribution. The supplied screenshot does not establish the original text distribution, so an exact workload reproduction requires that original input set. A prepared single-classification probe is available:

```sh
k6 run -e BASE_URL=https://flagged.duckdns.org -e RATE=50 -e DURATION=180s --summary-export=/tmp/flagged-classification-recheck.json scripts/frontend-classification-load.js
```

Run this from the repository root only when ready to load-test the service. It writes synthetic classification records and performs no corrections or explanations. Its default sample mix is a documented substitute, not proof of equivalence to the old benchmark. Compare p95/p99, successful classifications/s, errors, dropped iterations, and matching Grafana/host CPU observations. The script checks the target p95 <100 ms and zero errors/drops; a failed check must not be relabeled a pass. Update the visible figures and provenance only after collecting a new report.

No AWS or Cloudflare deployment is performed by the frontend build commands.
