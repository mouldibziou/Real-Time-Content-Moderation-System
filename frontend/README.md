# Flagged frontend

A React + TypeScript moderation workspace built with Vite. `/` introduces the project; `/demo` connects to the API for classification, explanations, batch inspection, and saved moderator corrections. Fonts and graphics are bundled locally.

**[Live demo](https://contentflagged.tech/demo) · [Backend setup and architecture](../README.md)**

## Local development

Use **Node 24** and **pnpm 11.25.0**:

```sh
cd frontend
pnpm install --frozen-lockfile
cp .env.example .env.local
pnpm dev
```

The example API URL is `https://flagged.duckdns.org`. To use the local Docker backend, set this in `.env.local`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Start the backend using the [root setup instructions](../README.md#backend), including its CORS origins. The explanation worker must run for uncached explanations to finish. All `VITE_` values are public build-time configuration; do not put credentials in them.

The workspace displays independent scores and metadata-derived thresholds, polls real explanation jobs, renders Unicode attribution spans, and retains server-confirmed review receipts. It reports API failures rather than substituting predictions. Submitted content is persisted; use non-sensitive development examples.

## Checks

```sh
pnpm test
pnpm build
pnpm preview
```

Tests cover API errors, thresholds, cached/asynchronous explanations, Unicode offsets, polling cancellation/timeouts, batch order, and correction submission/retry. The build is emitted to `dist/`.

## Cloudflare Pages

| Setting | Value |
| --- | --- |
| Root directory | `frontend` |
| Build command | `pnpm build` |
| Output directory | `dist` |
| Node version | `24` |
| pnpm version | `11.25.0` |
| `VITE_API_BASE_URL` | `https://flagged.duckdns.org` |

Use the checked-in lockfile. Set `NODE_VERSION=24` and `PNPM_VERSION=11.25.0` if the build image does not infer them. The frontend is static; no Cloudflare Functions are required.

Cloudflare Pages' built-in SPA fallback serves `/demo` without changing its URL. Avoid an explicit rewrite to `/index.html`, which Pages canonicalizes to `/`, and a top-level `404.html` that overrides the fallback. If the API origin changes, update `connect-src` in [`public/_headers`](public/_headers) as well as the build variable.

The API must permit the exact frontend origin through `TOXICITY_CORS_ORIGINS`. CORS does not provide authentication. The hosted demo has no login; private integrations need an appropriate access policy.

Landing-page latency figures describe a historical classification-only run before commit-before-response persistence. In-demo timing measures browser request round trip. See the [measurement scope](../README.md#model-and-measurements) before comparing them.
