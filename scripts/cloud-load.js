import http from 'k6/http';
import { sleep } from 'k6';
import exec from 'k6/execution';
import { Counter, Rate, Trend } from 'k6/metrics';

const base = (__ENV.BASE_URL || 'http://localhost:18000').replace(/\/$/, '');
const rate = Number(__ENV.RATE || 1);
const duration = __ENV.DURATION || '2m';
const explanationShare = Number(__ENV.EXPLAIN_SHARE || 0.05);
const runId = __ENV.RUN_ID || String(Date.now());
if (!Number.isInteger(rate) || rate < 1 || !Number.isFinite(explanationShare) || explanationShare < 0 || explanationShare > 1) {
  throw new Error('RATE must be a positive integer; EXPLAIN_SHARE must be between 0 and 1.');
}

const wallTime = new Trend('client_wall_ms', true);
const successfulLatency = new Trend('successful_request_ms', true);
const failures = new Rate('application_request_failed');
const successes = new Counter('successful_requests');
const items = new Counter('classified_items');
const statuses = new Counter('response_statuses');
const jobsSubmitted = new Counter('explanation_jobs_submitted');
const jobsFinished = new Rate('observed_explanation_completed');
const completionTime = new Trend('observed_explanation_completion_ms', true);

export const options = {
  scenarios: {
    traffic: {
      executor: 'constant-arrival-rate', rate, timeUnit: '1s', duration,
      preAllocatedVUs: 16, maxVUs: 64, gracefulStop: '150s',
    },
  },
  setupTimeout: '60s',
  summaryTrendStats: ['avg', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
  thresholds: Object.fromEntries(
    ['classify', 'classify_batch', 'read', 'explain_submit', 'job_poll', 'explain_cached']
      .map(operation => [`successful_request_ms{operation:${operation}}`, ['avg>=0']])
  ),
};

const comments = [
  'Thank you for explaining this. The example helped me understand the issue.',
  'I disagree with the proposed change because it makes the interface harder to use.',
  'You are an idiot and your argument is ridiculous.',
  'Merci pour cette réponse, elle est très utile.',
  'شكراً على الشرح، هذا مفيد جداً.',
  'The update works well overall, but searching older posts still feels slow. ' .repeat(8),
];

function text(index) {
  const suffix = ` [synthetic test ${runId}/${exec.vu.idInTest}/${exec.scenario.iterationInTest}/${index}]`;
  return comments[Math.floor(Math.random() * comments.length)] + suffix;
}

function request(method, path, payload, operation, expected = [200]) {
  const started = Date.now();
  const response = http.request(method, base + path, payload === null ? null : JSON.stringify(payload), {
    headers: { 'Content-Type': 'application/json' }, timeout: '30s',
    tags: { name: operation, operation },
    responseCallback: http.expectedStatuses(...expected),
  });
  const elapsed = Date.now() - started;
  const ok = expected.includes(response.status);
  wallTime.add(elapsed, { operation });
  failures.add(!ok, { operation });
  statuses.add(1, { operation, status: String(response.status) });
  if (ok) {
    successes.add(1, { operation });
    successfulLatency.add(elapsed, { operation });
  }
  let body = {};
  if (ok) {
    try { body = response.json(); } catch (_) { /* HTTP metrics still capture the response. */ }
  }
  return { ok, response, body };
}

export function setup() {
  const readiness = http.get(base + '/v1/health/ready', { timeout: '15s', tags: { name: 'preflight' } });
  if (readiness.status !== 200) throw new Error(`Readiness failed: HTTP ${readiness.status}`);
  const seed = http.post(base + '/v1/classify', JSON.stringify({ text: `Synthetic read seed ${runId}: Thank you for your help.` }), {
    headers: { 'Content-Type': 'application/json' }, timeout: '30s', tags: { name: 'preflight' },
  });
  if (seed.status !== 200 || !seed.json().classification_id) throw new Error('Unable to create read seed.');
  return { readId: seed.json().classification_id };
}

export default function (data) {
  const lane = Math.random();
  if (lane < 0.2) {
    request('GET', `/v1/classifications/${data.readId}`, null, 'read');
    return;
  }
  if (lane < 0.3) {
    const result = request('POST', '/v1/classify_batch', { texts: [text(0), text(1), text(2), text(3)] }, 'classify_batch');
    if (result.ok && Array.isArray(result.body.results)) items.add(result.body.results.length);
    return;
  }
  const classification = request('POST', '/v1/classify', { text: text(0) }, 'classify');
  if (!classification.ok || !classification.body.classification_id) return;
  items.add(1);
  if (Math.random() >= explanationShare) return;
  const payload = { classification_id: classification.body.classification_id, label: 'toxic' };
  const start = Date.now();
  const submitted = request('POST', '/v1/explain', payload, 'explain_submit', [200, 202]);
  if (!submitted.ok) return;
  if (submitted.response.status === 200) return;
  jobsSubmitted.add(1);
  if (!submitted.body.job_id) { jobsFinished.add(false); return; }
  const deadline = start + 120000;
  while (Date.now() < deadline) {
    sleep(2);
    const job = request('GET', `/v1/jobs/${submitted.body.job_id}`, null, 'job_poll');
    if (job.ok && job.body.status === 'done' && job.body.result) {
      jobsFinished.add(true);
      completionTime.add(Date.now() - start);
      request('POST', '/v1/explain', payload, 'explain_cached');
      return;
    }
    if (job.ok && job.body.status === 'failed') { jobsFinished.add(false); return; }
  }
  jobsFinished.add(false);
}

export function handleSummary(data) {
  const report = {
    base_url: base, run_id: runId, offered_journeys_per_second: rate,
    scheduled_duration: duration, explanation_share_of_single_classifications: explanationShare,
    assumptions: {
      journeys: '70% single classification, 10% four-item batch, 20% read; sampled explanations also poll and check cache.',
      network: 'localhost:18000 measures the SSH tunnel path, not the future public HTTPS endpoint.',
      completion: 'Observed completion includes up to two seconds of polling delay; an observation timeout does not cancel the job.',
      throughput: 'Use successful_requests and classified_items; HTTP totals include failed requests and preflight.',
      drain: 'In-flight journeys may continue after the scheduled arrival window; aggregate rates include drain.',
      percentiles: 'No mean or percentile subtraction can isolate network latency from server histograms.',
    },
    metrics: data.metrics,
  };
  const rendered = JSON.stringify(report, null, 2) + '\n';
  return { [__ENV.REPORT_PATH || 'cloud-load-report.json']: rendered, stdout: rendered };
}
