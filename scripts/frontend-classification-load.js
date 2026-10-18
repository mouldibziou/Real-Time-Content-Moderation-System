// Single-classification capacity probe. Writes synthetic records; run explicitly.
import http from 'k6/http';
import exec from 'k6/execution';
import {Counter, Rate, Trend} from 'k6/metrics';
const base = (__ENV.BASE_URL || 'http://localhost:8001').replace(/\/$/, '');
const rate = Number(__ENV.RATE || 50);
if (!Number.isInteger(rate) || rate < 1) throw new Error('RATE must be a positive integer');
const runId = __ENV.RUN_ID || String(Date.now());
const successes = new Counter('successful_classifications');
const errors = new Rate('classification_errors');
const latency = new Trend('classification_latency_ms', true);
// Explicit substitute distribution: three short English comments, equally weighted.
const samples = ['Thanks for sharing this helpful explanation.', 'I disagree with the argument and its conclusions.', 'You are an idiot and your opinions are stupid.'];
export const options = {
  scenarios: {classification: {executor:'constant-arrival-rate',rate,timeUnit:'1s',duration:__ENV.DURATION || '180s',preAllocatedVUs:32,maxVUs:100}},
  summaryTrendStats: ['avg','med','p(95)','p(99)','max'],
  thresholds: {classification_latency_ms:['p(95)<100'],classification_errors:['rate==0'],dropped_iterations:['count==0']},
};
export function setup() {
  if (http.get(`${base}/v1/health/ready`).status !== 200) throw new Error('API not ready');
}
export default function () {
  const i = exec.scenario.iterationInTest;
  const text = `${samples[i % samples.length]} [synthetic ${runId}/${i}]`;
  const response = http.post(`${base}/v1/classify`, JSON.stringify({text}), {headers:{'Content-Type':'application/json'},timeout:'30s'});
  let ok = response.status === 200;
  if (ok) {try {ok = Boolean(response.json().classification_id)} catch (_) {ok = false}}
  errors.add(!ok);
  if (ok) {successes.add(1);latency.add(response.timings.duration)}
}
