import type {
  Classification,
  Correction,
  Explanation,
  Job,
  Label,
  ModelInfo,
} from "./types";
export const API_BASE = (
  import.meta.env.VITE_API_BASE_URL || "https://flagged.duckdns.org"
).replace(/\/$/, "");
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}
const messages: Record<number, string> = {
  404: "This result could not be found. Classify the text again.",
  409: "This result uses a model version that is no longer available.",
  422: "Please check the text and input limits.",
  429: "The API is at capacity. Wait a moment, then try again.",
  503: "The service or its storage is temporarily unavailable. Please try again.",
  504: "The request took too long. Please try again.",
};
export async function request<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener("abort", abort, { once: true });
  if (signal?.aborted) controller.abort();
  const timer = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(`${API_BASE}/v1${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers:
        body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
      credentials: "omit",
    });
    if (!response.ok)
      throw new ApiError(
        response.status,
        messages[response.status] ||
          "The request could not be completed. Please try again.",
      );
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (signal?.aborted) throw new DOMException("Cancelled", "AbortError");
    if (controller.signal.aborted) throw new ApiError(504, messages[504]);
    throw new ApiError(
      0,
      "Cannot reach the API. Check your connection, or try again when the service is available.",
    );
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
  }
}
export const api = {
  info: (signal?: AbortSignal) =>
    request<ModelInfo>("/model_info", undefined, signal),
  ready: (signal?: AbortSignal) =>
    request<{ is_ready: boolean }>("/health/ready", undefined, signal),
  classify: (text: string, signal?: AbortSignal) =>
    request<Classification>("/classify", { text }, signal),
  batch: (texts: string[], signal?: AbortSignal) =>
    request<{ results: Classification[] }>(
      "/classify_batch",
      { texts },
      signal,
    ),
  explain: (id: string, label: Label, signal?: AbortSignal) =>
    request<Explanation | Job>(
      "/explain",
      { classification_id: id, label },
      signal,
    ),
  job: (id: string, signal?: AbortSignal) =>
    request<Job>(`/jobs/${id}`, undefined, signal),
  correct: (id: string, labels: Record<Label, boolean>, signal?: AbortSignal) =>
    request<Correction>(
      `/classifications/${id}/corrections`,
      { corrected_labels: labels, moderator_id: "public-demo" },
      signal,
    ),
};
export function delay(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer);
      reject(new DOMException("Cancelled", "AbortError"));
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, ms);
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) abort();
  });
}
