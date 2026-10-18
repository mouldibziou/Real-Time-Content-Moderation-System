import { afterEach, expect, it, vi } from "vitest";
import { api, ApiError, request } from "./api";
afterEach(() => vi.unstubAllGlobals());
it.each([422, 429, 503, 504])(
  "reports HTTP %s with an actionable message",
  async (status) => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("{}", { status })),
    );
    await expect(api.classify("test")).rejects.toMatchObject({ status });
  },
);
it("sends all six review labels without credentials", async () => {
  const fetch = vi.fn().mockResolvedValue(new Response("{}"));
  vi.stubGlobal("fetch", fetch);
  await api.correct("id", {
    toxic: true,
    severe_toxic: false,
    obscene: false,
    threat: false,
    insult: false,
    identity_hate: false,
  });
  const options = fetch.mock.calls[0][1];
  expect(options.credentials).toBe("omit");
  expect(JSON.parse(options.body).moderator_id).toBe("public-demo");
});
it("translates transport failure into a service message", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockRejectedValue(new TypeError("Failed to fetch")),
  );
  await expect(request("/model_info")).rejects.toBeInstanceOf(ApiError);
});
