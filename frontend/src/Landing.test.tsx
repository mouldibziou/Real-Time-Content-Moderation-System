import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { Landing } from "./Landing";
beforeEach(() => {
  vi.spyOn(api, "ready").mockResolvedValue({ is_ready: true });
  vi.stubGlobal("matchMedia", vi.fn().mockReturnValue({ matches: true }));
});
it("keeps the whole story visible when reduced motion is enabled", () => {
  const { container } = render(<Landing />);
  expect(container.querySelector(".reveal-ready")).toBeNull();
  expect(
    screen.getByRole("heading", { name: /Flag harmful content/ }),
  ).toBeInTheDocument();
});
it("switches the illustrated journey and architecture explanation", () => {
  render(<Landing />);
  fireEvent.click(
    screen.getByRole("button", { name: /02.*Explain.*See what contributed/ }),
  );
  expect(screen.getByText("The words behind the flag.")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /03.*monitoring/i }));
  expect(screen.getByText("Every layer, observed.")).toBeInTheDocument();
});
