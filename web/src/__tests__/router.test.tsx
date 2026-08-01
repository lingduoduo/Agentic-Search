import { render, renderHook, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, beforeEach } from "vitest";
import { NavLink, navigate, normalizeRoute, useCanonicalRoute, useRoute } from "../router";

beforeEach(() => {
  window.history.replaceState({}, "", "/");
});

describe("normalizeRoute", () => {
  it("maps / to the default route", () => {
    expect(normalizeRoute("/")).toBe("/assist");
  });

  it("keeps a known route", () => {
    expect(normalizeRoute("/chat")).toBe("/chat");
  });

  it("strips a trailing slash", () => {
    expect(normalizeRoute("/chat/")).toBe("/chat");
  });

  it("falls back to the default for an unknown path", () => {
    expect(normalizeRoute("/nope")).toBe("/assist");
  });
});

describe("useRoute", () => {
  it("reports the route for the current path", () => {
    window.history.replaceState({}, "", "/tools");
    const { result } = renderHook(() => useRoute());
    expect(result.current).toBe("/tools");
  });

  it("updates when navigate() is called", () => {
    const { result } = renderHook(() => useRoute());
    act(() => navigate("/search"));
    expect(result.current).toBe("/search");
    expect(window.location.pathname).toBe("/search");
  });

  // Back/forward reach the app as a popstate event; this is what the browser does.
  it("updates when the browser fires popstate", () => {
    const { result } = renderHook(() => useRoute());
    act(() => {
      window.history.replaceState({}, "", "/chat");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(result.current).toBe("/chat");
  });
});

describe("useCanonicalRoute", () => {
  it("rewrites / to the default route without adding a history entry", () => {
    const before = window.history.length;
    const { result } = renderHook(() => useCanonicalRoute());
    expect(result.current).toBe("/assist");
    expect(window.location.pathname).toBe("/assist");
    expect(window.history.length).toBe(before);
  });

  it("leaves a known route alone", () => {
    window.history.replaceState({}, "", "/tools");
    const { result } = renderHook(() => useCanonicalRoute());
    expect(result.current).toBe("/tools");
    expect(window.location.pathname).toBe("/tools");
  });
});

describe("NavLink", () => {
  it("renders a real href so copy-link and new-tab work", () => {
    render(<NavLink to="/chat">Chat</NavLink>);
    expect(screen.getByRole("link", { name: "Chat" })).toHaveAttribute("href", "/chat");
  });

  it("marks the active route with aria-current", () => {
    window.history.replaceState({}, "", "/chat");
    render(
      <>
        <NavLink to="/chat">Chat</NavLink>
        <NavLink to="/tools">Tools</NavLink>
      </>,
    );
    expect(screen.getByRole("link", { name: "Chat" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Tools" })).not.toHaveAttribute("aria-current");
  });

  it("navigates on a plain left click", async () => {
    render(<NavLink to="/tools">Tools</NavLink>);
    await userEvent.click(screen.getByRole("link", { name: "Tools" }));
    expect(window.location.pathname).toBe("/tools");
  });

  // A modified click means "new tab" — the browser owns it, we must not hijack it.
  // The direct userEvent API drops held modifiers between calls, so this case
  // needs a session from userEvent.setup() to keep Meta down across the click.
  it("leaves a cmd-click to the browser", async () => {
    const user = userEvent.setup();
    render(<NavLink to="/tools">Tools</NavLink>);
    await user.keyboard("{Meta>}");
    await user.click(screen.getByRole("link", { name: "Tools" }));
    await user.keyboard("{/Meta}");
    expect(window.location.pathname).toBe("/");
  });
});
