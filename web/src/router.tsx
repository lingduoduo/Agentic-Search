import { useCallback, useEffect, useSyncExternalStore } from "react";
import type { MouseEvent, ReactNode } from "react";

export const ROUTES = ["/assist", "/search", "/chat", "/tools"] as const;

export type Route = (typeof ROUTES)[number];

export const DEFAULT_ROUTE: Route = "/assist";

/**
 * Resolve any pathname onto one of the four pages. The navigation is a closed
 * set, so an unknown path is a typo rather than a 404 worth rendering.
 */
export function normalizeRoute(pathname: string): Route {
  const trimmed = pathname.replace(/\/+$/, "");
  return (ROUTES as readonly string[]).includes(trimmed) ? (trimmed as Route) : DEFAULT_ROUTE;
}

function subscribe(onStoreChange: () => void): () => void {
  window.addEventListener("popstate", onStoreChange);
  return () => window.removeEventListener("popstate", onStoreChange);
}

function getSnapshot(): Route {
  return normalizeRoute(window.location.pathname);
}

/** The route for the current URL, re-read whenever the history entry changes. */
export function useRoute(): Route {
  return useSyncExternalStore(subscribe, getSnapshot, () => DEFAULT_ROUTE);
}

/**
 * `useRoute`, plus an address-bar rewrite when the URL is not already the
 * resolved route (`/` and unknown paths). `replaceState` keeps Back pointing at
 * whatever preceded the app instead of looping on the redirect.
 */
export function useCanonicalRoute(): Route {
  const route = useRoute();
  useEffect(() => {
    if (window.location.pathname !== route) {
      window.history.replaceState({}, "", route);
    }
  }, [route]);
  return route;
}

/** pushState does not notify listeners, so publish the change ourselves. */
export function navigate(to: Route): void {
  if (window.location.pathname === to) return;
  window.history.pushState({}, "", to);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export function NavLink({
  to,
  className,
  children,
}: {
  to: Route;
  className?: string;
  children: ReactNode;
}) {
  const active = useRoute() === to;

  const handleClick = useCallback(
    (event: MouseEvent<HTMLAnchorElement>) => {
      // Anything but a plain left click means the user asked the browser for
      // something we cannot do: a new tab, a new window, a download.
      if (event.defaultPrevented || event.button !== 0) return;
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      navigate(to);
    },
    [to],
  );

  return (
    <a
      href={to}
      className={`${className ?? ""}${active ? " active" : ""}`.trim()}
      aria-current={active ? "page" : undefined}
      onClick={handleClick}
    >
      {children}
    </a>
  );
}
