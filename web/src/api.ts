import type {
  AdminSurfaceSummary,
  AgentExperienceRequest,
  AgentExperienceResponse,
  BreakdownAnalytics,
  ChatSessionView,
  SessionCreateRequest,
} from "./types";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      data && typeof data === "object" && "detail" in data
        ? String(data.detail)
        : `Request failed with ${response.status}`;
    throw new Error(detail);
  }
  return data as T;
}

export function createSession(
  request: SessionCreateRequest = {},
  init?: Pick<RequestInit, "signal">,
): Promise<ChatSessionView> {
  return requestJson<ChatSessionView>("/api/sessions", {
    method: "POST",
    body: JSON.stringify(request),
    signal: init?.signal,
  });
}

export function getSession(
  sessionId: string,
  init?: Pick<RequestInit, "signal">,
): Promise<ChatSessionView> {
  return requestJson<ChatSessionView>(`/api/sessions/${sessionId}`, {
    signal: init?.signal,
  });
}

export function getAdminSummary(
  init?: Pick<RequestInit, "signal">,
): Promise<AdminSurfaceSummary> {
  return requestJson<AdminSurfaceSummary>("/admin/observability/summary", {
    signal: init?.signal,
  });
}

export function getAnalyticsByLLM(
  init?: Pick<RequestInit, "signal">,
): Promise<BreakdownAnalytics> {
  return requestJson<BreakdownAnalytics>("/analytics/by-llm", {
    signal: init?.signal,
  });
}

export function getAnalyticsByPersona(
  init?: Pick<RequestInit, "signal">,
): Promise<BreakdownAnalytics> {
  return requestJson<BreakdownAnalytics>("/analytics/by-persona", {
    signal: init?.signal,
  });
}

export function getAnalyticsByFlow(
  init?: Pick<RequestInit, "signal">,
): Promise<BreakdownAnalytics> {
  return requestJson<BreakdownAnalytics>("/analytics/by-flow", {
    signal: init?.signal,
  });
}

export function runAgent(
  request: AgentExperienceRequest,
  init?: Pick<RequestInit, "signal">,
): Promise<AgentExperienceResponse> {
  return requestJson<AgentExperienceResponse>("/api/agent", {
    method: "POST",
    body: JSON.stringify(request),
    signal: init?.signal,
  });
}
