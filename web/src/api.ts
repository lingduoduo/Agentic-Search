import type {
  AdminSurfaceSummary,
  AgentExperienceRequest,
  AgentExperienceResponse,
  AuditSummary,
  BreakdownAnalytics,
  ChatSessionView,
  ConnectorCreateRequest,
  ConnectorDetailView,
  ConnectorView,
  QueryHistoryPage,
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

export function getQueryHistory(
  params: {
    page_num?: number;
    page_size?: number;
    start_time?: string;
    end_time?: string;
    user_id?: string;
    feedback_type?: "like" | "dislike";
  } = {},
  init?: Pick<RequestInit, "signal">,
): Promise<QueryHistoryPage> {
  const qs = new URLSearchParams();
  if (params.page_num !== undefined) qs.set("page_num", String(params.page_num));
  if (params.page_size !== undefined) qs.set("page_size", String(params.page_size));
  if (params.start_time) qs.set("start_time", params.start_time);
  if (params.end_time) qs.set("end_time", params.end_time);
  if (params.user_id) qs.set("user_id", params.user_id);
  if (params.feedback_type) qs.set("feedback_type", params.feedback_type);
  const query = qs.toString();
  return requestJson<QueryHistoryPage>(
    `/admin/chat-session-history${query ? `?${query}` : ""}`,
    { signal: init?.signal },
  );
}

export function getAuditSummary(
  params: { start_time?: string; end_time?: string } = {},
  init?: Pick<RequestInit, "signal">,
): Promise<AuditSummary> {
  const qs = new URLSearchParams();
  if (params.start_time) qs.set("start_time", params.start_time);
  if (params.end_time) qs.set("end_time", params.end_time);
  const query = qs.toString();
  return requestJson<AuditSummary>(
    `/admin/query-history/audit${query ? `?${query}` : ""}`,
    { signal: init?.signal },
  );
}

export function listConnectors(
  params: { enabled?: boolean } = {},
  init?: Pick<RequestInit, "signal">,
): Promise<ConnectorView[]> {
  const qs = new URLSearchParams();
  if (params.enabled !== undefined) qs.set("enabled", String(params.enabled));
  const query = qs.toString();
  return requestJson<ConnectorView[]>(
    `/admin/connectors${query ? `?${query}` : ""}`,
    { signal: init?.signal },
  );
}

export function createConnector(
  req: ConnectorCreateRequest,
  init?: Pick<RequestInit, "signal">,
): Promise<ConnectorView> {
  return requestJson<ConnectorView>("/admin/connectors", {
    method: "POST",
    body: JSON.stringify(req),
    signal: init?.signal,
  });
}

export function getConnector(
  connectorId: string,
  init?: Pick<RequestInit, "signal">,
): Promise<ConnectorDetailView> {
  return requestJson<ConnectorDetailView>(`/admin/connectors/${connectorId}`, {
    signal: init?.signal,
  });
}

export function updateConnector(
  connectorId: string,
  patch: { name?: string; config?: Record<string, unknown>; enabled?: boolean },
  init?: Pick<RequestInit, "signal">,
): Promise<ConnectorView> {
  return requestJson<ConnectorView>(`/admin/connectors/${connectorId}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
    signal: init?.signal,
  });
}

export function deleteConnector(
  connectorId: string,
  init?: Pick<RequestInit, "signal">,
): Promise<void> {
  return requestJson<void>(`/admin/connectors/${connectorId}`, {
    method: "DELETE",
    signal: init?.signal,
  });
}

export function runConnector(
  connectorId: string,
  init?: Pick<RequestInit, "signal">,
): Promise<{ attempt_id: string; message: string }> {
  return requestJson<{ attempt_id: string; message: string }>(
    `/admin/connectors/${connectorId}/run`,
    { method: "POST", signal: init?.signal },
  );
}

export function submitFeedback(
  messageId: string,
  isPositive: boolean,
  feedbackText?: string,
  init?: Pick<RequestInit, "signal">,
): Promise<void> {
  return requestJson<void>("/chat/create-chat-message-feedback", {
    method: "POST",
    body: JSON.stringify({
      chat_message_id: messageId,
      is_positive: isPositive,
      feedback_text: feedbackText ?? null,
    }),
    signal: init?.signal,
  });
}
