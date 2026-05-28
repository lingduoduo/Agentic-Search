import type {
  AgentExperienceRequest,
  AgentExperienceResponse,
  ChatSessionView,
  SessionCreateRequest,
} from "./types";

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
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
): Promise<ChatSessionView> {
  return requestJson<ChatSessionView>("/api/sessions", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function getSession(sessionId: string): Promise<ChatSessionView> {
  return requestJson<ChatSessionView>(`/api/sessions/${sessionId}`);
}

export function runAgent(
  request: AgentExperienceRequest,
): Promise<AgentExperienceResponse> {
  return requestJson<AgentExperienceResponse>("/api/agent", {
    method: "POST",
    body: JSON.stringify(request),
  });
}
