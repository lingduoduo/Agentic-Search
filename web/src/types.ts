export type ChatRole = "user" | "assistant" | "system" | string;

export interface ChatMessageView {
  role: ChatRole;
  content: string;
}

export interface ChatSessionView {
  id: string;
  title: string | null;
  user_id: string | null;
  messages: ChatMessageView[];
}

export interface SourceDocumentView {
  id: string;
  citation: string;
  title: string;
  content: string;
  url: string | null;
  score: number;
  metadata: Record<string, unknown>;
}

export interface AgentExperienceRequest {
  query: string;
  session_id?: string | null;
  user_id?: string | null;
  search_url?: string | null;
  top_k?: number;
}

export interface AgentExperienceResponse {
  session_id: string;
  answer: string;
  citations: string[];
  documents: SourceDocumentView[];
  messages: ChatMessageView[];
}

export interface SessionCreateRequest {
  title?: string | null;
  user_id?: string | null;
}

export type AdminSurfaceKey =
  | "connectors"
  | "indexing"
  | "access"
  | "auth"
  | "models"
  | "tools"
  | "analytics"
  | "enterprise";

export interface AdminSurfaceMetric {
  label: string;
  value: string;
  detail: string;
}

export interface AdminSurfaceSection {
  key: AdminSurfaceKey;
  title: string;
  status: string;
  tone: "good" | "watch" | "neutral";
  description: string;
  items: string[];
}

export interface AdminSurfaceSummary {
  healthLabel: string;
  healthScore: number;
  metrics: AdminSurfaceMetric[];
  sections: AdminSurfaceSection[];
}
