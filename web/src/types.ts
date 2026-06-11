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

export type AgentMode = "search_tool" | "hybrid_search" | "chat_once" | "chat_loop";
export type SearchSourceProvider =
  | "retrieval"
  | "google"
  | "serpapi"
  | "browser"
  | "all";

export interface AgentExperienceRequest {
  query: string;
  session_id?: string | null;
  user_id?: string | null;
  search_url?: string | null;
  top_k?: number;
  source_provider?: SearchSourceProvider;
  mode?: AgentMode;
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

export interface QueryHistoryItem {
  id: string;
  user_id: string | null;
  name: string | null;
  first_user_message: string;
  first_ai_message: string;
  time_created: string;
  feedback_type: "like" | "dislike" | "mixed" | null;
  flow_type: "chat" | "slack";
  conversation_length: number;
  llm_name: string | null;
}

export interface QueryHistoryPage {
  items: QueryHistoryItem[];
  total_items: number;
}

export interface AuditSummary {
  total_sessions: number;
  total_messages: number;
  sessions_with_feedback: number;
  sessions_with_dislike: number;
  dislike_rate: number;
  top_disliked_queries: string[];
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

export interface BreakdownItem {
  label: string;
  session_count: number;
}

export interface BreakdownAnalytics {
  dimension: string;
  items: BreakdownItem[];
  total_sessions: number;
}
