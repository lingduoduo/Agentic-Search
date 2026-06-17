export type ChatRole = "user" | "assistant" | "system" | string;

export interface ChatMessageView {
  role: ChatRole;
  content: string;
  metadata?: Record<string, unknown>;
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

export type AgentMode = "search_tool" | "hybrid_search" | "chat_once" | "chat_loop" | "search_agent" | "tool_agent";
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
  intent?: "search" | "chat" | "tool";
}

export interface SessionCreateRequest {
  title?: string | null;
  user_id?: string | null;
}

export type IndexAttemptStatus = "not_started" | "in_progress" | "success" | "failed";

export interface IndexAttemptView {
  id: string;
  status: IndexAttemptStatus;
  total_documents: number;
  total_chunks: number;
  error: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface ConnectorView {
  id: string;
  name: string;
  source: string;
  config: Record<string, unknown>;
  enabled: boolean;
  metadata: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
  last_attempt: IndexAttemptView | null;
}

export interface ConnectorDetailView extends ConnectorView {
  attempts: IndexAttemptView[];
  document_count: number;
}

export interface ConnectorCreateRequest {
  name: string;
  source: string;
  config: Record<string, unknown>;
  enabled?: boolean;
  metadata?: Record<string, unknown>;
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
  health_label: string;
  health_score: number;
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

export interface ToolView {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  source: "function" | "openapi" | string;
  provider_id: string | null;
}

export interface OpenAPIRegisterRequest {
  name: string;
  openapi_json: string;
  headers?: Record<string, string>;
}

export interface OpenAPIRegisterResponse {
  provider_id: string;
  tool_names: string[];
}

export interface ToolInvokeRequest {
  arguments: Record<string, unknown>;
}

export interface ToolInvokeResponse {
  response: string;
  raw: unknown;
  errors: string[];
}

// ---------------------------------------------------------------------------
// SSE streaming types
// ---------------------------------------------------------------------------

export interface SSEProgressEvent {
  type: "progress";
  text: string;
  turn: number;
}

export interface SSEAnswerEvent {
  type: "answer";
  text: string;
}

export interface SSEDoneEvent {
  type: "done";
  session_id: string;
  citations: string[];
  documents: SourceDocumentView[];
  intent?: "search" | "chat" | "tool";
}

export interface SSEErrorEvent {
  type: "error";
  detail: string;
}

export type SSEEvent =
  | SSEProgressEvent
  | SSEAnswerEvent
  | SSEDoneEvent
  | SSEErrorEvent;

export interface ProgressStep {
  turn: number;
  text: string;  // e.g. "search_routing_tool · 5 docs" or "writing answer..."
}
