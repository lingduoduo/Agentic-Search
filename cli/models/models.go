// Package models defines API request/response types for the Agentic Search CLI.
package models

// AgentSummary represents an agent from the API.
type AgentSummary struct {
	ID               int    `json:"id"`
	Name             string `json:"name"`
	Description      string `json:"description"`
	IsDefaultPersona bool   `json:"is_default_persona"`
	IsVisible        bool   `json:"is_listed"`
}

// ChatSessionDetails is a session with timestamps as strings.
type ChatSessionDetails struct {
	ID      string  `json:"id"`
	Name    *string `json:"name"`
	AgentID *int    `json:"persona_id"`
	Created string  `json:"time_created"`
	Updated string  `json:"time_updated"`
}

// ChatMessageDetail is a single message in a session.
type ChatMessageDetail struct {
	MessageID          int     `json:"message_id"`
	ParentMessage      *int    `json:"parent_message"`
	LatestChildMessage *int    `json:"latest_child_message"`
	Message            string  `json:"message"`
	MessageType        string  `json:"message_type"`
	TimeSent           string  `json:"time_sent"`
	Error              *string `json:"error"`
}

// ChatSessionDetailResponse is the full session detail from the API.
type ChatSessionDetailResponse struct {
	ChatSessionID string              `json:"chat_session_id"`
	Description   *string             `json:"description"`
	AgentID       *int                `json:"persona_id"`
	AgentName     *string             `json:"persona_name"`
	Messages      []ChatMessageDetail `json:"messages"`
}

// ChatFileType represents a file type for uploads.
type ChatFileType string

const (
	ChatFileImage     ChatFileType = "image"
	ChatFileDoc       ChatFileType = "document"
	ChatFilePlainText ChatFileType = "plain_text"
	ChatFileCSV       ChatFileType = "csv"
)

// FileDescriptorPayload is a file descriptor for send-message requests.
type FileDescriptorPayload struct {
	ID   string       `json:"id"`
	Type ChatFileType `json:"type"`
	Name string       `json:"name,omitempty"`
}

// UserFileSnapshot represents an uploaded file.
type UserFileSnapshot struct {
	ID           string       `json:"id"`
	Name         string       `json:"name"`
	FileID       string       `json:"file_id"`
	ChatFileType ChatFileType `json:"chat_file_type"`
}

// CategorizedFilesSnapshot is the response from file upload.
type CategorizedFilesSnapshot struct {
	UserFiles []UserFileSnapshot `json:"user_files"`
}

// ChatSessionCreationInfo is included when creating a new session inline.
type ChatSessionCreationInfo struct {
	AgentID int `json:"persona_id"`
}

// SendMessagePayload is the request body for POST /api/chat/send-chat-message.
type SendMessagePayload struct {
	Message          string                   `json:"message"`
	ChatSessionID    *string                  `json:"chat_session_id,omitempty"`
	ChatSessionInfo  *ChatSessionCreationInfo `json:"chat_session_info,omitempty"`
	ParentMessageID  *int                     `json:"parent_message_id"`
	FileDescriptors  []FileDescriptorPayload  `json:"file_descriptors"`
	Origin           string                   `json:"origin"`
	IncludeCitations bool                     `json:"include_citations"`
	Stream           bool                     `json:"stream"`
}

// SearchDoc represents a document found during search.
type SearchDoc struct {
	DocumentID         string  `json:"document_id"`
	SemanticIdentifier string  `json:"semantic_identifier"`
	Link               *string `json:"link"`
	SourceType         string  `json:"source_type"`
}

// Placement indicates where a stream event belongs in the conversation.
type Placement struct {
	TurnIndex    int  `json:"turn_index"`
	TabIndex     int  `json:"tab_index"`
	SubTurnIndex *int `json:"sub_turn_index"`
}

// SearchRequest is the request body for POST /api/search.
type SearchRequest struct {
	Query        string   `json:"query"`
	Sources      []string `json:"sources,omitempty"`
	DocumentSets []string `json:"document_sets,omitempty"`
	// TimeCutoff is an ISO 8601 timestamp. Only documents updated on or after
	// this moment are returned; naive (timezone-less) values are treated as
	// UTC server-side.
	TimeCutoff         *string `json:"time_cutoff,omitempty"`
	PersonaID          *int    `json:"persona_id,omitempty"`
	SkipQueryExpansion bool    `json:"skip_query_expansion,omitempty"`
}

// SearchResult is a single document result from the search API.
//
// Content is the full chunk text the LLM saw for this section. Multiple
// results may share a CitationID when the LLM selected multiple
// non-overlapping sections of the same document.
type SearchResult struct {
	CitationID *int    `json:"citation_id"`
	Title      string  `json:"title"`
	Content    string  `json:"content"`
	Link       *string `json:"link"`
	SourceType string  `json:"source_type"`
	UpdatedAt  *string `json:"updated_at"`
}

// SearchResponse is the response from POST /api/search.
type SearchResponse struct {
	Results []SearchResult `json:"results"`
}

// AgentRequest is the request body for POST /api/agent.
type AgentRequest struct {
	Query     string  `json:"query"`
	TopK      int     `json:"top_k"`
	SessionID *string `json:"session_id,omitempty"`
}

// AgentDocument is a single source document from the /api/agent response.
type AgentDocument struct {
	ID       string  `json:"id"`
	Citation string  `json:"citation"`
	Title    string  `json:"title"`
	Content  string  `json:"content"`
	URL      *string `json:"url,omitempty"`
	Score    float64 `json:"score"`
}

// AgentResult is the response from POST /api/agent.
type AgentResult struct {
	SessionID string          `json:"session_id"`
	Answer    string          `json:"answer"`
	Citations []string        `json:"citations"`
	Documents []AgentDocument `json:"documents"`
}

// --- Memory ---

// MemorySaveRequest is the body for POST /api/memory/save.
type MemorySaveRequest struct {
	Text string `json:"text"`
}

// MemorySaveResponse is the response from POST /api/memory/save.
type MemorySaveResponse struct {
	MemoryID *string `json:"memory_id"`
}

// MemoryRecord is one stored memory.
type MemoryRecord struct {
	ID        string  `json:"id"`
	Text      string  `json:"text"`
	UpdatedAt *string `json:"updated_at"`
}

// MemoryListResponse is the response from GET /api/memory/list.
type MemoryListResponse struct {
	Memories []MemoryRecord `json:"memories"`
}

// MemorySearchRequest is the body for POST /api/memory/search.
type MemorySearchRequest struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results,omitempty"`
}

// MemorySearchHit is one search result.
type MemorySearchHit struct {
	ID    string  `json:"id"`
	Text  string  `json:"text"`
	Score float64 `json:"score"`
}

// MemorySearchResponse is the response from POST /api/memory/search.
type MemorySearchResponse struct {
	Results []MemorySearchHit `json:"results"`
}

// MemoryConsolidateRequest is the body for POST /api/memory/consolidate.
type MemoryConsolidateRequest struct {
	ResolveConflicts bool `json:"resolve_conflicts"`
}

// MemoryConflict records one resolved attribute conflict.
type MemoryConflict struct {
	Attribute  string   `json:"attribute"`
	Kept       string   `json:"kept"`
	Superseded []string `json:"superseded"`
}

// MemoryConsolidateReport is the consolidate report.
type MemoryConsolidateReport struct {
	Initial           int              `json:"initial"`
	DuplicatesRemoved int              `json:"duplicates_removed"`
	ConflictsResolved []MemoryConflict `json:"conflicts_resolved"`
	Final             int              `json:"final"`
}

// MemoryConsolidateResponse is the response from POST /api/memory/consolidate.
type MemoryConsolidateResponse struct {
	Report MemoryConsolidateReport `json:"report"`
}

// MemoryProfileEntry is one profile entry (extra JSON fields are ignored).
type MemoryProfileEntry struct {
	Topic    string `json:"topic"`
	Subtopic string `json:"subtopic"`
	Content  string `json:"content"`
}

// MemoryProfileResponse is the response from the profile endpoints.
type MemoryProfileResponse struct {
	Profile []MemoryProfileEntry `json:"profile"`
}

// MemoryCurateRequest is the body for POST /api/memory/curate.
type MemoryCurateRequest struct {
	SessionID *string `json:"session_id,omitempty"`
}

// MemoryCurateResponse is the response from POST /api/memory/curate.
type MemoryCurateResponse struct {
	Status       string         `json:"status"`
	Message      string         `json:"message,omitempty"`
	TrajectoryID string         `json:"trajectory_id"`
	Counts       map[string]int `json:"counts"`
	MemoryCount  int            `json:"memory_count"`
}
