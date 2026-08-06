// Package models defines API request/response types for the Agentic Search CLI.
package models

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
