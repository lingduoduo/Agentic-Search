package models

// Stream events for the NDJSON chat response from POST /api/chat/send-chat-message.
//
// These types are never unmarshalled directly: parser.ParseStreamLine decodes
// each line into a map[string]any and builds the typed event by hand, so there
// are deliberately no JSON tags here. The wire contract lives in
// parser/parser.go's switch on the packet's "type" field.
//
// Reconstruction note: this file was absent from git history — a bare `models/`
// rule in .gitignore (meant for ML checkpoints) silently excluded it, so it
// existed only in one working copy and `cli/` never built from a clone. It was
// rebuilt from its consumers: parser/parser.go's constructors, the assertions in
// parser/parser_test.go, and the field reads in tui/app.go. Every struct field
// and the StreamEvent interface are therefore pinned by compilation plus the
// parser test suite. The EventType() constant *values* are not pinned — nothing
// reads them today — so they mirror the wire "type" strings where one exists.

// StreamEvent is one decoded packet from the chat stream.
type StreamEvent interface {
	// EventType identifies the event, mirroring the wire "type" field where the
	// packet has one.
	EventType() string
}

const (
	EventSessionCreated          = "session_created"
	EventMessageIDInfo           = "message_id_info"
	EventStop                    = "stop"
	EventError                   = "error"
	EventMessageStart            = "message_start"
	EventMessageDelta            = "message_delta"
	EventSearchStart             = "search_tool_start"
	EventSearchQueries           = "search_tool_queries_delta"
	EventSearchDocuments         = "search_tool_documents_delta"
	EventReasoningStart          = "reasoning_start"
	EventReasoningDelta          = "reasoning_delta"
	EventReasoningDone           = "reasoning_done"
	EventCitationInfo            = "citation_info"
	EventCustomToolStart         = "custom_tool_start"
	EventDeepResearchPlanStart   = "deep_research_plan_start"
	EventDeepResearchPlanDelta   = "deep_research_plan_delta"
	EventResearchAgentStart      = "research_agent_start"
	EventIntermediateReportStart = "intermediate_report_start"
	EventIntermediateReportDelta = "intermediate_report_delta"
	EventUnknown                 = "unknown"
)

// SessionCreatedEvent carries the session id the backend allocated for a chat
// started without one.
type SessionCreatedEvent struct {
	ChatSessionID string
}

func (e SessionCreatedEvent) EventType() string { return EventSessionCreated }

// MessageIDEvent reports the ids assigned to this turn. ReservedAgentMessageID
// becomes the next turn's parent message id.
type MessageIDEvent struct {
	UserMessageID          *int
	ReservedAgentMessageID int
}

func (e MessageIDEvent) EventType() string { return EventMessageIDInfo }

// StopEvent ends a turn. StopReason is nil when the backend did not send one.
type StopEvent struct {
	Placement  *Placement
	StopReason *string
}

func (e StopEvent) EventType() string { return EventStop }

// ErrorEvent reports a stream failure. It is also synthesized client-side for
// transport and decode errors, in which case Placement is nil. IsRetryable is
// false for those synthesized cases where retrying cannot help (a marshal
// failure, malformed stream data). StatusCode is 0 unless the failure came from
// an HTTP response, which only the client-side synthesized cases carry.
type ErrorEvent struct {
	Placement   *Placement
	Error       string
	StackTrace  *string
	IsRetryable bool
	StatusCode  int
}

func (e ErrorEvent) EventType() string { return EventError }

// MessageStartEvent opens the answer. Documents holds the final cited set when
// the backend sends it up front.
type MessageStartEvent struct {
	Placement *Placement
	Documents []SearchDoc
}

func (e MessageStartEvent) EventType() string { return EventMessageStart }

// MessageDeltaEvent is one token chunk of the answer.
type MessageDeltaEvent struct {
	Placement *Placement
	Content   string
}

func (e MessageDeltaEvent) EventType() string { return EventMessageDelta }

// SearchStartEvent announces a retrieval step.
type SearchStartEvent struct {
	Placement        *Placement
	IsInternetSearch bool
}

func (e SearchStartEvent) EventType() string { return EventSearchStart }

// SearchQueriesEvent lists the queries the agent chose to run.
type SearchQueriesEvent struct {
	Placement *Placement
	Queries   []string
}

func (e SearchQueriesEvent) EventType() string { return EventSearchQueries }

// SearchDocumentsEvent lists the documents a retrieval step returned.
type SearchDocumentsEvent struct {
	Placement *Placement
	Documents []SearchDoc
}

func (e SearchDocumentsEvent) EventType() string { return EventSearchDocuments }

// ReasoningStartEvent opens a reasoning block.
type ReasoningStartEvent struct {
	Placement *Placement
}

func (e ReasoningStartEvent) EventType() string { return EventReasoningStart }

// ReasoningDeltaEvent is one chunk of reasoning text.
type ReasoningDeltaEvent struct {
	Placement *Placement
	Reasoning string
}

func (e ReasoningDeltaEvent) EventType() string { return EventReasoningDelta }

// ReasoningDoneEvent closes a reasoning block.
type ReasoningDoneEvent struct {
	Placement *Placement
}

func (e ReasoningDoneEvent) EventType() string { return EventReasoningDone }

// CitationEvent maps a citation number in the answer to a document id.
type CitationEvent struct {
	Placement      *Placement
	CitationNumber int
	DocumentID     string
}

func (e CitationEvent) EventType() string { return EventCitationInfo }

// ToolStartEvent announces a tool invocation. Type is the wire event name, so
// one struct covers open_url, image generation, python, file reader, and custom
// tools; ToolName is the display label.
type ToolStartEvent struct {
	Placement *Placement
	Type      string
	ToolName  string
}

func (e ToolStartEvent) EventType() string { return e.Type }

// DeepResearchPlanStartEvent opens the deep-research plan.
type DeepResearchPlanStartEvent struct {
	Placement *Placement
}

func (e DeepResearchPlanStartEvent) EventType() string { return EventDeepResearchPlanStart }

// DeepResearchPlanDeltaEvent is one chunk of the deep-research plan.
type DeepResearchPlanDeltaEvent struct {
	Placement *Placement
	Content   string
}

func (e DeepResearchPlanDeltaEvent) EventType() string { return EventDeepResearchPlanDelta }

// ResearchAgentStartEvent announces a sub-agent starting a research task.
type ResearchAgentStartEvent struct {
	Placement    *Placement
	ResearchTask string
}

func (e ResearchAgentStartEvent) EventType() string { return EventResearchAgentStart }

// IntermediateReportStartEvent opens an intermediate research report.
type IntermediateReportStartEvent struct {
	Placement *Placement
}

func (e IntermediateReportStartEvent) EventType() string { return EventIntermediateReportStart }

// IntermediateReportDeltaEvent is one chunk of an intermediate research report.
type IntermediateReportDeltaEvent struct {
	Placement *Placement
	Content   string
}

func (e IntermediateReportDeltaEvent) EventType() string { return EventIntermediateReportDelta }

// UnknownEvent preserves a packet this CLI does not model, so a newer backend
// streaming unfamiliar events degrades quietly instead of failing.
type UnknownEvent struct {
	Placement *Placement
	RawData   map[string]any
}

func (e UnknownEvent) EventType() string { return EventUnknown }
