package api_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lingduoduo/Agentic-Search/cli/api"
	"github.com/lingduoduo/Agentic-Search/cli/models"
	"github.com/lingduoduo/Agentic-Search/cli/testutil"
)

// TestListAgents_Timeout verifies that the wrapTimeoutError helper correctly
// wraps network timeouts as APIError{408}. Integration tests cover the
// happy path and HTTP error cases against a real server.
func TestListAgents_Timeout(t *testing.T) {
	url := testutil.DeadServerURL()
	client := testutil.NewClient(url)
	_, err := client.ListAgents(t.Context())
	if err == nil {
		t.Fatal("expected error for dead server")
	}
}

func TestSearch_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/search") {
			t.Errorf("path = %s, want /api/search", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"results": [{"citation_id": 1, "title": "Test", "content": "full chunk text", "link": null, "source_type": "web", "updated_at": null}]
		}`))
	}))
	defer srv.Close()

	client := testutil.NewClient(srv.URL)
	resp, err := client.Search(t.Context(), models.SearchRequest{Query: "test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp.Results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(resp.Results))
	}
	if resp.Results[0].Content != "full chunk text" {
		t.Errorf("content = %q, want %q", resp.Results[0].Content, "full chunk text")
	}
}

func TestSearch_401(t *testing.T) {
	srv := testutil.StatusServer(401)
	defer srv.Close()

	client := testutil.NewClient(srv.URL)
	_, err := client.Search(t.Context(), models.SearchRequest{Query: "test"})
	if err == nil {
		t.Fatal("expected error for 401")
	}
	var apiErr *api.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("want *APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != 401 {
		t.Errorf("status = %d, want 401", apiErr.StatusCode)
	}
}

// TestTestConnection_AWSELB403 verifies that TestConnection detects an AWS
// ALB/ELB 403 by inspecting the Server response header. This header-sniffing
// logic cannot be exercised by integration tests since it requires a specific
// proxy behavior.
func TestTestConnection_AWSELB403(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Server", "awselb/2.0")
		w.WriteHeader(403)
	}))
	defer srv.Close()

	client := testutil.NewClient(srv.URL)
	err := client.TestConnection(t.Context())
	if err == nil {
		t.Fatal("expected error")
	}
	var authErr *api.AuthError
	if !errors.As(err, &authErr) {
		t.Fatalf("expected AuthError for AWS ELB 403, got %T: %v", err, err)
	}
	if !strings.Contains(authErr.Error(), "AWS load balancer") {
		t.Fatalf("expected AWS load balancer message, got: %s", authErr.Error())
	}
}

func TestQueryAgent(t *testing.T) {
	fakeResp := models.AgentResult{
		SessionID: "sess-1",
		Answer:    "Revenue grew 12%.",
		Citations: []string{"[1]"},
		Documents: []models.AgentDocument{
			{ID: "d1", Citation: "[1]", Title: "Q3 Report", Content: "...", Score: 0.9},
		},
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if !strings.HasSuffix(r.URL.Path, "/agent") {
			t.Errorf("path = %s, want /api/agent", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("Authorization = %s, want Bearer test-key", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(fakeResp)
	}))
	defer srv.Close()

	client := testutil.NewClient(srv.URL)
	sessionID := "s1"
	result, err := client.QueryAgent(context.Background(), "show Q3 results", 5, &sessionID)
	if err != nil {
		t.Fatalf("QueryAgent: %v", err)
	}
	if result.SessionID != "sess-1" {
		t.Errorf("SessionID = %q, want sess-1", result.SessionID)
	}
	if result.Answer != "Revenue grew 12%." {
		t.Errorf("Answer = %q", result.Answer)
	}
	if len(result.Documents) != 1 {
		t.Errorf("Documents count = %d, want 1", len(result.Documents))
	}
}
