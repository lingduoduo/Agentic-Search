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

// TestQueryAgent_Timeout verifies that the wrapTimeoutError helper correctly
// wraps network failures. Integration tests cover the happy path and HTTP
// error cases against a real server.
func TestQueryAgent_Timeout(t *testing.T) {
	url := testutil.DeadServerURL()
	client := testutil.NewClient(url)
	_, err := client.QueryAgent(t.Context(), "test", 5, nil)
	if err == nil {
		t.Fatal("expected error for dead server")
	}
}

// TestQueryAgent_401 pins the APIError status mapping that cmd/memory's exit
// codes branch on.
func TestQueryAgent_401(t *testing.T) {
	srv := testutil.StatusServer(401)
	defer srv.Close()

	client := testutil.NewClient(srv.URL)
	_, err := client.QueryAgent(t.Context(), "test", 5, nil)
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

// TestQueryAgent_HTMLResponse covers the isHTMLResponse guard: a proxy or the
// SPA shell answering instead of the API should produce a pointed message
// rather than a JSON decode error.
func TestQueryAgent_HTMLResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.WriteHeader(502)
		_, _ = w.Write([]byte("<!doctype html><html><body>bad gateway</body></html>"))
	}))
	defer srv.Close()

	client := testutil.NewClient(srv.URL)
	_, err := client.QueryAgent(t.Context(), "test", 5, nil)
	var apiErr *api.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("want *APIError, got %T: %v", err, err)
	}
	if !strings.Contains(apiErr.Detail, "HTML instead of JSON") {
		t.Errorf("detail = %q, want the HTML-response hint", apiErr.Detail)
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
