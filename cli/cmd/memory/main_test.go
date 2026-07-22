package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/lingduoduo/Agentic-Search/cli/exitcodes"
	"github.com/lingduoduo/Agentic-Search/cli/models"
)

func TestRunUnknownSubcommand(t *testing.T) {
	if code := run([]string{"bogus"}); code == 0 {
		t.Fatalf("unknown subcommand should exit non-zero")
	}
}

func TestRunNoArgsShowsUsage(t *testing.T) {
	if code := run(nil); code == 0 {
		t.Fatalf("no subcommand should exit non-zero")
	}
}

func TestRunAddHitsSaveEndpoint(t *testing.T) {
	var gotPath, gotMethod string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath, gotMethod = r.URL.Path, r.Method
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"memory_id":"mem_1"}`))
	}))
	defer srv.Close()
	t.Setenv("AGENTIC_SEARCH_URL", srv.URL)

	if code := run([]string{"add", "likes", "tea"}); code != 0 {
		t.Fatalf("add exited %d", code)
	}
	if gotMethod != "POST" || !strings.HasSuffix(gotPath, "/memory/save") {
		t.Errorf("got %s %s, want POST .../memory/save", gotMethod, gotPath)
	}
}

// captureStdout runs fn with os.Stdout redirected to a pipe and returns what it wrote.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w
	fn()
	_ = w.Close()
	os.Stdout = old
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, r); err != nil {
		t.Fatalf("copy: %v", err)
	}
	return buf.String()
}

func TestRunListJSONOutput(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"memories":[{"id":"m1","text":"hi","updated_at":null}]}`))
	}))
	defer srv.Close()
	t.Setenv("AGENTIC_SEARCH_URL", srv.URL)

	var code int
	out := captureStdout(t, func() { code = run([]string{"list", "--json"}) })
	if code != 0 {
		t.Fatalf("list --json exited %d", code)
	}
	var got models.MemoryListResponse
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("--json output is not valid JSON: %v\noutput: %q", err, out)
	}
	if len(got.Memories) != 1 || got.Memories[0].ID != "m1" {
		t.Errorf("decoded memories = %+v", got.Memories)
	}
}

func TestRunExitCodeAuthFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()
	t.Setenv("AGENTIC_SEARCH_URL", srv.URL)

	if code := run([]string{"list"}); code != int(exitcodes.AuthFailure) {
		t.Errorf("exit code = %d, want %d (AuthFailure)", code, int(exitcodes.AuthFailure))
	}
}

func TestRunExitCodeBadRequestOnMissingText(t *testing.T) {
	// No network: the missing-text check fires before any HTTP call.
	if code := run([]string{"add"}); code != int(exitcodes.BadRequest) {
		t.Errorf("exit code = %d, want %d (BadRequest)", code, int(exitcodes.BadRequest))
	}
}
