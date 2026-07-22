package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
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
