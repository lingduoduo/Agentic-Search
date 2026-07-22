package api_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/lingduoduo/Agentic-Search/cli/testutil"
)

func TestSaveMemory_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" || !strings.HasSuffix(r.URL.Path, "/memory/save") {
			t.Errorf("got %s %s, want POST /api/memory/save", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"memory_id": "mem_123"}`))
	}))
	defer srv.Close()

	resp, err := testutil.NewClient(srv.URL).SaveMemory(t.Context(), "likes tea")
	if err != nil {
		t.Fatalf("SaveMemory: %v", err)
	}
	if resp.MemoryID == nil || *resp.MemoryID != "mem_123" {
		t.Errorf("memory_id = %v, want mem_123", resp.MemoryID)
	}
}

func TestListMemories_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || !strings.HasSuffix(r.URL.Path, "/memory/list") {
			t.Errorf("got %s %s, want GET /api/memory/list", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"memories": [{"id": "m1", "text": "hi", "updated_at": null}]}`))
	}))
	defer srv.Close()

	resp, err := testutil.NewClient(srv.URL).ListMemories(t.Context())
	if err != nil {
		t.Fatalf("ListMemories: %v", err)
	}
	if len(resp.Memories) != 1 || resp.Memories[0].ID != "m1" {
		t.Errorf("memories = %+v", resp.Memories)
	}
}

func TestCurateMemory_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/memory/curate") {
			t.Errorf("path = %s, want /api/memory/curate", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok","trajectory_id":"t1","counts":{"add":1},"memory_count":1}`))
	}))
	defer srv.Close()

	resp, err := testutil.NewClient(srv.URL).CurateMemory(t.Context(), nil)
	if err != nil {
		t.Fatalf("CurateMemory: %v", err)
	}
	if resp.Status != "ok" || resp.Counts["add"] != 1 {
		t.Errorf("resp = %+v", resp)
	}
}

func TestCurateMemory_EmptyDecodesMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"empty","message":"no conversations or notes yet","counts":{}}`))
	}))
	defer srv.Close()

	resp, err := testutil.NewClient(srv.URL).CurateMemory(t.Context(), nil)
	if err != nil {
		t.Fatalf("CurateMemory: %v", err)
	}
	if resp.Status != "empty" || resp.Message != "no conversations or notes yet" {
		t.Errorf("resp = %+v", resp)
	}
}
