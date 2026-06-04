package render_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/lingduoduo/Agentic-Search/cli/models"
	"github.com/lingduoduo/Agentic-Search/cli/render"
)

func strPtr(s string) *string { return &s }

func TestSources_PrintsTable(t *testing.T) {
	docs := []models.AgentDocument{
		{Citation: "[1]", Title: "Q3 Report", URL: strPtr("https://internal.corp/q3"), Content: "x"},
		{Citation: "[2]", Title: "", URL: nil, Content: "y"},
	}
	var buf bytes.Buffer
	render.Sources(&buf, docs)
	out := buf.String()
	if !strings.Contains(out, "Q3 Report") {
		t.Errorf("expected 'Q3 Report' in output, got:\n%s", out)
	}
	if !strings.Contains(out, "internal.corp") {
		t.Errorf("expected URL in output, got:\n%s", out)
	}
	if !strings.Contains(out, "[1]") {
		t.Errorf("expected '[1]' in output, got:\n%s", out)
	}
}

func TestSources_EmptyNoOutput(t *testing.T) {
	var buf bytes.Buffer
	render.Sources(&buf, nil)
	if buf.Len() != 0 {
		t.Errorf("expected empty output for nil docs, got: %q", buf.String())
	}
}

func TestProgressive_ContainsAllWords(t *testing.T) {
	var buf bytes.Buffer
	render.Progressive(&buf, "Hello world foo bar", 10000.0, 100)
	out := buf.String()
	for _, w := range []string{"Hello", "world", "foo", "bar"} {
		if !strings.Contains(out, w) {
			t.Errorf("expected %q in output, got:\n%s", w, out)
		}
	}
}

func TestProgressive_EmptyNoOutput(t *testing.T) {
	var buf bytes.Buffer
	render.Progressive(&buf, "", 10000.0, 100)
	if buf.Len() != 0 {
		t.Errorf("expected empty output for empty answer, got: %q", buf.String())
	}
}
