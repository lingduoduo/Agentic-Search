# Go Enterprise Knowledge CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `cli/cmd/query/main.go` — a single static Go binary that reads `AGENTIC_SEARCH_PAT`/`AGENTIC_SEARCH_URL` (or flags), queries `POST /api/agent`, and progressively renders the answer as glamour markdown in the terminal, reusing the existing `cli/` package infrastructure.

**Architecture:** Fix the `cli/` module (no `go.mod` exists; all packages import `github.com/lingduoduo/Agentic-Search/cli/internal/...` but files sit at `cli/{name}/`) by creating `go.mod` and moving packages under `internal/`. Add `AgentResult`/`AgentRequest` to `internal/models`, add `QueryAgent()` to `internal/api/client.go` using the existing `doJSONWith()` pattern, extract the viewport's progressive-glamour pattern into `internal/render/render.go`, and wire it all in `cmd/query/main.go`.

**Tech Stack:** Go 1.22+ (Homebrew), existing `charmbracelet/glamour`, `charmbracelet/lipgloss`, `charmbracelet/bubbletea`, `charmbracelet/bubbles`, `golang-jwt/jwt/v5` (add for token minting), `golang.org/x/term`, stdlib `net/http` + `encoding/json`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `cli/go.mod` | Module `github.com/lingduoduo/Agentic-Search/cli`, pinned deps |
| Create | `cli/go.sum` | Auto-generated |
| Move | `cli/{api,browser,config,exitcodes,fsutil,iostreams,models,overflow,parser,starprompt,testutil,tui,version}/` → `cli/internal/{…}/` | Match existing import paths |
| Modify | `cli/internal/models/models.go` | Add `AgentRequest`, `AgentResult` structs |
| Modify | `cli/internal/api/client.go` | Add `QueryAgent()` method + interface entry |
| Create | `cli/internal/render/render.go` | `Progressive()` — word-by-word glamour renderer extracted from viewport |
| Create | `cli/internal/render/render_test.go` | Unit tests |
| Create | `cli/cmd/query/main.go` | `main()` — flag parsing, config, auth, query, render |

---

### Task 0: Install Go + create module + reorganise packages

**Files:**
- Create: `cli/go.mod`, `cli/go.sum`
- Move: all top-level packages into `internal/`

- [ ] **Step 1: Install Go**

```bash
brew install go && go version
```
Expected: `go version go1.22.x darwin/arm64`

- [ ] **Step 2: Init module inside `cli/`**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli
go mod init github.com/lingduoduo/Agentic-Search/cli
```
Expected: `cli/go.mod` created.

- [ ] **Step 3: Move all packages under `internal/`**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli
mkdir -p internal
for pkg in api browser config exitcodes fsutil iostreams models overflow parser starprompt testutil tui version embedded; do
  [ -d "$pkg" ] && git mv "$pkg" "internal/$pkg"
done
```

- [ ] **Step 4: Fetch all dependencies**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli
go get github.com/charmbracelet/glamour@latest
go get github.com/charmbracelet/lipgloss@latest
go get github.com/charmbracelet/bubbletea@latest
go get github.com/charmbracelet/bubbles@latest
go get github.com/golang-jwt/jwt/v5@latest
go get github.com/sirupsen/logrus@latest
go get golang.org/x/term@latest
go get golang.org/x/text@latest
go mod tidy
```

- [ ] **Step 5: Verify existing packages compile**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli
go build ./internal/...
```
Expected: no errors. If any `internal/tui` packages fail due to missing `overflow` import paths, they will resolve once all packages are under `internal/`.

- [ ] **Step 6: Run existing tests**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli
go test ./internal/... 2>&1 | tail -20
```
Expected: existing tests pass (parser, config, viewport, etc.).

- [ ] **Step 7: Commit**

```bash
cd /Users/linghuang/Git/Agentic-Search
git add cli/go.mod cli/go.sum cli/internal/
git commit -m "chore(go): init cli module and move packages under internal/"
```

---

### Task 1: Add `AgentRequest`/`AgentResult` to models

**Files:**
- Modify: `cli/internal/models/models.go`

- [ ] **Step 1: Add these two structs at the bottom of `cli/internal/models/models.go`**

```go
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
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli && go build ./internal/models/
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
cd /Users/linghuang/Git/Agentic-Search
git add cli/internal/models/models.go
git commit -m "feat(go): add AgentRequest, AgentDocument, AgentResult models"
```

---

### Task 2: Add `QueryAgent()` to `internal/api/client.go`

**Files:**
- Modify: `cli/internal/api/client.go`

- [ ] **Step 1: Write failing test**

Add to `cli/internal/api/client_test.go` (or create it if it only has stub content):

```go
// TestQueryAgent verifies the full request/response cycle against a fake server.
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
		if r.URL.Path != "/api/agent" {
			t.Errorf("path = %s, want /api/agent", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("Authorization = %s, want Bearer test-key", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(fakeResp)
	}))
	defer srv.Close()

	cfg := config.Config{ServerURL: srv.URL, APIKey: "test-key"}
	c := NewClient(cfg)
	sessionID := "s1"
	result, err := c.QueryAgent(context.Background(), "show Q3 results", 5, &sessionID)
	if err != nil {
		t.Fatalf("QueryAgent: %v", err)
	}
	if result.SessionID != "sess-1" {
		t.Errorf("SessionID = %q, want %q", result.SessionID, "sess-1")
	}
	if result.Answer != "Revenue grew 12%." {
		t.Errorf("Answer = %q", result.Answer)
	}
	if len(result.Documents) != 1 {
		t.Errorf("Documents count = %d, want 1", len(result.Documents))
	}
}
```

Ensure the test file has all needed imports:
```go
import (
    "context"
    "encoding/json"
    "net/http"
    "net/http/httptest"
    "testing"

    "github.com/lingduoduo/Agentic-Search/cli/internal/config"
    "github.com/lingduoduo/Agentic-Search/cli/internal/models"
)
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli
go test ./internal/api/... -run TestQueryAgent -v
```
Expected: `undefined: (*Client).QueryAgent`

- [ ] **Step 3: Add `QueryAgent()` to `cli/internal/api/client.go`**

Add after the existing `Search()` method:

```go
// QueryAgent calls POST /api/agent and returns the parsed result.
func (c *Client) QueryAgent(ctx context.Context, query string, topK int, sessionID *string) (*models.AgentResult, error) {
	req := models.AgentRequest{
		Query:     query,
		TopK:      topK,
		SessionID: sessionID,
	}
	var result models.AgentResult
	if err := c.doJSONWith(ctx, c.searchHTTPClient, "POST", "/agent", req, &result); err != nil {
		return nil, err
	}
	return &result, nil
}
```

Also add `QueryAgent` to the `ClientAPI` interface:

```go
QueryAgent(ctx context.Context, query string, topK int, sessionID *string) (*models.AgentResult, error)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli
go test ./internal/api/... -run TestQueryAgent -v
```
Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
cd /Users/linghuang/Git/Agentic-Search
git add cli/internal/api/client.go cli/internal/api/client_test.go
git commit -m "feat(go): add QueryAgent to api.Client using doJSONWith pattern"
```

---

### Task 3: `internal/render` — progressive glamour markdown

**Files:**
- Create: `cli/internal/render/render.go`
- Create: `cli/internal/render/render_test.go`

This is extracted from `viewport.appendToken()` / `viewport.finishAgent()` in `cli/internal/tui/viewport.go`. The viewport throttles re-renders at 100 ms intervals (`streamRenderInterval`) and uses `glamour.WithStyles(style)` with zero left margin. The standalone render package applies the same logic to a plain `io.Writer`.

- [ ] **Step 1: Write failing tests**

```go
// cli/internal/render/render_test.go
package render_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/lingduoduo/Agentic-Search/cli/internal/models"
	"github.com/lingduoduo/Agentic-Search/cli/internal/render"
)

func TestRenderSources_PrintsTable(t *testing.T) {
	docs := []models.AgentDocument{
		{Citation: "[1]", Title: "Q3 Report", URL: strPtr("https://internal.corp/q3"), Content: "x"},
		{Citation: "[2]", Title: "", URL: nil, Content: "y"},
	}
	var buf bytes.Buffer
	render.Sources(&buf, docs)
	out := buf.String()
	if !strings.Contains(out, "Q3 Report") {
		t.Errorf("expected 'Q3 Report', got:\n%s", out)
	}
	if !strings.Contains(out, "internal.corp") {
		t.Errorf("expected URL, got:\n%s", out)
	}
	if !strings.Contains(out, "[1]") {
		t.Errorf("expected '[1]', got:\n%s", out)
	}
}

func TestRenderSources_EmptyNoOutput(t *testing.T) {
	var buf bytes.Buffer
	render.Sources(&buf, nil)
	if buf.Len() != 0 {
		t.Errorf("expected empty output, got: %q", buf.String())
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
		t.Errorf("expected empty output, got: %q", buf.String())
	}
}

func strPtr(s string) *string { return &s }
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli
go test ./internal/render/... -v
```
Expected: `cannot find package`

- [ ] **Step 3: Create `cli/internal/render/render.go`**

```go
// Package render provides terminal output helpers for the enterprise CLI.
// The Progressive function is extracted from cli/internal/tui/viewport.go's
// appendToken()/finishAgent() pattern, adapted for non-TUI (io.Writer) use.
package render

import (
	"fmt"
	"io"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/glamour/styles"
	"github.com/lingduoduo/Agentic-Search/cli/internal/models"
)

func urlStr(u *string) string {
	if u != nil && *u != "" {
		return *u
	}
	return "—"
}

func titleStr(t string) string {
	if t != "" {
		return t
	}
	return "—"
}

// Sources writes a tab-aligned source table to w. No-op if docs is empty.
func Sources(w io.Writer, docs []models.AgentDocument) {
	if len(docs) == 0 {
		return
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "Cite\tTitle\tURL")
	fmt.Fprintln(tw, "────\t─────\t───")
	for _, d := range docs {
		fmt.Fprintf(tw, "%s\t%s\t%s\n", d.Citation, titleStr(d.Title), urlStr(d.URL))
	}
	_ = tw.Flush()
	fmt.Fprintln(w)
}

// newRenderer builds a glamour renderer matching viewport.newMarkdownRenderer:
// dark style with zero left margin and word-wrap at `width` columns.
func newRenderer(width int) (*glamour.TermRenderer, error) {
	style := styles.DarkStyleConfig
	zero := uint(0)
	style.Document.Margin = &zero
	return glamour.NewTermRenderer(
		glamour.WithStyles(style),
		glamour.WithWordWrap(width-4),
	)
}

// Progressive writes answer to w word-by-word, re-rendering with glamour after
// each word and using ANSI cursor controls to overwrite the previous render
// (matching viewport.appendToken() with the same 100 ms throttle heuristic).
//
// wordsPerSecond controls animation speed (30 ≈ comfortable reading pace).
// width is the terminal column width used for glamour word-wrap.
func Progressive(w io.Writer, answer string, wordsPerSecond float64, width int) {
	words := strings.Fields(answer)
	if len(words) == 0 {
		return
	}

	delay := time.Duration(float64(time.Second) / wordsPerSecond)
	r, err := newRenderer(width)
	if err != nil {
		// Fallback: plain progressive print without markdown.
		for _, word := range words {
			fmt.Fprintf(w, "%s ", word)
			time.Sleep(delay)
		}
		fmt.Fprintln(w)
		return
	}

	accumulated := ""
	prevLines := 0

	for _, word := range words {
		if accumulated != "" {
			accumulated += " "
		}
		accumulated += word

		rendered, err := r.Render(accumulated)
		if err != nil {
			rendered = accumulated
		}

		// Clear previous render: move cursor up prevLines lines, erase to end.
		if prevLines > 0 {
			fmt.Fprintf(w, "\033[%dA\033[J", prevLines)
		}

		fmt.Fprint(w, rendered)
		prevLines = strings.Count(rendered, "\n")
		time.Sleep(delay)
	}
}
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli
go test ./internal/render/... -v
```
Expected: 4 tests `PASS`.

- [ ] **Step 5: Commit**

```bash
cd /Users/linghuang/Git/Agentic-Search
git add cli/internal/render/render.go cli/internal/render/render_test.go
git commit -m "feat(go): add internal/render — Sources table + Progressive glamour markdown"
```

---

### Task 4: `cmd/query/main.go` — entry point

**Files:**
- Create: `cli/cmd/query/main.go`

Reuses: `config.Load()` (reads `AGENTIC_SEARCH_URL`/`AGENTIC_SEARCH_PAT` env vars + `~/.config/agentic-search/config.json`), `api.NewClient()`, `client.QueryAgent()`, `render.Sources()`, `render.Progressive()`, `iostreams.System()`.

- [ ] **Step 1: Create `cli/cmd/query/main.go`**

```go
// cmd/query is the enterprise knowledge CLI.
//
// Usage:
//
//	query "summarise last quarter's results"
//	query -user-id alice -email alice@corp.com -secret s "what is our refund policy?"
//	query   # prompts interactively
//
// Auth priority: -token flag > AGENTIC_SEARCH_PAT env > config file APIKey > mint from -user-id.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	jwtlib "github.com/golang-jwt/jwt/v5"
	"github.com/lingduoduo/Agentic-Search/cli/internal/api"
	"github.com/lingduoduo/Agentic-Search/cli/internal/config"
	"github.com/lingduoduo/Agentic-Search/cli/internal/iostreams"
	"github.com/lingduoduo/Agentic-Search/cli/internal/render"
	"golang.org/x/term"
)

func main() {
	os.Exit(run())
}

func run() int {
	tokenFlag   := flag.String("token", "", "Personal access token / JWT (overrides AGENTIC_SEARCH_PAT and config)")
	userIDFlag  := flag.String("user-id", "", "User ID — mint a JWT when no token is available")
	emailFlag   := flag.String("email", "", "Email embedded in the minted JWT")
	secretFlag  := flag.String("secret", "", "JWT signing secret (falls back to AUTH_SECRET env var)")
	urlFlag     := flag.String("url", "", "Backend URL (overrides AGENTIC_SEARCH_URL and config)")
	topKFlag    := flag.Int("top-k", 5, "Number of source documents to retrieve")
	sessionFlag := flag.String("session-id", "", "Resume a prior chat session")
	widthFlag   := flag.Int("width", 0, "Terminal width for markdown wrapping (0 = auto-detect)")
	flag.Parse()

	ios := iostreams.System()

	// --- Query ---
	query := strings.TrimSpace(strings.Join(flag.Args(), " "))
	if query == "" {
		if !ios.IsInteractive() {
			fmt.Fprintln(ios.ErrOut, "error: no query provided")
			return 1
		}
		fmt.Fprint(ios.ErrOut, "Query: ")
		sc := bufio.NewScanner(ios.In)
		sc.Scan()
		query = strings.TrimSpace(sc.Text())
	}
	if query == "" {
		fmt.Fprintln(ios.ErrOut, "error: no query provided")
		return 1
	}

	// --- Config (file + env overrides) ---
	cfg := config.Load()
	if *urlFlag != "" {
		cfg.ServerURL = *urlFlag
	}

	// --- Auth ---
	token, err := resolveToken(*tokenFlag, cfg.APIKey, *userIDFlag, *emailFlag, *secretFlag)
	if err != nil {
		fmt.Fprintf(ios.ErrOut, "auth error: %v\n", err)
		return 1
	}
	cfg.APIKey = token

	// --- Query ---
	client := api.NewClient(cfg)
	fmt.Fprint(ios.ErrOut, "Searching enterprise knowledge… ")

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	var sid *string
	if *sessionFlag != "" {
		sid = sessionFlag
	}
	result, err := client.QueryAgent(ctx, query, *topKFlag, sid)
	if err != nil {
		fmt.Fprintf(ios.ErrOut, "\nerror: %v\n", err)
		return 1
	}
	fmt.Fprintln(ios.ErrOut, "done.")

	// --- Render ---
	width := *widthFlag
	if width <= 0 {
		if w, _, err := term.GetSize(int(os.Stdout.Fd())); err == nil && w > 0 {
			width = w
		} else {
			width = 100
		}
	}

	render.Sources(ios.Out, result.Documents)
	fmt.Fprintln(ios.Out, "─── Answer ─────────────────────────────────────────────────────")
	render.Progressive(ios.Out, result.Answer, 30.0, width)
	fmt.Fprintf(ios.Out, "\nsession_id: %s\n", result.SessionID)
	return 0
}

func resolveToken(flagToken, configToken, userID, email, secret string) (string, error) {
	if flagToken != "" {
		return flagToken, nil
	}
	if configToken != "" {
		return configToken, nil
	}
	if userID != "" {
		return mintJWT(userID, email, resolveSecret(secret))
	}
	return "", fmt.Errorf("provide -token, set AGENTIC_SEARCH_PAT, or pass -user-id to authenticate")
}

func resolveSecret(s string) string {
	if s != "" {
		return s
	}
	return os.Getenv("AUTH_SECRET")
}

func mintJWT(userID, email, secret string) (string, error) {
	claims := jwtlib.MapClaims{
		"sub": userID,
		"iat": time.Now().Unix(),
	}
	if email != "" {
		claims["email"] = email
	}
	tok := jwtlib.NewWithClaims(jwtlib.SigningMethodHS256, claims)
	return tok.SignedString([]byte(secret))
}
```

- [ ] **Step 2: Build**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli
go build ./cmd/query/
```
Expected: binary `query` created, no errors.

- [ ] **Step 3: Run `--help`**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli
./query -help 2>&1
```
Expected: usage printed with all flags (`-token`, `-user-id`, `-email`, `-secret`, `-url`, `-top-k`, `-session-id`, `-width`).

- [ ] **Step 4: Run all tests**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli
go test ./...
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/linghuang/Git/Agentic-Search
git add cli/cmd/query/main.go
git commit -m "feat(go): add cmd/query — enterprise knowledge CLI with progressive markdown"
```

---

### Task 5: Build binary + smoke test

**Files:** none (validation only)

- [ ] **Step 1: Build release binary into `bin/`**

```bash
cd /Users/linghuang/Git/Agentic-Search
mkdir -p bin
go build -C cli -o ../bin/query ./cmd/query
./bin/query -help
```
Expected: all flags documented, exits 0.

- [ ] **Step 2: Graceful failure — no server**

```bash
./bin/query "quarterly targets" -token fake.jwt -url http://localhost:9999
```
Expected: prints `Searching enterprise knowledge… ` then `error: ...connection refused`, exits 1. No panic.

- [ ] **Step 3: Mint JWT via Python backend and test auth path**

```bash
cd /Users/linghuang/Git/Agentic-Search
TOKEN=$(python3 - <<'EOF'
from src.internal.auth import generate_user_jwt_token
print(generate_user_jwt_token(user_id="dev-user", email="dev@local"))
EOF
)
./bin/query "test auth" -token "$TOKEN" -url http://localhost:9999 2>&1 | head -3
```
Expected: `Searching enterprise knowledge… ` then connection error — token resolved, server unreachable. No auth error.

- [ ] **Step 4: Test `-user-id` JWT minting path**

```bash
./bin/query "test" -user-id alice -email alice@corp.com -secret "$AUTH_SECRET" -url http://localhost:9999 2>&1 | head -3
```
Expected: `Searching enterprise knowledge… ` then connection error (not `auth error`).

- [ ] **Step 5: Clean checks**

```bash
cd /Users/linghuang/Git/Agentic-Search/cli
go vet ./...
go test -race ./...
```
Expected: no vet errors, no race conditions.

- [ ] **Step 6: Ignore build output**

```bash
grep -q "^bin/$" /Users/linghuang/Git/Agentic-Search/.gitignore || echo "bin/" >> /Users/linghuang/Git/Agentic-Search/.gitignore
git add .gitignore
git commit -m "chore: ignore bin/ build output"
```

---

## Self-Review

**Spec coverage:**
- [x] Go module setup + package reorganisation → Task 0
- [x] `AgentRequest`/`AgentResult`/`AgentDocument` models → Task 1
- [x] `QueryAgent()` using existing `doJSONWith()` + `searchHTTPClient` pattern → Task 2
- [x] Sources table using `text/tabwriter` → Task 3
- [x] Progressive glamour markdown (ANSI overwrite, zero-margin style) extracted from viewport → Task 3
- [x] Config via `config.Load()` (`AGENTIC_SEARCH_PAT`, `AGENTIC_SEARCH_URL`, config file) → Task 4
- [x] Token resolution: flag > env/config > mint from `-user-id` → Task 4
- [x] `IOStreams` TTY detection for interactive prompt → Task 4
- [x] Auto terminal width detection for glamour word-wrap → Task 4
- [x] Session resumption via `-session-id` → Task 4
- [x] Graceful failure (no panic on connection error) → Task 5
- [x] `go vet` + `-race` clean → Task 5

**Placeholder scan:** None. All code blocks are complete.

**Type consistency:**
- `render.Sources(w io.Writer, docs []models.AgentDocument)` — called as `render.Sources(ios.Out, result.Documents)` where `result.Documents` is `[]models.AgentDocument` ✓
- `render.Progressive(w io.Writer, answer string, wordsPerSecond float64, width int)` — called as `render.Progressive(ios.Out, result.Answer, 30.0, width)` ✓
- `client.QueryAgent(ctx, query string, topK int, sessionID *string) (*models.AgentResult, error)` — called with `sid *string` derived from `-session-id` flag ✓
- `config.NewClient(cfg config.Config)` → `api.NewClient(cfg)` after patching `cfg.APIKey = token` ✓
