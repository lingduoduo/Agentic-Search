// cmd/query is the enterprise knowledge CLI.
//
// Usage:
//
//	query "summarise last quarter's results"
//	query -user-id alice -email alice@corp.com -secret s "what is our refund policy?"
//	query   # prompts interactively when stdin is a TTY
//
// Auth priority: -token flag > AGENTIC_SEARCH_PAT env / config file > mint JWT from -user-id.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/lingduoduo/Agentic-Search/cli/api"
	"github.com/lingduoduo/Agentic-Search/cli/clientauth"
	"github.com/lingduoduo/Agentic-Search/cli/config"
	"github.com/lingduoduo/Agentic-Search/cli/iostreams"
	"github.com/lingduoduo/Agentic-Search/cli/render"
	"golang.org/x/term"
)

func main() {
	os.Exit(run())
}

func run() int {
	tokenFlag := flag.String("token", "", "Personal access token / JWT (overrides AGENTIC_SEARCH_PAT and config)")
	userIDFlag := flag.String("user-id", "", "User ID — mint a JWT when no token is available")
	emailFlag := flag.String("email", "", "Email embedded in the minted JWT")
	secretFlag := flag.String("secret", "", "JWT signing secret (falls back to AUTH_SECRET env var)")
	urlFlag := flag.String("url", "", "Backend URL (overrides AGENTIC_SEARCH_URL and config)")
	topKFlag := flag.Int("top-k", 5, "Number of source documents to retrieve")
	sessionFlag := flag.String("session-id", "", "Resume a prior chat session")
	widthFlag := flag.Int("width", 0, "Terminal width for markdown wrapping (0 = auto-detect)")
	flag.Parse()

	ios := iostreams.System()

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

	cfg := config.Load()
	if *urlFlag != "" {
		cfg.ServerURL = *urlFlag
	}

	token, err := clientauth.ResolveToken(*tokenFlag, cfg.APIKey, *userIDFlag, *emailFlag, *secretFlag)
	if err != nil {
		fmt.Fprintf(ios.ErrOut, "auth error: %v\n", err)
		return 1
	}
	cfg.APIKey = token

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
