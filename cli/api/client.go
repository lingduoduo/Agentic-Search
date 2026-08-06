// Package api provides the HTTP client for communicating with the Agentic Search server.
package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/lingduoduo/Agentic-Search/cli/config"
	"github.com/lingduoduo/Agentic-Search/cli/models"
)

// Client is the Agentic Search API client.
//
// Two http.Clients are kept so each call site can pick a timeout matched to its
// expected work: 3min for quick JSON endpoints, and 5min for the LLM-backed ones
// (/agent, /memory/profile/generate, /memory/curate).
type Client struct {
	baseURL          string
	apiKey           string
	httpClient       *http.Client
	searchHTTPClient *http.Client
}

// NewClient creates a new API client from config.
// ServerURL is the server origin (e.g. "http://localhost:7860").
// APIURL appends the /api prefix to form the API base URL.
func NewClient(cfg config.Config) *Client {
	var transport *http.Transport
	if t, ok := http.DefaultTransport.(*http.Transport); ok {
		transport = t.Clone()
	} else {
		transport = &http.Transport{}
	}
	return &Client{
		baseURL: config.APIURL(cfg.ServerURL),
		apiKey:  cfg.APIKey,
		httpClient: &http.Client{
			Timeout:   3 * time.Minute,
			Transport: transport,
		},
		searchHTTPClient: &http.Client{
			Timeout:   5 * time.Minute,
			Transport: transport,
		},
	}
}

func (c *Client) newRequest(ctx context.Context, method, path string, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, body)
	if err != nil {
		return nil, err
	}
	if c.apiKey != "" {
		bearer := "Bearer " + c.apiKey
		req.Header.Set("Authorization", bearer)
	}
	return req, nil
}

func checkResponse(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	body, _ := io.ReadAll(resp.Body)
	if isHTMLResponse(resp.Header.Get("Content-Type"), body) {
		return &APIError{
			StatusCode: resp.StatusCode,
			Detail:     "server returned HTML instead of JSON — check that your server URL is correct",
		}
	}
	return &APIError{StatusCode: resp.StatusCode, Detail: string(body)}
}

func isHTMLResponse(contentType string, body []byte) bool {
	if strings.Contains(contentType, "text/html") {
		return true
	}
	lower := strings.ToLower(strings.TrimSpace(string(body)))
	return strings.HasPrefix(lower, "<!doctype") || strings.HasPrefix(lower, "<html")
}

func wrapTimeoutError(err error) error {
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return &APIError{StatusCode: 408, Detail: fmt.Sprintf("request timed out: %v", err)}
	}
	return err
}

func (c *Client) doJSONWith(ctx context.Context, httpClient *http.Client, method, path string, reqBody any, result any) error {
	var body io.Reader
	if reqBody != nil {
		data, err := json.Marshal(reqBody)
		if err != nil {
			return err
		}
		body = bytes.NewReader(data)
	}

	req, err := c.newRequest(ctx, method, path, body)
	if err != nil {
		return err
	}
	if reqBody != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return wrapTimeoutError(err)
	}
	defer func() { _ = resp.Body.Close() }()

	if err := checkResponse(resp); err != nil {
		return err
	}

	if result != nil {
		return json.NewDecoder(resp.Body).Decode(result)
	}
	return nil
}

func (c *Client) doJSON(ctx context.Context, method, path string, reqBody any, result any) error {
	return c.doJSONWith(ctx, c.httpClient, method, path, reqBody, result)
}

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

// SaveMemory calls POST /api/memory/save.
func (c *Client) SaveMemory(ctx context.Context, text string) (*models.MemorySaveResponse, error) {
	var resp models.MemorySaveResponse
	if err := c.doJSON(ctx, "POST", "/memory/save", models.MemorySaveRequest{Text: text}, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ListMemories calls GET /api/memory/list.
func (c *Client) ListMemories(ctx context.Context) (*models.MemoryListResponse, error) {
	var resp models.MemoryListResponse
	if err := c.doJSON(ctx, "GET", "/memory/list", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// SearchMemories calls POST /api/memory/search.
func (c *Client) SearchMemories(ctx context.Context, query string, maxResults int) (*models.MemorySearchResponse, error) {
	var resp models.MemorySearchResponse
	req := models.MemorySearchRequest{Query: query, MaxResults: maxResults}
	if err := c.doJSON(ctx, "POST", "/memory/search", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ConsolidateMemories calls POST /api/memory/consolidate.
func (c *Client) ConsolidateMemories(ctx context.Context, resolveConflicts bool) (*models.MemoryConsolidateResponse, error) {
	var resp models.MemoryConsolidateResponse
	req := models.MemoryConsolidateRequest{ResolveConflicts: resolveConflicts}
	if err := c.doJSON(ctx, "POST", "/memory/consolidate", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// GetMemoryProfile calls GET /api/memory/profile.
func (c *Client) GetMemoryProfile(ctx context.Context) (*models.MemoryProfileResponse, error) {
	var resp models.MemoryProfileResponse
	if err := c.doJSON(ctx, "GET", "/memory/profile", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// GenerateMemoryProfile calls POST /api/memory/profile/generate.
func (c *Client) GenerateMemoryProfile(ctx context.Context) (*models.MemoryProfileResponse, error) {
	var resp models.MemoryProfileResponse
	if err := c.doJSONWith(ctx, c.searchHTTPClient, "POST", "/memory/profile/generate", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// CurateMemory calls POST /api/memory/curate.
func (c *Client) CurateMemory(ctx context.Context, sessionID *string) (*models.MemoryCurateResponse, error) {
	var resp models.MemoryCurateResponse
	req := models.MemoryCurateRequest{SessionID: sessionID}
	if err := c.doJSONWith(ctx, c.searchHTTPClient, "POST", "/memory/curate", req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ClientAPI is the interface satisfied by Client.
type ClientAPI interface {
	QueryAgent(ctx context.Context, query string, topK int, sessionID *string) (*models.AgentResult, error)
	SaveMemory(ctx context.Context, text string) (*models.MemorySaveResponse, error)
	ListMemories(ctx context.Context) (*models.MemoryListResponse, error)
	SearchMemories(ctx context.Context, query string, maxResults int) (*models.MemorySearchResponse, error)
	ConsolidateMemories(ctx context.Context, resolveConflicts bool) (*models.MemoryConsolidateResponse, error)
	GetMemoryProfile(ctx context.Context) (*models.MemoryProfileResponse, error)
	GenerateMemoryProfile(ctx context.Context) (*models.MemoryProfileResponse, error)
	CurateMemory(ctx context.Context, sessionID *string) (*models.MemoryCurateResponse, error)
}

var _ ClientAPI = (*Client)(nil)
