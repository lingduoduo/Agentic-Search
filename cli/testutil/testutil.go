package testutil

import (
	"net/http"
	"net/http/httptest"

	"github.com/lingduoduo/Agentic-Search/cli/api"
	"github.com/lingduoduo/Agentic-Search/cli/config"
)

// NewClient creates a test API client pointed at the given URL.
func NewClient(url string) *api.Client {
	return api.NewClient(config.Config{ServerURL: url, APIKey: "test-key"})
}

// StatusServer returns an httptest.Server that always responds with the given status code.
func StatusServer(status int) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
	}))
}

// DeadServerURL returns a URL whose server has already been closed.
func DeadServerURL() string {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	srv.Close()
	return url
}
