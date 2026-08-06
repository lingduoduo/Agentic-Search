package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func clearEnvVars(t *testing.T) {
	t.Helper()
	for _, key := range []string{EnvServerURL, EnvAPIKey} {
		t.Setenv(key, "")
	}
}

func writeConfig(t *testing.T, dir string, data []byte) {
	t.Helper()
	agentSearchDir := filepath.Join(dir, "agentic-search")
	if err := os.MkdirAll(agentSearchDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentSearchDir, "config.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.ServerURL != "http://localhost:7860" {
		t.Errorf("expected default server URL, got %s", cfg.ServerURL)
	}
	if cfg.APIKey != "" {
		t.Errorf("expected empty API key, got %s", cfg.APIKey)
	}
}

func TestLoadDefaults(t *testing.T) {
	clearEnvVars(t)
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	cfg := Load()
	if cfg.ServerURL != "http://localhost:7860" {
		t.Errorf("expected default URL, got %s", cfg.ServerURL)
	}
	if cfg.APIKey != "" {
		t.Errorf("expected empty key, got %s", cfg.APIKey)
	}
}

func TestLoadFromFile(t *testing.T) {
	clearEnvVars(t)
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	data, _ := json.Marshal(map[string]interface{}{
		"server_url": "https://my-agentic-search.example.com",
		"api_key":    "test-key-123",
	})
	writeConfig(t, dir, data)

	cfg := Load()
	if cfg.ServerURL != "https://my-agentic-search.example.com" {
		t.Errorf("got %s", cfg.ServerURL)
	}
	if cfg.APIKey != "test-key-123" {
		t.Errorf("got %s", cfg.APIKey)
	}
}

func TestLoadCorruptFile(t *testing.T) {
	clearEnvVars(t)
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	writeConfig(t, dir, []byte("not valid json {{{"))

	cfg := Load()
	if cfg.ServerURL != "http://localhost:7860" {
		t.Errorf("expected default URL on corrupt file, got %s", cfg.ServerURL)
	}
}

func TestEnvOverrideServerURL(t *testing.T) {
	clearEnvVars(t)
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv(EnvServerURL, "https://env-override.com")

	cfg := Load()
	if cfg.ServerURL != "https://env-override.com" {
		t.Errorf("got %s", cfg.ServerURL)
	}
}

func TestEnvOverrideAPIKey(t *testing.T) {
	clearEnvVars(t)
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	t.Setenv(EnvAPIKey, "env-key")

	cfg := Load()
	if cfg.APIKey != "env-key" {
		t.Errorf("got %s", cfg.APIKey)
	}
}

func TestEnvOverridesFileValues(t *testing.T) {
	clearEnvVars(t)
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	data, _ := json.Marshal(map[string]interface{}{
		"server_url": "https://file-url.com",
		"api_key":    "file-key",
	})
	writeConfig(t, dir, data)

	t.Setenv(EnvServerURL, "https://env-url.com")

	cfg := Load()
	if cfg.ServerURL != "https://env-url.com" {
		t.Errorf("env should override file, got %s", cfg.ServerURL)
	}
	if cfg.APIKey != "file-key" {
		t.Errorf("file value should be kept, got %s", cfg.APIKey)
	}
}

func TestAPIURL(t *testing.T) {
	cases := []struct {
		input string
		want  string
	}{
		{"http://localhost:7860", "http://localhost:7860/api"},
		{"http://localhost:7860/", "http://localhost:7860/api"},
		{"http://localhost:8080", "http://localhost:8080/api"},
		{"http://localhost:3000", "http://localhost:3000/api"},
	}
	for _, tc := range cases {
		got := APIURL(tc.input)
		if got != tc.want {
			t.Errorf("APIURL(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestAPIURLEmptyPrefix(t *testing.T) {
	t.Setenv("AGENTIC_SEARCH_API_PREFIX", "")
	cases := []struct {
		input string
		want  string
	}{
		{"http://localhost:8080", "http://localhost:8080"},
		{"http://localhost:8080/", "http://localhost:8080"},
		{"http://localhost:7860", "http://localhost:7860"},
	}
	for _, tc := range cases {
		got := APIURL(tc.input)
		if got != tc.want {
			t.Errorf("APIURL(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}
