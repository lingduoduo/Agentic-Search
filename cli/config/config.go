package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	EnvServerURL = "AGENTIC_SEARCH_URL"
	EnvAPIKey    = "AGENTIC_SEARCH_PAT"
)

// Config holds the CLI configuration.
type Config struct {
	ServerURL string `json:"server_url"`
	APIKey    string `json:"api_key"`
}

// DefaultConfig returns a config with default values.
func DefaultConfig() Config {
	return Config{
		ServerURL: "http://localhost:7860",
		APIKey:    "",
	}
}

// APIURL appends the API prefix (default "/api") to the server origin.
// Set AGENTIC_SEARCH_API_PREFIX="" for direct backend access without the proxy prefix.
func APIURL(serverURL string) string {
	prefix := "/api"
	if v, ok := os.LookupEnv("AGENTIC_SEARCH_API_PREFIX"); ok {
		prefix = v
	}
	u := strings.TrimRight(serverURL, "/")
	if prefix == "" {
		return u
	}
	return u + "/" + strings.Trim(prefix, "/")
}

// ConfigDir returns ~/.config/agentic-search
func ConfigDir() string {
	if xdg := os.Getenv("XDG_CONFIG_HOME"); xdg != "" {
		return filepath.Join(xdg, "agentic-search")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(".", ".config", "agentic-search")
	}
	return filepath.Join(home, ".config", "agentic-search")
}

// ConfigFilePath returns the full path to the config file.
func ConfigFilePath() string {
	return filepath.Join(ConfigDir(), "config.json")
}

// Load reads config from file and applies environment variable overrides.
// A malformed config file is reported on stderr and the defaults are used.
func Load() Config {
	cfg := DefaultConfig()

	if data, err := os.ReadFile(ConfigFilePath()); err == nil {
		if jsonErr := json.Unmarshal(data, &cfg); jsonErr != nil {
			fmt.Fprintf(os.Stderr, "warning: config file %s is malformed: %v (using defaults)\n", ConfigFilePath(), jsonErr)
		}
	}

	if v := os.Getenv(EnvServerURL); v != "" {
		cfg.ServerURL = v
	}
	if v := os.Getenv(EnvAPIKey); v != "" {
		cfg.APIKey = v
	}

	return cfg
}
