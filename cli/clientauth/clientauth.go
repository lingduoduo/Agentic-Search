// Package clientauth resolves and mints bearer tokens for the CLI binaries.
package clientauth

import (
	"fmt"
	"os"
	"time"

	jwtlib "github.com/golang-jwt/jwt/v5"
)

// ResolveToken picks a bearer token: an explicit flag, then a config token,
// then a freshly minted JWT for userID. Returns an error if none is available.
func ResolveToken(flagToken, configToken, userID, email, secret string) (string, error) {
	if flagToken != "" {
		return flagToken, nil
	}
	if configToken != "" {
		return configToken, nil
	}
	if userID != "" {
		return MintJWT(userID, email, ResolveSecret(secret))
	}
	return "", fmt.Errorf("provide -token, set AGENTIC_SEARCH_PAT, or pass -user-id to authenticate")
}

// ResolveSecret returns the JWT signing secret: an explicit value, else
// AGENTIC_SEARCH_AUTH_SECRET (what the backend verifies with), else AUTH_SECRET.
func ResolveSecret(s string) string {
	if s != "" {
		return s
	}
	if v := os.Getenv("AGENTIC_SEARCH_AUTH_SECRET"); v != "" {
		return v
	}
	return os.Getenv("AUTH_SECRET")
}

// MintJWT mints an HS256 token with sub/iat (and email when present).
func MintJWT(userID, email, secret string) (string, error) {
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
