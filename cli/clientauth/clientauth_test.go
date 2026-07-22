package clientauth_test

import (
	"testing"

	jwtlib "github.com/golang-jwt/jwt/v5"
	"github.com/lingduoduo/Agentic-Search/cli/clientauth"
)

func TestMintJWTHasSubClaim(t *testing.T) {
	tok, err := clientauth.MintJWT("alice", "a@example.com", "s3cret")
	if err != nil {
		t.Fatalf("MintJWT: %v", err)
	}
	parsed, err := jwtlib.Parse(tok, func(*jwtlib.Token) (any, error) { return []byte("s3cret"), nil })
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	claims := parsed.Claims.(jwtlib.MapClaims)
	if claims["sub"] != "alice" {
		t.Errorf("sub = %v, want alice", claims["sub"])
	}
	if claims["email"] != "a@example.com" {
		t.Errorf("email = %v, want a@example.com", claims["email"])
	}
}

func TestResolveSecretPrefersAgenticEnv(t *testing.T) {
	t.Setenv("AGENTIC_SEARCH_AUTH_SECRET", "agentic")
	t.Setenv("AUTH_SECRET", "legacy")
	if got := clientauth.ResolveSecret(""); got != "agentic" {
		t.Errorf("ResolveSecret = %q, want agentic", got)
	}
	if got := clientauth.ResolveSecret("explicit"); got != "explicit" {
		t.Errorf("ResolveSecret(explicit) = %q, want explicit", got)
	}
}

func TestResolveTokenPrefersFlag(t *testing.T) {
	got, err := clientauth.ResolveToken("flagtok", "cfgtok", "", "", "")
	if err != nil || got != "flagtok" {
		t.Fatalf("ResolveToken = %q, %v; want flagtok", got, err)
	}
}
