"""OAuth provider registry."""

from .confluence_cloud import ConfluenceCloudOAuth
from .google_drive import GoogleDriveOAuth
from .provider import ConnectorOAuthProvider, OAuthSession
from .slack import SlackOAuth

OAUTH_PROVIDERS: dict[str, type[ConnectorOAuthProvider]] = {
    provider.PROVIDER_ID: provider
    for provider in (SlackOAuth, ConfluenceCloudOAuth, GoogleDriveOAuth)
}


def get_oauth_provider(provider_id: str) -> type[ConnectorOAuthProvider] | None:
    return OAUTH_PROVIDERS.get(provider_id.lower())


__all__ = [
    "ConnectorOAuthProvider",
    "ConfluenceCloudOAuth",
    "GoogleDriveOAuth",
    "OAUTH_PROVIDERS",
    "OAuthSession",
    "SlackOAuth",
    "get_oauth_provider",
]
