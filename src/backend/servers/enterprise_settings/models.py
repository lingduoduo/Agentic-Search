"""Pydantic models for enterprise settings.

Kept close to the sampled Onyx models so callers that reference them compile
unchanged. The NavigationItem validator is ported to proper Pydantic v2 style
(model_validator instead of overriding model_validate).
"""

from __future__ import annotations

from enum import Enum
from urllib.parse import urlparse

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


class NavigationItem(BaseModel):
    link: str
    title: str
    icon: str | None = None
    svg_logo: str | None = None

    @model_validator(mode="after")
    def _exactly_one_of_icon_or_svg(self) -> "NavigationItem":
        if bool(self.icon) == bool(self.svg_logo):
            raise ValueError("Exactly one of icon or svg_logo must be specified.")
        return self


class LogoDisplayStyle(str, Enum):
    LOGO_AND_NAME = "logo_and_name"
    LOGO_ONLY = "logo_only"
    NAME_ONLY = "name_only"


class EnterpriseSettings(BaseModel):
    """Branding and UI customisation settings.

    Safe to expose without auth — nothing sensitive should be stored here.
    """

    application_name: str | None = None
    use_custom_logo: bool = False
    use_custom_logotype: bool = False
    logo_display_style: LogoDisplayStyle | None = None

    custom_nav_items: list[NavigationItem] = Field(default_factory=list)

    two_lines_for_chat_header: bool | None = None
    custom_lower_disclaimer_content: str | None = None
    custom_header_content: str | None = None
    custom_popup_header: str | None = None
    custom_popup_content: str | None = None
    enable_consent_screen: bool | None = None
    consent_screen_prompt: str | None = None
    show_first_visit_notice: bool | None = None
    custom_greeting_message: str | None = None

    custom_help_link_url: str | None = None
    custom_help_link_label: str | None = None
    hide_branding: bool | None = None

    @field_validator("custom_help_link_url")
    @classmethod
    def _validate_help_link_scheme(cls, v: str | None) -> str | None:
        if not v:
            return v
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                "custom_help_link_url must be an absolute http or https URL."
            )
        return v


class AnalyticsScriptUpload(BaseModel):
    script: str
    secret_key: str


__all__ = [
    "AnalyticsScriptUpload",
    "EnterpriseSettings",
    "LogoDisplayStyle",
    "NavigationItem",
]
