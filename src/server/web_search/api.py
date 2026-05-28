"""FastAPI routes for managing web-search providers.

The admin surface is intentionally small and in-process. It mirrors the provider
management shape used by larger FastAPI apps without pulling in a database or
auth system, which keeps this repo useful as a compact search-agent playground.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import field

from fastapi import APIRouter, FastAPI, HTTPException, Response

from src.tools.search import google_custom_search
from src.tools.search import retrieval_search
from src.tools.search import serpapi_search

from .models import WebContentProviderConfig
from .models import WebContentProviderTestRequest
from .models import WebContentProviderType
from .models import WebContentProviderUpsertRequest
from .models import WebContentProviderView
from .models import WebSearchProviderConfig
from .models import WebSearchProviderTestRequest
from .models import WebSearchProviderType
from .models import WebSearchProviderUpsertRequest
from .models import WebSearchProviderView


@dataclass
class _StoredSearchProvider:
    id: int
    name: str
    provider_type: WebSearchProviderType
    config: WebSearchProviderConfig = field(default_factory=WebSearchProviderConfig)
    api_key: str | None = None
    is_active: bool = False


@dataclass
class _StoredContentProvider:
    id: int
    name: str
    provider_type: WebContentProviderType
    config: WebContentProviderConfig = field(default_factory=WebContentProviderConfig)
    api_key: str | None = None
    is_active: bool = False


class WebSearchProviderStore:
    """Tiny in-memory store used by the FastAPI admin router."""

    def __init__(self) -> None:
        self._next_id = 1
        self._search_providers: dict[int, _StoredSearchProvider] = {}
        self._content_providers: dict[int, _StoredContentProvider] = {}

    def list_search_providers(self) -> list[WebSearchProviderView]:
        return [
            _search_view(provider)
            for provider in sorted(self._search_providers.values(), key=lambda p: p.id)
        ]

    def upsert_search_provider(
        self, request: WebSearchProviderUpsertRequest
    ) -> WebSearchProviderView:
        self._ensure_unique_search_name(request.name, request.id)

        provider_id = request.id or self._allocate_id()
        self._next_id = max(self._next_id, provider_id + 1)
        existing = self._search_providers.get(provider_id)
        api_key = existing.api_key if existing else None
        if request.api_key_changed or request.api_key is not None:
            api_key = request.api_key

        provider = _StoredSearchProvider(
            id=provider_id,
            name=request.name.strip(),
            provider_type=request.provider_type,
            config=request.config,
            api_key=api_key,
            is_active=existing.is_active if existing else False,
        )
        self._search_providers[provider_id] = provider
        if request.activate:
            self.activate_search_provider(provider_id)
        return _search_view(self._search_providers[provider_id])

    def delete_search_provider(self, provider_id: int) -> None:
        if self._search_providers.pop(provider_id, None) is None:
            raise HTTPException(status_code=404, detail="Search provider not found.")

    def activate_search_provider(self, provider_id: int) -> WebSearchProviderView:
        provider = self._get_search_provider(provider_id)
        for stored in self._search_providers.values():
            stored.is_active = stored.id == provider.id
        return _search_view(provider)

    def deactivate_search_provider(self, provider_id: int) -> None:
        self._get_search_provider(provider_id).is_active = False

    def get_search_provider_by_type(
        self, provider_type: WebSearchProviderType
    ) -> _StoredSearchProvider | None:
        active = [
            provider
            for provider in self._search_providers.values()
            if provider.provider_type == provider_type and provider.is_active
        ]
        if active:
            return active[0]
        return next(
            (
                provider
                for provider in self._search_providers.values()
                if provider.provider_type == provider_type
            ),
            None,
        )

    def list_content_providers(self) -> list[WebContentProviderView]:
        return [
            _content_view(provider)
            for provider in sorted(self._content_providers.values(), key=lambda p: p.id)
        ]

    def upsert_content_provider(
        self, request: WebContentProviderUpsertRequest
    ) -> WebContentProviderView:
        self._ensure_unique_content_name(request.name, request.id)

        provider_id = request.id or self._allocate_id()
        self._next_id = max(self._next_id, provider_id + 1)
        existing = self._content_providers.get(provider_id)
        api_key = existing.api_key if existing else None
        if request.api_key_changed or request.api_key is not None:
            api_key = request.api_key

        provider = _StoredContentProvider(
            id=provider_id,
            name=request.name.strip(),
            provider_type=request.provider_type,
            config=request.config,
            api_key=api_key,
            is_active=existing.is_active if existing else False,
        )
        self._content_providers[provider_id] = provider
        if request.activate:
            self.activate_content_provider(provider_id)
        return _content_view(self._content_providers[provider_id])

    def delete_content_provider(self, provider_id: int) -> None:
        if self._content_providers.pop(provider_id, None) is None:
            raise HTTPException(status_code=404, detail="Content provider not found.")

    def activate_content_provider(self, provider_id: int) -> WebContentProviderView:
        provider = self._get_content_provider(provider_id)
        for stored in self._content_providers.values():
            stored.is_active = stored.id == provider.id
        return _content_view(provider)

    def deactivate_content_provider(self, provider_id: int) -> None:
        self._get_content_provider(provider_id).is_active = False

    def reset_content_provider_default(self) -> None:
        for provider in self._content_providers.values():
            provider.is_active = False

    def get_content_provider_by_type(
        self, provider_type: WebContentProviderType
    ) -> _StoredContentProvider | None:
        active = [
            provider
            for provider in self._content_providers.values()
            if provider.provider_type == provider_type and provider.is_active
        ]
        if active:
            return active[0]
        return next(
            (
                provider
                for provider in self._content_providers.values()
                if provider.provider_type == provider_type
            ),
            None,
        )

    def _allocate_id(self) -> int:
        provider_id = self._next_id
        self._next_id += 1
        return provider_id

    def _get_search_provider(self, provider_id: int) -> _StoredSearchProvider:
        provider = self._search_providers.get(provider_id)
        if provider is None:
            raise HTTPException(status_code=404, detail="Search provider not found.")
        return provider

    def _get_content_provider(self, provider_id: int) -> _StoredContentProvider:
        provider = self._content_providers.get(provider_id)
        if provider is None:
            raise HTTPException(status_code=404, detail="Content provider not found.")
        return provider

    def _ensure_unique_search_name(self, name: str, provider_id: int | None) -> None:
        normalized = name.strip().lower()
        for existing in self._search_providers.values():
            if existing.name.lower() == normalized and existing.id != provider_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"A search provider named '{name}' already exists.",
                )

    def _ensure_unique_content_name(self, name: str, provider_id: int | None) -> None:
        normalized = name.strip().lower()
        for existing in self._content_providers.values():
            if existing.name.lower() == normalized and existing.id != provider_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"A content provider named '{name}' already exists.",
                )


def create_admin_router(
    store: WebSearchProviderStore | None = None,
) -> APIRouter:
    store = store or WebSearchProviderStore()
    router = APIRouter(prefix="/admin/web-search", tags=["web-search-admin"])

    @router.get("/search-providers", response_model=list[WebSearchProviderView])
    def list_search_providers() -> list[WebSearchProviderView]:
        return store.list_search_providers()

    @router.post("/search-providers", response_model=WebSearchProviderView)
    def upsert_search_provider(
        request: WebSearchProviderUpsertRequest,
    ) -> WebSearchProviderView:
        _validate_search_provider_config(
            request.provider_type, request.config, request.api_key
        )
        return store.upsert_search_provider(request)

    @router.delete(
        "/search-providers/{provider_id}",
        status_code=204,
        response_class=Response,
    )
    def delete_search_provider(provider_id: int) -> Response:
        store.delete_search_provider(provider_id)
        return Response(status_code=204)

    @router.post(
        "/search-providers/{provider_id}/activate",
        response_model=WebSearchProviderView,
    )
    def activate_search_provider(provider_id: int) -> WebSearchProviderView:
        return store.activate_search_provider(provider_id)

    @router.post("/search-providers/{provider_id}/deactivate")
    def deactivate_search_provider(provider_id: int) -> dict[str, str]:
        store.deactivate_search_provider(provider_id)
        return {"status": "ok"}

    @router.post("/search-providers/test")
    def test_search_provider(request: WebSearchProviderTestRequest) -> dict[str, str]:
        api_key = _resolve_search_api_key(store, request)
        _validate_search_provider_config(
            request.provider_type, request.config, api_key, for_test=True
        )
        if request.live:
            _run_async(_test_search_provider_live(request, api_key))
        return {"status": "ok"}

    @router.get("/content-providers", response_model=list[WebContentProviderView])
    def list_content_providers() -> list[WebContentProviderView]:
        return store.list_content_providers()

    @router.post("/content-providers", response_model=WebContentProviderView)
    def upsert_content_provider(
        request: WebContentProviderUpsertRequest,
    ) -> WebContentProviderView:
        return store.upsert_content_provider(request)

    @router.delete(
        "/content-providers/{provider_id}",
        status_code=204,
        response_class=Response,
    )
    def delete_content_provider(provider_id: int) -> Response:
        store.delete_content_provider(provider_id)
        return Response(status_code=204)

    @router.post(
        "/content-providers/{provider_id}/activate",
        response_model=WebContentProviderView,
    )
    def activate_content_provider(provider_id: int) -> WebContentProviderView:
        return store.activate_content_provider(provider_id)

    @router.post("/content-providers/reset-default")
    def reset_content_provider_default() -> dict[str, str]:
        store.reset_content_provider_default()
        return {"status": "ok"}

    @router.post("/content-providers/{provider_id}/deactivate")
    def deactivate_content_provider(provider_id: int) -> dict[str, str]:
        store.deactivate_content_provider(provider_id)
        return {"status": "ok"}

    @router.post("/content-providers/test")
    def test_content_provider(request: WebContentProviderTestRequest) -> dict[str, str]:
        _resolve_content_api_key(store, request)
        if request.live:
            from src.tools.search import fetch_url

            body = _run_async(
                fetch_url(
                    request.url,
                    max_length=request.config.max_length,
                    timeout_seconds=request.config.timeout_seconds,
                )
            )
            if not body or body.startswith("[fetch error]"):
                raise HTTPException(
                    status_code=400,
                    detail="Content provider validation failed.",
                )
        return {"status": "ok"}

    return router


def create_app(store: WebSearchProviderStore | None = None) -> FastAPI:
    app = FastAPI(title="Web Search Admin Server")

    @app.get("/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(create_admin_router(store))
    return app


def provider_requires_api_key(provider_type: WebSearchProviderType) -> bool:
    return provider_type in {
        WebSearchProviderType.GOOGLE,
        WebSearchProviderType.SERPAPI,
    }


def _validate_search_provider_config(
    provider_type: WebSearchProviderType,
    config: WebSearchProviderConfig,
    api_key: str | None,
    *,
    for_test: bool = False,
) -> None:
    if provider_requires_api_key(provider_type) and not api_key:
        raise HTTPException(
            status_code=400,
            detail="API key is required for this provider.",
        )
    if provider_type == WebSearchProviderType.RETRIEVAL and not config.search_url:
        raise HTTPException(
            status_code=400,
            detail="config.search_url is required for retrieval providers.",
        )
    if provider_type == WebSearchProviderType.GOOGLE and not config.cse_id:
        raise HTTPException(
            status_code=400,
            detail="config.cse_id is required for Google providers.",
        )
    if for_test and not config.page_size:
        raise HTTPException(status_code=400, detail="config.page_size is required.")


def _resolve_search_api_key(
    store: WebSearchProviderStore,
    request: WebSearchProviderTestRequest,
) -> str | None:
    if not request.use_stored_key:
        return request.api_key
    provider = store.get_search_provider_by_type(request.provider_type)
    if provider is None or not provider.api_key:
        raise HTTPException(
            status_code=400,
            detail="No stored API key found for this provider type.",
        )
    return provider.api_key


def _resolve_content_api_key(
    store: WebSearchProviderStore,
    request: WebContentProviderTestRequest,
) -> str | None:
    if not request.use_stored_key:
        return request.api_key
    provider = store.get_content_provider_by_type(request.provider_type)
    if provider is None or not provider.api_key:
        raise HTTPException(
            status_code=400,
            detail="No stored API key found for this provider type.",
        )
    return provider.api_key


async def _test_search_provider_live(
    request: WebSearchProviderTestRequest, api_key: str | None
) -> None:
    if request.provider_type == WebSearchProviderType.RETRIEVAL:
        assert request.config.search_url is not None
        pages = await retrieval_search(
            request.query,
            search_url=request.config.search_url,
            page_size=request.config.page_size,
            timeout_seconds=request.config.timeout_seconds,
        )
    elif request.provider_type == WebSearchProviderType.GOOGLE:
        pages = await google_custom_search(
            request.query,
            page_size=request.config.page_size,
            api_key=api_key,
            cse_id=request.config.cse_id,
            timeout_seconds=request.config.timeout_seconds,
        )
    else:
        pages = await serpapi_search(
            request.query,
            page_size=request.config.page_size,
            api_key=api_key,
            timeout_seconds=request.config.timeout_seconds,
        )
    if not pages or pages[0].error:
        raise HTTPException(
            status_code=400,
            detail=pages[0].error if pages else "Provider returned no results.",
        )


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise HTTPException(
        status_code=400,
        detail="Live provider tests cannot run from an active event loop.",
    )


def _search_view(provider: _StoredSearchProvider) -> WebSearchProviderView:
    return WebSearchProviderView(
        id=provider.id,
        name=provider.name,
        provider_type=provider.provider_type,
        is_active=provider.is_active,
        config=provider.config,
        masked_api_key=_mask_api_key(provider.api_key),
    )


def _content_view(provider: _StoredContentProvider) -> WebContentProviderView:
    return WebContentProviderView(
        id=provider.id,
        name=provider.name,
        provider_type=provider.provider_type,
        is_active=provider.is_active,
        config=provider.config,
        masked_api_key=_mask_api_key(provider.api_key),
    )


def _mask_api_key(api_key: str | None) -> str | None:
    if not api_key:
        return None
    if len(api_key) <= 8:
        return f"{api_key[:2]}...{api_key[-1:]}"
    return f"{api_key[:4]}...{api_key[-4:]}"


admin_router = create_admin_router()
