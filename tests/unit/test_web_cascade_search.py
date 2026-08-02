import asyncio


from src.internal.tools.search import SearchPage, make_web_cascade_search


def _ok(url):
    return SearchPage(title="t", summary="s", url=url)


def _err():
    return SearchPage(error="boom")


def test_serpapi_hit_skips_browser():
    async def fake_serp(query, **kw):
        return [_ok("http://serp/1")]

    async def fake_browser(query, **kw):  # must NOT be called
        raise AssertionError("browser should not run when serpapi returns results")

    fn = make_web_cascade_search(
        browser_search_url="http://browser/retrieve",
        serpapi_fn=fake_serp,
        browser_fn=fake_browser,
    )
    pages = asyncio.run(fn("q"))
    assert [p.url for p in pages] == ["http://serp/1"]


def test_serpapi_empty_falls_back_to_browser():
    async def fake_serp(query, **kw):
        return []

    async def fake_browser(query, *, provider, search_url, **kw):
        assert provider == "retrieval"
        assert search_url == "http://browser/retrieve"
        return [_ok("http://browser/1")]

    fn = make_web_cascade_search(
        browser_search_url="http://browser/retrieve",
        serpapi_fn=fake_serp,
        browser_fn=fake_browser,
    )
    pages = asyncio.run(fn("q"))
    assert [p.url for p in pages] == ["http://browser/1"]


def test_serpapi_error_falls_back_to_browser():
    async def fake_serp(query, **kw):
        return [_err()]

    async def fake_browser(query, **kw):
        return [_ok("http://browser/2")]

    fn = make_web_cascade_search(
        browser_search_url="http://browser/retrieve",
        serpapi_fn=fake_serp,
        browser_fn=fake_browser,
    )
    pages = asyncio.run(fn("q"))
    assert [p.url for p in pages] == ["http://browser/2"]


def test_failed_legs_report_why_instead_of_looking_empty():
    # "No results found." is indistinguishable from a working search over a
    # topic with no hits, so a rate-limited key or an unconfigured fallback
    # looked like the tool ran fine and the web had nothing.
    async def fake_serp(query, **kw):
        return [SearchPage(error="429, message='Too Many Requests'")]

    fn = make_web_cascade_search(browser_search_url=None, serpapi_fn=fake_serp)
    pages = asyncio.run(fn("q"))

    errors = " ".join(p.error or "" for p in pages)
    assert "Too Many Requests" in errors
    assert "AGENTIC_SEARCH_BROWSER_SEARCH_URL" in errors  # names the missing fallback
    assert all(not p.url for p in pages)  # error pages carry no result


def test_both_legs_failing_reports_both():
    async def fake_serp(query, **kw):
        return [SearchPage(error="serp down")]

    async def fake_browser(query, **kw):
        return [SearchPage(error="browser down")]

    fn = make_web_cascade_search(
        browser_search_url="http://browser/retrieve",
        serpapi_fn=fake_serp,
        browser_fn=fake_browser,
    )
    errors = " ".join(p.error or "" for p in asyncio.run(fn("q")))
    assert "serp down" in errors and "browser down" in errors


def test_a_browser_exception_is_reported_not_swallowed():
    async def fake_serp(query, **kw):
        return [SearchPage(error="serp down")]

    async def fake_browser(query, **kw):
        raise ConnectionError("no route to host")

    fn = make_web_cascade_search(
        browser_search_url="http://browser/retrieve",
        serpapi_fn=fake_serp,
        browser_fn=fake_browser,
    )
    errors = " ".join(p.error or "" for p in asyncio.run(fn("q")))
    assert "no route to host" in errors


def test_a_genuinely_empty_search_stays_empty():
    # No error anywhere: the web really had nothing. Do not invent a failure.
    async def fake_serp(query, **kw):
        return []

    fn = make_web_cascade_search(browser_search_url=None, serpapi_fn=fake_serp)
    assert asyncio.run(fn("q")) == []


def test_seeded_web_search_uses_cascade_not_retrieval():
    from src.internal.tools.knowledge_base import tool_knowledge_base

    tools = {t.name: t for t in tool_knowledge_base(search_url="http://x/retrieve")}
    web = tools["web_search"]
    # The cascade search_fn is bound; the tool no longer routes to retrieval.
    assert web._search_fn is not None
    assert web._provider != "retrieval" or web._search_fn.__name__ == "_cascade"
