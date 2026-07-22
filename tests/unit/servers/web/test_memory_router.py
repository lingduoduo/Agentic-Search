from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.internal.db.store import AgenticSearchStore
from src.internal.memory.router import create_memory_router


class _FakeLLM:
    def __init__(self, text="[]"):
        self._text = text

    def complete(self, prompt, **kwargs):
        return self._text


def _client(db, llm=None):
    app = FastAPI()
    app.include_router(create_memory_router(db, llm))
    return TestClient(app)


def test_save_list_search_consolidate_profile_without_llm():
    db = AgenticSearchStore(":memory:")
    c = _client(db)

    r = c.post("/api/memory/save", json={"text": "User enjoys hiking in the mountains"})
    assert r.status_code == 200 and r.json()["memory_id"]

    listed = c.get("/api/memory/list").json()["memories"]
    assert listed[0]["text"] == "User enjoys hiking in the mountains"

    hits = c.post(
        "/api/memory/search", json={"query": "mountain hiking", "max_results": 5}
    ).json()
    assert hits["results"] and "hiking" in hits["results"][0]["text"]

    report = c.post("/api/memory/consolidate", json={"resolve_conflicts": True}).json()[
        "report"
    ]
    assert report["initial"] == 1 and report["final"] == 1

    assert c.get("/api/memory/profile").json()["profile"] == []
    db.close()


def test_llm_endpoints_503_without_llm():
    db = AgenticSearchStore(":memory:")
    c = _client(db, llm=None)
    assert c.post("/api/memory/profile/generate").status_code == 503
    assert c.post("/api/memory/curate", json={}).status_code == 503
    db.close()


def test_generate_profile_with_fake_llm():
    db = AgenticSearchStore(":memory:")
    db.add_user_memory("default_user", "User is a software engineer at TechCorp")
    llm = _FakeLLM(
        '[{"topic":"work","subtopic":"role","content":"Software engineer at TechCorp"}]'
    )
    c = _client(db, llm)
    profile = c.post("/api/memory/profile/generate").json()["profile"]
    assert profile[0]["topic"] == "work"
    assert (
        c.get("/api/memory/profile").json()["profile"][0]["content"]
        == "Software engineer at TechCorp"
    )
    db.close()
