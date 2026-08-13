import json
from pathlib import Path

from src.model.intent_seed import propose_modules, write_seed_canonical


def test_imperative_send_proposes_the_send_module():
    assert "send" in propose_modules("send the runbook to the team", "tool")


def test_currency_cue_proposes_current_info():
    assert "current_info" in propose_modules(
        "what is the current price of bitcoin", "search"
    )


def test_document_noun_proposes_document_lookup():
    assert "lookup_document" in propose_modules(
        "find the onboarding doc for new engineers", "search"
    )


def test_summarize_verb_proposes_the_summarize_module():
    assert propose_modules("summarize the Q3 earnings report", "chat") == ("summarize",)


def test_a_proposal_may_carry_several_modules():
    """Real requests carry more than one intent at once."""
    proposed = propose_modules(
        "explain why the reranker got slower and compare it to the old one", "chat"
    )

    assert set(proposed) == {"explain", "compare"}


def test_every_proposal_is_valid_for_its_route():
    from src.model.intent_taxonomy import modules_for_route

    for route in ("chat", "search", "tool"):
        for text in ("something entirely unmatched by any cue", "the thing"):
            proposed = propose_modules(text, route)
            assert proposed, (route, text)
            assert set(proposed) <= set(modules_for_route(route))


def test_write_seed_produces_a_file_the_canonical_loader_accepts(tmp_path: Path):
    from src.model.intent_data import load_canonical_examples

    examples = [
        {"id": f"e-{i}", "text": t, "label": lbl, "source": "s"}
        for i, (t, lbl) in enumerate(
            [
                ("find the onboarding doc", "search"),
                ("summarize the outage postmortem", "chat"),
                ("send the runbook to the team", "tool"),
            ]
        )
    ]
    source = tmp_path / "examples.json"
    source.write_text(json.dumps(examples), encoding="utf-8")
    output = tmp_path / "canonical.json"

    count = write_seed_canonical(source, output)

    assert count == 3
    assert len(load_canonical_examples(output)) == 3
