from src.model.intent_classifier import IntentPipeline
from src.model.intent_distillation import main


def test_cli_offline_trains_from_file(tmp_path):
    q = tmp_path / "q.txt"
    q.write_text(
        "\n".join(
            [
                "find faiss",
                "find bm25",
                "what is faiss",
                "what is hnsw",
                "create a ticket for faiss",
                "create a ticket for bm25",
            ]
        )
    )
    pt = tmp_path / "model.pt"
    rc = main(["--queries-file", str(q), "--output", str(pt), "--epochs", "30"])
    assert rc == 0
    assert pt.exists()
    assert IntentPipeline.load(str(pt)).predict_text("find hnsw").intent in {
        "chat",
        "search",
        "tool",
    }
