import types
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("torch")


@pytest.mark.asyncio
async def test_run_feedback_grpo_step_wires_and_saves(monkeypatch, tmp_path):
    import examples._grpo_common as gc

    ex = [
        types.SimpleNamespace(question="q1", ground_truth="a1", metadata={"m": 1}),
        types.SimpleNamespace(question="q2", ground_truth="a2", metadata={"m": 2}),
    ]
    monkeypatch.setattr(
        "src.model.post_training.data.load_feedback_examples", lambda *a, **k: ex
    )

    trainer = MagicMock()
    trainer.step_async = AsyncMock(return_value={"loss": 0.1})
    trainer.policy = MagicMock()
    trainer.tokenizer = MagicMock()
    monkeypatch.setattr(
        "src.model.post_training.grpo.training.SearchAgentGRPOTrainer.from_pretrained",
        lambda *a, **k: trainer,
    )

    out = tmp_path / "ckpt"
    metrics = await gc.run_feedback_grpo_step(
        model_path="base",
        db_path=":memory:",
        output_dir=out,
        min_ratings=1,
        human_feedback_weight=0.5,
        num_rollouts=2,
        search_url="http://x/retrieve",
        device="cpu",
    )

    assert metrics == {"loss": 0.1}
    args, kwargs = trainer.step_async.call_args
    assert args[0] == ["q1", "q2"]
    assert args[1] == ["a1", "a2"]
    assert kwargs["metadata"] == [{"m": 1}, {"m": 2}]
    trainer.policy.save_pretrained.assert_called_once_with(out)
    trainer.tokenizer.save_pretrained.assert_called_once_with(out)
