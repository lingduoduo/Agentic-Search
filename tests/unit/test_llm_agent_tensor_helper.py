"""Unit tests for src.model.tensor_helper."""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed", exc_type=ImportError)

from src import TensorConfig, TensorHelper  # noqa: E402


def _helper() -> TensorHelper:
    return TensorHelper(
        TensorConfig(
            pad_token_id=0,
            max_prompt_length=32,
            max_obs_length=16,
            max_start_length=8,
        )
    )


def test_convert_pad_structure_moves_padding_left_by_default():
    helper = _helper()
    tensor = torch.tensor([[1, 2, 0, 0]])
    result, _ = helper.convert_pad_structure(tensor)
    assert result.tolist() == [[0, 0, 1, 2]]


def test_convert_pad_structure_moves_padding_right_when_requested():
    helper = _helper()
    tensor = torch.tensor([[0, 0, 1, 2]])
    result, _ = helper.convert_pad_structure(tensor, pad_to_left=False)
    assert result.tolist() == [[1, 2, 0, 0]]


def test_example_level_pad_restores_batch_shape():
    helper = _helper()
    responses = torch.tensor([[7, 8], [9, 10]])
    active_mask = torch.tensor([True, False, True])
    padded, padded_str = helper.example_level_pad(
        responses,
        ["first", "third"],
        active_mask,
    )
    assert padded.tolist() == [[7, 8], [0, 0], [9, 10]]
    assert padded_str == ["first", "", "third"]


def test_example_level_pad_validates_active_count():
    helper = _helper()
    with pytest.raises(ValueError, match="active_mask"):
        helper.example_level_pad(
            torch.tensor([[1, 2]]), ["one"], torch.tensor([False, True, True])
        )
