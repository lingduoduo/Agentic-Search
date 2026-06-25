from src.model.serving import ServerManager


class _Conformer:
    async def generate(self, request_id, prompt_ids, sampling_params):
        return [1, 2, 3]


class _NonConformer:
    def something_else(self): ...


def test_protocol_is_runtime_checkable():
    assert isinstance(_Conformer(), ServerManager)
    assert not isinstance(_NonConformer(), ServerManager)


def test_managers_importable_from_serving():
    from src.model.serving import OpenAIServerManager, LocalServerManager

    assert OpenAIServerManager is not None and LocalServerManager is not None


def test_managers_still_importable_from_examples_shim():
    # Back-compat: existing call sites import from the examples module.
    from examples.run_agentic_search import OpenAIServerManager as A
    from src.model.serving import OpenAIServerManager as B

    assert A is B  # same class object, not a copy


def test_concrete_managers_conform_to_protocol():
    from src.model.serving import OpenAIServerManager, LocalServerManager

    # Do not instantiate LocalServerManager (it loads a real HF model).
    # Assert the async generate method exists on each class.
    assert callable(getattr(OpenAIServerManager, "generate", None))
    assert callable(getattr(LocalServerManager, "generate", None))
