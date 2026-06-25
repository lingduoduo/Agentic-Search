from src.model.serving import ServerManager


class _Conformer:
    async def generate(self, request_id, prompt_ids, sampling_params):
        return [1, 2, 3]


class _NonConformer:
    def something_else(self): ...


def test_protocol_is_runtime_checkable():
    assert isinstance(_Conformer(), ServerManager)
    assert not isinstance(_NonConformer(), ServerManager)
