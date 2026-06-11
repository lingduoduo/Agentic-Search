from src.internal.servers.evals.models import EvalProvider
from src.internal.servers.evals.providers.braintrust import BraintrustEvalProvider
from src.internal.servers.evals.providers.local import LocalEvalProvider


def get_provider(local_only: bool = False) -> EvalProvider:
    if local_only:
        return LocalEvalProvider()
    return BraintrustEvalProvider()
