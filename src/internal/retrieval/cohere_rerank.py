"""Cohere rerank API helper.

Extracted from the retired ``natural_language_processing`` package;
``retrieval/reranker.py`` is its only caller. Kept self-contained: importing this
module requires the optional ``cohere`` dependency, so callers import it inside a
try/except ImportError guard.
"""

import logging

from cohere import AsyncClient as CohereAsyncClient
from cohere.core.api_error import ApiError

logger = logging.getLogger(__name__)


class CohereBillingLimitError(Exception):
    """Raised when Cohere rejects requests because the billing cap is reached."""


async def cohere_rerank_api(
    query: str, docs: list[str], model_name: str, api_key: str
) -> list[float]:
    cohere_client = CohereAsyncClient(api_key=api_key)
    try:
        response = await cohere_client.rerank(
            query=query, documents=docs, model=model_name
        )
    except ApiError as err:
        if err.status_code == 402:
            logger.warning(
                "Cohere rerank request rejected due to billing cap. Falling back to retrieval ordering until billing resets."
            )
            raise CohereBillingLimitError(
                "Cohere billing limit reached for reranking"
            ) from err
        raise
    results = response.results
    sorted_results = sorted(results, key=lambda item: item.index)
    return [result.relevance_score for result in sorted_results]
