import generated.agentic_search_openapi_client.agentic_search_openapi_client as app_api  # ty: ignore[unresolved-import]

from tests.integration.common_utils.constants import API_SERVER_URL

api_config = app_api.Configuration(host=API_SERVER_URL)
