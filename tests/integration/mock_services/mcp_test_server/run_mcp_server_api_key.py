import sys

from fastapi.responses import PlainTextResponse
from fastmcp import FastMCP
from fastmcp.server.auth import StaticTokenVerifier


def make_many_tools(mcp: FastMCP) -> None:
    def make_tool(i: int) -> None:
        @mcp.tool(name=f"tool_{i}", description=f"Get secret value {i}")
        def tool_name(name: str) -> str:  # noqa: ARG001
            """Get secret value."""
            return f"Secret value {200 - i}!"

    for i in range(100):
        make_tool(i)


if __name__ == "__main__":
    # Accept only these tokens (treat them like API keys) and require a scope
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = "dev-api-key-123"

    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    else:
        port = 8001

    auth = StaticTokenVerifier(
        tokens={
            api_key: {"client_id": "evan", "scopes": ["mcp:use"]},
        },
        required_scopes=["mcp:use"],
    )

    # Create FastMCP instance - it will handle /mcp path internally
    make_many_tools(mcp)  # noqa: F821,F841

    # Get the MCP HTTP app (configured to serve at /mcp)

    # Create wrapper FastAPI app with the MCP app's lifespan

    # Health check (unprotected)
    @app.get("/healthz")  # noqa: F821,F841
    def health() -> PlainTextResponse:
        return PlainTextResponse("ok")

    # Mount MCP app at root - it handles /mcp internally

    # Run the server
