"""Shared fallback objects for the generated OpenAPI client package."""


class Configuration:
    def __init__(self, host: str = "http://localhost:8080") -> None:
        self.host = host


class ApiClient:
    def __init__(self, configuration: Configuration | None = None) -> None:
        self.configuration = configuration or Configuration()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class ApiException(Exception):
    def __init__(self, status: int = 0, reason: str = "") -> None:
        self.status = status
        self.reason = reason
        super().__init__(f"({status}) {reason}")
