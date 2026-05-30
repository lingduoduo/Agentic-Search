"""Stub exceptions for the generated API client."""


class ApiException(Exception):
    def __init__(self, status: int = 0, reason: str = "") -> None:
        self.status = status
        self.reason = reason
        super().__init__(f"({status}) {reason}")
