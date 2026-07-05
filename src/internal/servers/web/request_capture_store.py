"""In-memory rolling store of request-capture snapshots for the Dev Console.

Bounded ring buffer keyed by request_id. Not persisted — cleared on restart.
"""

from __future__ import annotations

from collections import OrderedDict


class RequestCaptureStore:
    def __init__(self, max_size: int = 20) -> None:
        self._max = max(1, max_size)
        self._items: "OrderedDict[str, dict]" = OrderedDict()

    def put(self, snapshot: dict) -> None:
        rid = snapshot["request_id"]
        self._items[rid] = snapshot
        self._items.move_to_end(rid)
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def get(self, request_id: str) -> dict | None:
        return self._items.get(request_id)

    def list(self) -> list[dict]:
        out = [
            {
                "request_id": s["request_id"],
                "query": s["query"],
                "created_at": s["created_at"],
                "route": s.get("route"),
                "stage_count": len(s.get("stages", [])),
            }
            for s in self._items.values()
        ]
        out.reverse()  # newest first
        return out
