"""Lightweight spawn-safe probes for document parser boundary tests."""

from __future__ import annotations

import os
import time


def sleeping_parser(pid_queue, seconds: float):
    pid_queue.put(os.getpid())
    time.sleep(seconds)
    return {"document": {"file_name": "x.pdf", "file_type": "pdf"}}


def concurrency_probe_parser(active, maximum, lock, seconds: float):
    with lock:
        active.value += 1
        maximum.value = max(maximum.value, active.value)
    time.sleep(seconds)
    with lock:
        active.value -= 1
    return {"document": {"file_name": "x.pdf", "file_type": "pdf"}}


def resource_failure_parser():
    raise MemoryError("SENSITIVE-PARSER-DETAIL")


def memory_hog_parser(pid_connection, allocated_bytes: int, seconds: float):
    pid_connection.send(os.getpid())
    pid_connection.close()
    allocation = bytearray(allocated_bytes)
    time.sleep(seconds)
    return {
        "document": {
            "file_name": "x.pdf",
            "file_type": "pdf",
            "allocated": len(allocation),
        }
    }
