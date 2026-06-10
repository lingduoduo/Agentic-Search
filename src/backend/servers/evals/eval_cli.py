#!/usr/bin/env python3
"""CLI for running evaluations."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import requests

from src.backend.servers.evals.eval import run_eval
from src.backend.servers.evals.models import EvalationAck
from src.backend.servers.evals.models import EvalConfigurationOptions
from src.backend.servers.evals.provider import get_provider


def load_data_local(local_data_path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(local_data_path):
        raise ValueError(f"Local data file does not exist: {local_data_path}")
    with open(local_data_path) as f:
        return json.load(f)


def configure_logging_for_evals(verbose: bool) -> None:
    if verbose:
        return
    os.environ["LOG_LEVEL"] = "WARNING"
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    for handler in root.handlers:
        handler.setLevel(logging.WARNING)
    for name in list(logging.Logger.manager.loggerDict.keys()):
        log = logging.getLogger(name)
        log.setLevel(logging.WARNING)
        for handler in log.handlers:
            handler.setLevel(logging.WARNING)
    logging.basicConfig(level=logging.WARNING, force=True)


def run_local(
    local_data_path: str | None,
    remote_dataset_name: str | None,
    search_permissions_email: str | None = None,
    no_send_logs: bool = False,
    local_only: bool = False,
    verbose: bool = False,
) -> EvalationAck:
    configure_logging_for_evals(verbose=verbose)

    if search_permissions_email is None:
        raise ValueError("search_permissions_email is required for local evaluation")

    configuration = EvalConfigurationOptions(
        search_permissions_email=search_permissions_email,
        dataset_name=remote_dataset_name or "local",
        no_send_logs=no_send_logs,
    )

    provider = get_provider(local_only=local_only)

    if remote_dataset_name:
        return run_eval(
            configuration=configuration,
            remote_dataset_name=remote_dataset_name,
            provider=provider,
        )

    if local_data_path is None:
        raise ValueError(
            "local_data_path or remote_dataset_name is required for local evaluation"
        )
    data = load_data_local(local_data_path)
    return run_eval(configuration=configuration, data=data, provider=provider)


def run_remote(
    base_url: str,
    api_key: str,
    remote_dataset_name: str,
    search_permissions_email: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if payload is None:
        payload = {}
    payload["search_permissions_email"] = search_permissions_email
    payload["dataset_name"] = remote_dataset_name
    url = f"{base_url}/api/evals/eval_run"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluations")
    parser.add_argument("--local-data-path", type=str)
    parser.add_argument("--remote-dataset-name", type=str)
    parser.add_argument("--braintrust-project", type=str, default="Agentic-Search")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--base-url", type=str, default="http://localhost:7860")
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--search-permissions-email", type=str)
    parser.add_argument("--no-send-logs", action="store_true", default=False)
    parser.add_argument("--local-only", action="store_true", default=False)
    args = parser.parse_args()

    if args.remote:
        api_key: str = args.api_key or os.environ.get("ONYX_EVAL_API_KEY", "")
        print(f"Running evaluation on remote server: {args.base_url}")
        try:
            result = run_remote(
                args.base_url,
                api_key,
                args.remote_dataset_name,
                search_permissions_email=args.search_permissions_email,
            )
            print(f"Remote evaluation triggered successfully: {result}")
        except requests.RequestException as e:
            print(f"Error triggering remote evaluation: {e}")
    else:
        if args.local_only:
            print("Running in local-only mode (no Braintrust)")
        if args.search_permissions_email:
            print(f"Using search permissions email: {args.search_permissions_email}")
        run_local(
            local_data_path=args.local_data_path,
            remote_dataset_name=args.remote_dataset_name,
            search_permissions_email=args.search_permissions_email,
            no_send_logs=args.no_send_logs,
            local_only=args.local_only,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
