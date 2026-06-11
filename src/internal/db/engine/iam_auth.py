from __future__ import annotations

import functools
import os
import ssl
from typing import Any

import boto3

POSTGRES_HOST: str = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT: str = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_USER: str = os.environ.get("POSTGRES_USER", "postgres")
USE_IAM_AUTH: bool = os.environ.get("USE_IAM_AUTH", "false").lower() == "true"
SSL_CERT_FILE: str = os.environ.get("SSL_CERT_FILE", "rds-combined-ca-bundle.pem")


def get_iam_auth_token(
    host: str, port: str, user: str, region: str = "us-east-2"
) -> str:
    client = boto3.client("rds", region_name=region)
    return client.generate_db_auth_token(
        DBHostname=host, Port=int(port), DBUsername=user
    )


def configure_psycopg2_iam_auth(
    cparams: dict[str, Any], host: str, port: str, user: str, region: str
) -> None:
    token = get_iam_auth_token(host, port, user, region)
    cparams["password"] = token
    cparams["sslmode"] = "require"
    cparams["sslrootcert"] = SSL_CERT_FILE


def provide_iam_token(
    dialect: Any,  # noqa: ARG001
    conn_rec: Any,  # noqa: ARG001
    cargs: Any,  # noqa: ARG001
    cparams: Any,
) -> None:
    if USE_IAM_AUTH:
        region = os.environ.get("AWS_REGION_NAME", "us-east-2")
        configure_psycopg2_iam_auth(
            cparams, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, region
        )


@functools.cache
def create_ssl_context_if_iam() -> ssl.SSLContext | None:
    if USE_IAM_AUTH:
        return ssl.create_default_context(cafile=SSL_CERT_FILE)
    return None
