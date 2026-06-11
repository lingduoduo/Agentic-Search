"""AES-256-CBC symmetric encryption for sensitive credential secrets.

When no key is configured the functions pass through plaintexts unchanged
so local development works without setting an env var. Set
``AGENTIC_SEARCH_ENCRYPTION_KEY`` (minimum 16 bytes) to enable encryption.
Keys longer than 32 bytes are trimmed to the largest valid AES key size
(32 / 24 / 16 bytes) that fits.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from os import urandom

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding as sym_padding
from cryptography.hazmat.primitives.ciphers import Cipher
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.ciphers import modes

logger = logging.getLogger(__name__)

ENCRYPTION_KEY_ENV = "AGENTIC_SEARCH_ENCRYPTION_KEY"


def _get_encryption_key() -> str | None:
    return os.environ.get(ENCRYPTION_KEY_ENV)


@lru_cache(maxsize=2)
def _get_trimmed_key(key: str) -> bytes:
    encoded = key.encode()
    if len(encoded) < 16:
        raise ValueError(
            f"{ENCRYPTION_KEY_ENV} is too short — minimum 16 bytes required."
        )
    for size in (32, 24, 16):
        if len(encoded) >= size:
            return encoded[:size]
    raise AssertionError("unreachable")


def _encrypt_string(input_str: str, key: str | None = None) -> bytes:
    effective_key = key if key is not None else _get_encryption_key()
    if not effective_key:
        return input_str.encode()

    trimmed = _get_trimmed_key(effective_key)
    iv = urandom(16)
    padder = sym_padding.PKCS7(algorithms.AES.block_size).padder()
    padded = padder.update(input_str.encode()) + padder.finalize()
    cipher = Cipher(algorithms.AES(trimmed), modes.CBC(iv), backend=default_backend())
    enc = cipher.encryptor()
    return iv + enc.update(padded) + enc.finalize()


def _decrypt_bytes(input_bytes: bytes, key: str | None = None) -> str:
    effective_key = key if key is not None else _get_encryption_key()
    if not effective_key:
        return input_bytes.decode()

    trimmed = _get_trimmed_key(effective_key)
    try:
        iv, ciphertext = input_bytes[:16], input_bytes[16:]
        cipher = Cipher(
            algorithms.AES(trimmed), modes.CBC(iv), backend=default_backend()
        )
        dec = cipher.decryptor()
        padded = dec.update(ciphertext) + dec.finalize()
        unpadder = sym_padding.PKCS7(algorithms.AES.block_size).unpadder()
        return (unpadder.update(padded) + unpadder.finalize()).decode()
    except (ValueError, UnicodeDecodeError):
        if key is not None:
            raise
        logger.warning(
            "AES decryption failed — falling back to raw UTF-8 decode. "
            "Run the re-encrypt secrets script to rotate to the current key."
        )
        try:
            return input_bytes.decode()
        except UnicodeDecodeError:
            raise ValueError(
                "Data is not valid UTF-8 and could not be decrypted with the "
                "current key. Likely encrypted with a different key."
            ) from None


def encrypt_string_to_bytes(input_str: str, key: str | None = None) -> bytes:
    """Encrypt *input_str* with the configured AES key. Pass-through if no key."""
    return _encrypt_string(input_str, key=key)


def decrypt_bytes_to_string(input_bytes: bytes, key: str | None = None) -> str:
    """Decrypt *input_bytes* to a string. Pass-through if no key."""
    return _decrypt_bytes(input_bytes, key=key)


def verify_encryption() -> None:
    """Round-trip self-test. Raises ``RuntimeError`` on mismatch."""
    test = "agentic-search-encryption-test"
    if decrypt_bytes_to_string(encrypt_string_to_bytes(test)) != test:
        raise RuntimeError("Encryption/decryption self-test failed.")


__all__ = [
    "ENCRYPTION_KEY_ENV",
    "decrypt_bytes_to_string",
    "encrypt_string_to_bytes",
    "verify_encryption",
]
