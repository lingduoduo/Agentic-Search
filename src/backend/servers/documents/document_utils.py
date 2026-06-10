from cryptography.hazmat.primitives.serialization import pkcs12

from src.backend.document_index.utils import setup_logger

logger = setup_logger(__name__)


def _is_password_related_error(error: Exception) -> bool:
    error_msg = str(error).lower()
    password_keywords = ["mac", "integrity", "password", "authentication", "verify"]
    return any(keyword in error_msg for keyword in password_keywords)


def validate_pkcs12_content(file_bytes: bytes) -> bool:
    """Validate that file_bytes is a valid PKCS#12 container (format check, not password check)."""
    try:
        if len(file_bytes) < 10:
            logger.debug("File too small to be a valid PKCS#12 file")
            return False

        if file_bytes[0] != 0x30:
            logger.debug("File does not start with ASN.1 SEQUENCE tag")
            return False

        try:
            pkcs12.load_key_and_certificates(file_bytes, password=None)
            return True
        except ValueError as e:
            if _is_password_related_error(e):
                logger.debug(
                    "PKCS#12 format appears valid, password-related error: %s", e
                )
                return True
            logger.debug("PKCS#12 format validation failed: %s", e)
            return False
        except Exception as e:
            try:
                pkcs12.load_key_and_certificates(file_bytes, password=b"")
                return True
            except ValueError as e2:
                if _is_password_related_error(e2):
                    logger.debug(
                        "PKCS#12 format appears valid with empty password attempt: %s",
                        e2,
                    )
                    return True
                logger.debug(
                    "PKCS#12 validation failed on both attempts: %s, %s", e, e2
                )
                return False
            except Exception:
                logger.debug("PKCS#12 validation failed: %s", e)
                return False

    except Exception as e:
        logger.debug("Unexpected error during PKCS#12 validation: %s", e)
        return False
