import base64
from enum import Enum
from typing import Protocol

from fastapi import HTTPException
from fastapi import UploadFile

from src.backend.servers.documents.document_utils import validate_pkcs12_content


class ProcessPrivateKeyFileProtocol(Protocol):
    def __call__(self, file: UploadFile) -> str:
        """Validate and return file contents as a base64-encoded string."""
        ...


class PrivateKeyFileTypes(Enum):
    SHAREPOINT_PFX_FILE = "sharepoint_pfx_file"


def process_sharepoint_private_key_file(file: UploadFile) -> str:
    """Validate a .pfx upload and return its contents as base64."""
    if not (file.filename and file.filename.lower().endswith(".pfx")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Only .pfx files are supported."
        )

    private_key_bytes = file.file.read()

    if not validate_pkcs12_content(private_key_bytes):
        raise HTTPException(
            status_code=400,
            detail="Invalid file content. The uploaded file does not appear to be a valid PKCS#12 (.pfx) file.",
        )

    return base64.b64encode(private_key_bytes).decode("ascii")


FILE_TYPE_TO_FILE_PROCESSOR: dict[
    PrivateKeyFileTypes, ProcessPrivateKeyFileProtocol
] = {
    PrivateKeyFileTypes.SHAREPOINT_PFX_FILE: process_sharepoint_private_key_file,
}
