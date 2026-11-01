from pathlib import Path

from pra_core.documents.filenames import normalize_filename
from pra_core.documents.models import ParsedDocument


class UnsupportedDocumentTypeError(ValueError):
    pass


class DocumentParser:
    """Phase 1 parser for plain text and markdown uploads."""

    _SUPPORTED_CONTENT_TYPES = {
        "text/plain",
        "text/markdown",
        "application/octet-stream",
    }
    _SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}

    def parse(self, content: bytes, filename: str, content_type: str) -> ParsedDocument:
        normalized_filename = normalize_filename(filename)
        suffix = Path(normalized_filename).suffix.lower()
        if (
            content_type not in self._SUPPORTED_CONTENT_TYPES
            and suffix not in self._SUPPORTED_SUFFIXES
        ):
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type: {content_type}"
            )

        text = content.decode("utf-8").strip()
        if not text:
            raise ValueError("Uploaded document is empty.")

        return ParsedDocument(
            filename=normalized_filename,
            content_type=content_type,
            text=text,
        )
