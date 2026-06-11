import re


def clean_text(text: str) -> str:
    """Remove null bytes and collapse excessive whitespace."""
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def shared_precompare_cleanup(text: str) -> str:
    """Normalize text for length-based offset comparison."""
    return text.rstrip()
