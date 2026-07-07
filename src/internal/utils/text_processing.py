import re

_MATCH_WS = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    """Lowercase, trim surrounding punctuation/space, collapse inner whitespace."""
    text = (text or "").strip().lower().strip(".,!?;:\"'()[]{}").strip()
    return _MATCH_WS.sub(" ", text)


def clean_text(text: str) -> str:
    """Remove null bytes and collapse excessive whitespace."""
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def shared_precompare_cleanup(text: str) -> str:
    """Normalize text for length-based offset comparison."""
    return text.rstrip()


def levenshtein_lt2(a: str, b: str) -> bool:
    """True iff the Levenshtein distance between a and b is 0 or 1 (bounded, O(n))."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    if la > lb:
        a, b = b, a  # ensure a is the shorter string
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            if skipped:
                return False
            skipped = True
            j += 1  # consume one extra char from the longer string
    return True
