import re
import unicodedata


HTML_RE = re.compile(r"<[^>]+>")
SYMBOL_RE = re.compile(r"[^\w\s\.,!?;:'\"()\-\n]")
REPEAT_TOKEN_RE = re.compile(r"\b(\w+)(\s+\1\b){2,}", flags=re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")


def normalize_text_basic(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def normalize_text_aggressive(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = HTML_RE.sub(" ", text)
    text = SYMBOL_RE.sub(" ", text)
    text = REPEAT_TOKEN_RE.sub(r"\1", text)
    text = text.replace("\u2026", "...")
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text

