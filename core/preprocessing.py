import re
from typing import List


def _split_sentences(text: str) -> List[str]:
    """Generic sentence splitter used by core preprocessing helpers."""
    parts: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            line = line.lstrip("# ").strip()
        sentences = re.split(r"(?<=[.!?])\s+", line)
        for s in sentences:
            s = s.strip(" -•\t")
            if len(s) >= 3:
                parts.append(s)
    return parts


def split_requirements(md_text: str) -> List[str]:
    """Split requirements document into sentence-like units."""
    return _split_sentences(md_text)


def split_feature_units(feat_text: str) -> List[str]:
    """Extract sentence units from feature file text, stripping Gherkin keywords and tags."""
    cleaned: List[str] = []
    for raw in feat_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        first_non_ws = raw.lstrip()[:1]
        if first_non_ws in ("#", "@"):
            continue
        line = re.sub(r"^(Функция:|Сценарий:|Предыстория:|Given|When|Then|And|But|Допустим|Когда|Тогда|И)\s*", "", line, flags=re.IGNORECASE)
        if len(line) >= 3:
            cleaned.append(line)
    out: List[str] = []
    for l in cleaned:
        out.extend(_split_sentences(l))
    return out


