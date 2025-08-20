from pathlib import Path
import os
import glob


def load_text(p: Path) -> str:
    """Read UTF-8 text from path."""
    return p.read_text(encoding="utf-8")


def concat_features(feature_files: list[Path]) -> str:
    """Concatenate multiple feature files with double newlines, preserving order."""
    return "\n\n".join(load_text(p) for p in feature_files)


def parse_max_category() -> int | None:
    """Parse TEST_CATEGORY env (supports single digit 0-9 or till-X) into integer upper bound."""
    cat = os.environ.get("TEST_CATEGORY", "").strip()
    if not cat:
        return None
    if cat.isdigit() and len(cat) == 1:
        return int(cat)
    if cat.startswith("till-") and cat.split("-", 1)[-1].isdigit():
        try:
            return int(cat.split("-", 1)[-1])
        except Exception:
            return None
    return None


