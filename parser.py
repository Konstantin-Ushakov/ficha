import re
from pathlib import Path
from typing import Tuple, List


def extract_goal(md_text: str) -> str:
    """Return text after first line starting with 'Цель:' (trimmed) or empty string."""
    for raw in md_text.splitlines():
        m = re.match(r"^\s*Цель:\s*(.+)$", raw.strip())
        if m:
            return m.group(1).strip()
    return ""


def extract_acceptance_text(md_text: str) -> str:
    """Return content of section starting with '## Описание фичи' until next '##' or end.

    Fallback: return full document minus goal line if section not found.
    """
    lines = md_text.splitlines()
    # remove first goal line from body when falling back
    goal_found = False
    body_lines: List[str] = []
    for raw in lines:
        if not goal_found and re.match(r"^\s*Цель:\s*", raw.strip()):
            goal_found = True
            continue
        body_lines.append(raw)

    start = None
    end = None
    for i, raw in enumerate(lines):
        if re.match(r"^\s*##\s+Критерии\s+приемки\b", raw.strip(), flags=re.IGNORECASE):
            start = i + 1
            break
    if start is not None:
        for j in range(start, len(lines)):
            if re.match(r"^\s*##\s+", lines[j].strip()):
                end = j
                break
        return "\n".join(lines[start:end]) if end is not None else "\n".join(lines[start:])
    return "\n".join(body_lines)


def extract_feature_paths(md_text: str, repo_root: Path) -> List[Path]:
    """Parse '## Связь с файлами' -> '### Реализация' and return absolute Paths to .feature files.

    If section not found, return empty list.
    """
    lines = md_text.splitlines()
    start_sf = None
    for i, raw in enumerate(lines):
        if raw.strip().lower().startswith("## связь с файлами"):
            start_sf = i
            break
    if start_sf is None:
        return []
    start_impl = None
    for j in range(start_sf + 1, len(lines)):
        t = lines[j].strip().lower()
        if t.startswith("### реализация"):
            start_impl = j + 1
            break
        if t.startswith("## "):
            return []
    if start_impl is None:
        return []
    out: List[Path] = []
    for k in range(start_impl, len(lines)):
        s = lines[k].strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("## ") or low.startswith("### "):
            break
        m = re.search(r"[\-*•]\s+(.+?\.feature)\b", s)
        if m:
            rel = m.group(1).strip()
            p = (repo_root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
            out.append(p)
    return out


def parse_md(md_path: Path, repo_root: Path) -> Tuple[str, str, List[Path]]:
    """Return (goal, acceptance_text, feature_paths) for given .md file.

    The parser follows the new structure described in feature-description: goal by prefix,
    acceptance by section, feature files by mapping section.
    """
    text = md_path.read_text(encoding="utf-8")
    goal = extract_goal(text)
    acceptance = extract_acceptance_text(text)
    feats = extract_feature_paths(text, repo_root=repo_root)
    return goal, acceptance, feats



