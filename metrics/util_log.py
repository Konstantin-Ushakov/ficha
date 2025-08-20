from typing import Set

_used: Set[str] = set()


def log_once(name: str) -> None:
    """Print metric usage once per process run.

    Example output:
      [METRIC] using kw_hit_rate
    """
    try:
        if name not in _used:
            _used.add(name)
            print(f"[METRIC] using {name}")
    except Exception:
        # Silent on logging failures
        pass



