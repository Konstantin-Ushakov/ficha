from typing import Dict
from ..extras import compute_extras


def compute(md_text: str, feat_text: str, cov: Dict[str, object]) -> Dict[str, float]:
    out = compute_extras(md_text, feat_text, cov)
    res: Dict[str, float] = {}
    for k in ("per_scenario_alignment", "trace_density"):
        if k in out:
            res[k] = float(out[k])
    return res



