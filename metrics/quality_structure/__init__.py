from typing import Dict
from ..extras import compute_extras
from ..coverage import compute_goal_alignment
from ..contradictions import compute_contradictions


def compute(md_text: str, feat_text: str, cov: Dict[str, object], model=None) -> Dict[str, float | bool]:
    out = compute_extras(md_text, feat_text, cov)
    res: Dict[str, float | bool] = {}
    for k in ("feature_header_valid", "duplicate_scenarios_ratio"):
        if k in out:
            res[k] = float(out[k])
    if model is not None:
        g = compute_goal_alignment(md_text, feat_text, model)
        res.update({
            "goal_sim": g.get("goal_sim", 0.0),
            "goal_hit": g.get("goal_hit", False),
        })
        contra_flag, _ = compute_contradictions(md_text, feat_text, model)
        res["contradictions"] = bool(contra_flag)
    return res



