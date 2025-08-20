from typing import Dict

from ..extras import compute_extras
from ..multi_intent import compute_multi_intent_metrics


def compute_feature_definition(md_text: str, feat_text: str, cov: Dict[str, object]) -> Dict[str, float | bool]:
    out = compute_extras(md_text, feat_text, cov)
    result: Dict[str, float | bool] = {}
    if "terminology_consistency" in out:
        result["terminology_consistency"] = float(out["terminology_consistency"])  # 0..1
    if "fvi_mit" in out:
        result["fvi_mit"] = float(out["fvi_mit"])  # 0, 0.5, 1
    # Multi-intent is computed separately (boolean/ratios)
    try:
        mi = compute_multi_intent_metrics(md_text)
        result["multi_intent"] = bool(mi.get("multi_intent", False))
    except Exception:
        pass
    return result



