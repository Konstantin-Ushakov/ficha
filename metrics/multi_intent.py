import os
import math
from typing import Any, Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util
import re


def _token_set(s: str) -> set[str]:
    """Tokenize into lowercase word-like tokens for simple heuristics."""
    return set(re.findall(r"[\w\-/]+", s.lower()))


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences; drop headers and trivial fragments."""
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


def looks_like_section_header(s: str) -> bool:
    """Heuristic: short "Word:" pattern treated as a section header."""
    if ":" in s:
        head = s.split(":", 1)[0]
        if 1 <= len(head) <= 40 and len(head.split()) <= 6:
            return True
    return False


def _normalize_vec(v):
    """L2-normalize a torch vector; return original if norm==0."""
    import torch

    n = v.norm()
    if n.item() == 0:
        return v
    return v / n


def cluster_sentences(sentences: List[str], model: SentenceTransformer, sim_threshold: float = 0.80):
    """Greedy clustering of sentence embeddings with cosine threshold."""
    if not sentences:
        return [], []
    import torch

    emb = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    clusters: List[List[int]] = []
    centroids: List = []
    for i in range(emb.size(0)):
        placed = False
        for cidx, cid in enumerate(clusters):
            rep = centroids[cidx]
            cs = float(util.cos_sim(emb[i].unsqueeze(0), rep.unsqueeze(0)).item())
            if cs >= sim_threshold:
                cid.append(i)
                members = [emb[j] for j in cid]
                centroids[cidx] = _normalize_vec(sum(members) / len(members))
                placed = True
                break
        if not placed:
            clusters.append([i])
            centroids.append(emb[i])
    return clusters, centroids


def compute_multi_intent_metrics(md_text: str, model: SentenceTransformer) -> Dict[str, Any]:
    """Detect multi-intent signals in .md: topic clusters, main cluster share, non-behavioral ratio."""
    sentences = _split_sentences(md_text)
    sentences = [s for s in sentences if len(_token_set(s)) >= 2]
    n = len(sentences)
    if n == 0:
        return {"topic_clusters": 0, "main_cluster_ratio": 0.0, "multi_intent": False, "non_behavioral_ratio": 0.0}
    clusters, centroids = cluster_sentences(sentences, model, sim_threshold=float(os.environ.get("TOPIC_SIM", "0.80")))
    sizes = [len(c) for c in clusters]
    if not sizes:
        return {"topic_clusters": 0, "main_cluster_ratio": 0.0, "multi_intent": False, "non_behavioral_ratio": 0.0}
    main_idx = int(max(range(len(sizes)), key=lambda i: sizes[i]))
    main_cent = centroids[main_idx]
    diverging_clusters = 0
    for i, cent in enumerate(centroids):
        if i == main_idx:
            continue
        import torch

        sim = float(util.cos_sim(main_cent.unsqueeze(0), cent.unsqueeze(0)).item())
        if sim <= float(os.environ.get("TOPIC_MAIN_SIM_MAX", "0.78")) and sizes[i] >= max(1, math.ceil(0.20 * n)):
            diverging_clusters += 1
    topic_clusters = 1 + diverging_clusters if diverging_clusters > 0 else 1
    main_cluster_ratio = sizes[main_idx] / max(1, n)

    tag_like = 0
    header_like = 0
    for s in sentences:
        if "@" in s:
            tag_like += 1
        if looks_like_section_header(s):
            header_like += 1
    non_behavioral_ratio = (tag_like + header_like) / max(1, n)

    multi_intent = (diverging_clusters >= 1) or (non_behavioral_ratio >= float(os.environ.get("NON_BEHAVIORAL_MAX", "0.30")))
    # Collect up to 2 examples from non-main clusters for explainability
    examples: List[str] = []
    try:
        for i, c in enumerate(clusters):
            if i == main_idx:
                continue
            for idx in c[:2]:
                if 0 <= idx < len(sentences):
                    examples.append(sentences[idx])
            if len(examples) >= 2:
                break
    except Exception:
        examples = []
    return {
        "topic_clusters": topic_clusters,
        "main_cluster_ratio": main_cluster_ratio,
        "multi_intent": bool(multi_intent),
        "non_behavioral_ratio": non_behavioral_ratio,
        "examples": examples[:2],
    }


