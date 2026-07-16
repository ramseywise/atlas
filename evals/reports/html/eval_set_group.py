"""Resolve eval_set / slice label for suite breakdown grouping."""

from __future__ import annotations


def group_key_from_meta(meta: dict | None, *, prefer_sentiment: bool = False) -> str:
    """Stable slice key — eval_set / reclass / source, or sentiment when requested."""
    meta = meta or {}

    turn_sent = meta.get("_turn_sentiment") or meta.get("rating")
    if prefer_sentiment and turn_sent in ("liked", "disliked"):
        return f"stratified_{turn_sent}"

    if es := meta.get("eval_set"):
        return str(es)

    rc = meta.get("overlap_classification") or meta.get("reclassified_as")
    if rc:
        return str(rc)

    src = (meta.get("source") or "").lower()
    if "regression" in src:
        return "regression"
    if "capability_escalation" in src or ("capability" in src and "escalation" in src):
        return "capability_escalation"
    if "capability_no_sources" in src or "no_sources" in src:
        return "capability_no_sources_unknown"
    if "capability" in src:
        return "capability_sources"
    if "edge" in src:
        return "edge_cases"

    if turn_sent in ("liked", "disliked"):
        return f"stratified_{turn_sent}"

    liked = meta.get("conv_like_flag") or meta.get("rating") == "liked"
    disliked = meta.get("conv_dislike_flag") or meta.get("rating") == "disliked"
    if liked and not disliked:
        return "stratified_liked"
    if disliked and not liked:
        return "stratified_disliked"

    return "unspecified"


_GROUP_LABELS = {
    "regression": "Regression",
    "capability_sources": "Capability — sources",
    "capability_escalation": "Capability — escalation",
    "capability_no_sources_unknown": "Capability — no source",
    "edge_cases": "Edge cases",
    "stratified_liked": "Cal sample — liked",
    "stratified_disliked": "Cal sample — disliked",
    "unspecified": "Unspecified slice",
    "unknown": "Unknown",
}


def group_display_name(key: str, *, cal_sample: bool = False) -> str:
    if key == "stratified_liked":
        return "Cal sample — liked" if cal_sample else "Rated — liked"
    if key == "stratified_disliked":
        return "Cal sample — disliked" if cal_sample else "Rated — disliked"
    return _GROUP_LABELS.get(key, key.replace("_", " ").title())


_GROUP_ORDER = [
    "regression",
    "stratified_liked",
    "stratified_disliked",
    "capability_sources",
    "capability_escalation",
    "capability_no_sources_unknown",
    "edge_cases",
    "grounded_regression",
    "capability_test",
    "edge_case",
    "verify_grounding",
]
