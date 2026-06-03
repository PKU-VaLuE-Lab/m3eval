from __future__ import annotations

import sys
from pathlib import Path

from lmms_eval.tasks.m3eval.m3eval_parsing import (
    VIDEO_READ_FAILURE_SENTINEL,
    option_letters_from_options,
    parse_strict_final_answer,
    parse_strict_final_yes_no_answer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PUBLIC_DATA_ROOT = PROJECT_ROOT / "data" / "m3eval"

DEFAULT_MC_POST_PROMPT = (
    "\n"
    "Given the question above with the provided choices, provide your answer directly without extra explanation.\n"
    "Your final line must be exactly:\n"
    "The final answer is X\n"
    "X must be one uppercase option letter.\n"
)

M3EVAL_MEMORY_INTERFERENCE_ACCURACY_METRICS = {
    "proactive": "m3eval_memory_interference_proactive_accuracy",
    "retroactive": "m3eval_memory_interference_retroactive_accuracy",
}

M3EVAL_MEMORY_INTERFERENCE_INTRUSION_METRICS = {
    "proactive": "m3eval_memory_interference_proactive_intrusion_rate",
    "retroactive": "m3eval_memory_interference_retroactive_intrusion_rate",
}

M3EVAL_SPLIT_SCREEN_SUBTYPE_METRICS = {
    ("source_memory", "spatial"): "m3eval_divided_attention_spatial_source_grounding_accuracy",
    ("storyline_reconstruction", "swap0", "source_confusion"): "m3eval_divided_attention_no_swap_source_identification_accuracy",
    ("storyline_reconstruction", "swap0", "order_disruption"): "m3eval_divided_attention_no_swap_order_understanding_accuracy",
    ("storyline_reconstruction", "swap0", "content_forgetting"): "m3eval_divided_attention_no_swap_content_retention_accuracy",
    ("storyline_reconstruction", "swap10", "source_confusion"): "m3eval_divided_attention_swap_source_identification_accuracy",
    ("storyline_reconstruction", "swap10", "order_disruption"): "m3eval_divided_attention_swap_order_understanding_accuracy",
    ("storyline_reconstruction", "swap10", "content_forgetting"): "m3eval_divided_attention_swap_content_retention_accuracy",
}

M3EVAL_INTERLEAVED_SUBTYPE_METRICS = {
    ("source_memory", "narrative"): "m3eval_interleaved_events_temporal_source_grounding_accuracy",
    ("false_source_attribution", "narrative"): "m3eval_interleaved_events_false_memory_discrimination_accuracy",
    ("storyline_reconstruction", "source_confusion"): "m3eval_interleaved_events_source_identification_accuracy",
    ("storyline_reconstruction", "order_disruption"): "m3eval_interleaved_events_order_understanding_accuracy",
    ("storyline_reconstruction", "content_forgetting"): "m3eval_interleaved_events_content_retention_accuracy",
}

M3EVAL_NBACK_ATTRIBUTE_METRICS = {
    "action": "m3eval_nback_action_accuracy",
    "scene": "m3eval_nback_scene_accuracy",
}

M3EVAL_NBACK_NVALUE_METRICS = {
    1: "m3eval_nback_n1_accuracy",
    2: "m3eval_nback_n2_accuracy",
    3: "m3eval_nback_n3_accuracy",
    4: "m3eval_nback_n4_accuracy",
}


def _require_public_data_root() -> Path:
    if PUBLIC_DATA_ROOT.exists():
        return PUBLIC_DATA_ROOT
    sys.exit(
        "M3Eval dataset not found. Expected unpacked data under data/m3eval/. "
        "Please follow README.md and run the dataset unpack step first."
    )


def _resolve_release_path(path_str: str) -> Path:
    path = Path(str(path_str or "").strip())
    if not str(path):
        sys.exit("Missing relative path in M3Eval sample.")

    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)

    public_root = _require_public_data_root()
    candidates.extend([public_root / path, PROJECT_ROOT / path])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    sys.exit(f"M3Eval asset not found: {path_str}")


def _mc_doc_to_text(doc, lmms_eval_specific_kwargs=None) -> str:
    kwargs = dict(lmms_eval_specific_kwargs or {})
    pre_prompt = str(kwargs.get("pre_prompt", "") or "")
    post_prompt = str(kwargs.get("post_prompt", "") or DEFAULT_MC_POST_PROMPT)

    question = str(doc.get("question", "")).strip()
    options = [str(option).strip() for option in doc.get("options", []) if str(option).strip()]
    if options:
        question += "\n" + "\n".join(options)
    return f"{pre_prompt}{question}{post_prompt}"


def public_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    video_rel = str(doc.get("video", "") or "").strip()
    if not video_rel:
        sys.exit(f"Missing 'video' field in doc: {doc.get('id', '?')}")
    return [str(_resolve_release_path(video_rel))]


def public_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _mc_doc_to_text(doc, lmms_eval_specific_kwargs=lmms_eval_specific_kwargs)


def _mc_score_and_pred(doc, results):
    pred = results[0]
    if str(pred).startswith(VIDEO_READ_FAILURE_SENTINEL):
        return None, ""

    allowed_letters = option_letters_from_options(doc.get("options", []))
    pred_answer, _ = parse_strict_final_answer(pred, allowed_letters=allowed_letters)
    gt_answer = str(doc.get("answer", "")).strip().upper()
    return bool(pred_answer) and pred_answer == gt_answer, pred_answer


def _mc_process_results(doc, results, metric_name: str):
    score, _ = _mc_score_and_pred(doc, results)
    return {metric_name: score}


def _append_score(scores: dict[str, object], metric_name: str | None, score: object) -> None:
    if metric_name:
        scores[metric_name] = score


def _memory_interference_subtype_scores(doc, score: object, intrusion_score: object) -> dict[str, object]:
    subtype_scores: dict[str, object] = {}
    direction = str(doc.get("paper_v4_metric_group") or doc.get("interference_direction") or "").strip()
    _append_score(subtype_scores, M3EVAL_MEMORY_INTERFERENCE_ACCURACY_METRICS.get(direction), score)
    _append_score(subtype_scores, M3EVAL_MEMORY_INTERFERENCE_INTRUSION_METRICS.get(direction), intrusion_score)
    return subtype_scores


def _memory_interference_intrusion_score(doc, pred_answer: str, score: object) -> object:
    if score is None:
        return None
    option_roles = doc.get("option_role_by_letter") or {}
    if isinstance(option_roles, dict):
        role = str(option_roles.get(pred_answer, "")).strip().lower()
        if role:
            return role == "intrusion"

    intrusion_letter = str(doc.get("intrusion_option_letter") or "").strip().upper()
    if intrusion_letter:
        return bool(pred_answer) and pred_answer == intrusion_letter
    return None


def _split_screen_subtype_scores(doc, score: object) -> dict[str, object]:
    subtype_scores: dict[str, object] = {}
    metric_detail = str(doc.get("paper_v4_metric_detail") or "").strip()
    metric_group = str(doc.get("paper_v4_metric_group") or "").strip()
    storyline_error_type = str(doc.get("storyline_error_type") or "").strip()
    source_dimension = str(doc.get("source_dimension") or "").strip()

    if metric_detail == "source_memory":
        _append_score(subtype_scores, M3EVAL_SPLIT_SCREEN_SUBTYPE_METRICS.get((metric_detail, source_dimension)), score)
    elif metric_detail == "storyline_reconstruction":
        _append_score(subtype_scores, M3EVAL_SPLIT_SCREEN_SUBTYPE_METRICS.get((metric_detail, metric_group, storyline_error_type)), score)
    return subtype_scores


def _interleaved_subtype_scores(doc, score: object) -> dict[str, object]:
    subtype_scores: dict[str, object] = {}
    metric_detail = str(doc.get("paper_v4_metric_detail") or "").strip()
    storyline_error_type = str(doc.get("storyline_error_type") or "").strip()
    source_dimension = str(doc.get("source_dimension") or "").strip()

    if metric_detail in {"source_memory", "false_source_attribution"}:
        _append_score(subtype_scores, M3EVAL_INTERLEAVED_SUBTYPE_METRICS.get((metric_detail, source_dimension)), score)
    elif metric_detail == "storyline_reconstruction":
        _append_score(subtype_scores, M3EVAL_INTERLEAVED_SUBTYPE_METRICS.get((metric_detail, storyline_error_type)), score)
    return subtype_scores


def m3eval_memory_interference_process_results(doc, results):
    score, pred_answer = _mc_score_and_pred(doc, results)
    intrusion_score = _memory_interference_intrusion_score(doc, pred_answer, score)
    scores = {"m3eval_memory_interference_accuracy": score}
    scores.update(_memory_interference_subtype_scores(doc, score, intrusion_score))
    return scores


def m3eval_split_screen_process_results(doc, results):
    scores = _mc_process_results(doc, results, "m3eval_divided_attention_accuracy")
    score = scores["m3eval_divided_attention_accuracy"]
    scores.update(_split_screen_subtype_scores(doc, score))
    return scores


def m3eval_interleaved_process_results(doc, results):
    scores = _mc_process_results(doc, results, "m3eval_interleaved_events_accuracy")
    score = scores["m3eval_interleaved_events_accuracy"]
    scores.update(_interleaved_subtype_scores(doc, score))
    return scores


def public_nback_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    del lmms_eval_specific_kwargs
    video_rel = str(doc.get("video", "") or "").strip()
    if not video_rel:
        sys.exit(f"Missing 'video' field in N-back doc: {doc.get('question_id', '?')}")
    return [str(_resolve_release_path(video_rel))]


def public_nback_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = dict(lmms_eval_specific_kwargs or {})
    pre_prompt = str(kwargs.get("pre_prompt", "") or "")
    post_prompt = str(
        kwargs.get("post_prompt", "")
        or (
            "\n"
            "Given the question above, provide your answer directly without extra explanation.\n"
            "Your final line must be exactly one of:\n"
            "The final answer is Yes\n"
            "The final answer is No\n"
        )
    )

    question = str(doc.get("question", "")).strip()
    options = [str(option).strip() for option in doc.get("options", []) if str(option).strip()]
    if options:
        question += "\n" + "\n".join(f"- {option}" for option in options)
    return f"{pre_prompt}{question}{post_prompt}"


def public_nback_process_results(doc, results):
    pred = results[0]
    if str(pred).startswith(VIDEO_READ_FAILURE_SENTINEL):
        return {"m3eval_nback_accuracy": None}

    pred_answer, _ = parse_strict_final_yes_no_answer(pred)
    gt_answer = str(doc.get("answer", "")).strip().capitalize()
    score = bool(pred_answer) and pred_answer == gt_answer
    scores = {"m3eval_nback_accuracy": score}

    attribute = str(doc.get("attribute") or "").strip()
    _append_score(scores, M3EVAL_NBACK_ATTRIBUTE_METRICS.get(attribute), score)

    try:
        n_value = int(doc.get("n_value"))
    except (TypeError, ValueError):
        n_value = None
    _append_score(scores, M3EVAL_NBACK_NVALUE_METRICS.get(n_value), score)
    return scores
