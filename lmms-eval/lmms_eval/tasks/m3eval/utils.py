"""M3Eval prompt rendering and local scoring for the frozen paper profile."""

import os
import sys
from pathlib import Path

from lmms_eval.tasks.m3eval.parsing_utils import (
    VIDEO_READ_FAILURE_SENTINEL,
    option_letters_from_options,
    parse_mc_response_with_details,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
PUBLIC_DATA_ROOT_ENV_KEY = "VIDEO_MEMORY_EVAL_DATA_ROOT"
DEFAULT_PUBLIC_DATA_ROOT = Path("data") / "m3eval"
V4_CONFIDENCE_ENV_KEY = "VIDEO_MEMORY_EVAL_CONFIDENCE"
V4_FORMAT_MODE_ENV_KEY = "VIDEO_MEMORY_EVAL_FORMAT_MODE"


def _resolve_public_data_root() -> Path:
    raw = os.getenv(PUBLIC_DATA_ROOT_ENV_KEY, "").strip()
    if not raw:
        return PROJECT_ROOT / DEFAULT_PUBLIC_DATA_ROOT

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _resolve_v4_video_path(video_rel: str, video_base: str | None = None) -> Path:
    path = Path(str(video_rel or "").strip())
    candidates: list[Path] = []

    if path.is_absolute():
        candidates.append(path)

    if video_base:
        base = Path(str(video_base))
        if base.is_absolute():
            candidates.append(base / path)
        else:
            candidates.append(PROJECT_ROOT / base / path)

    # Released videos are relative to the dataset root.
    candidates.append(_resolve_public_data_root() / path)

    # Also support paths relative to the project root.
    candidates.append(PROJECT_ROOT / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    sys.exit(f"Video not found: {video_rel}")


def v4_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    """Resolve a released video without changing its bytes or frame sampling."""
    video_rel = doc.get("video", "")
    if not video_rel:
        sys.exit(f"Missing 'video' field in doc: {doc.get('id', '?')}")
    video_path = _resolve_v4_video_path(video_rel, doc.get("video_base"))
    return [str(video_path)]


def _resolve_confidence_mode(kwargs: dict | None) -> str:
    # Prefer the environment, then configuration; default to off.
    runtime_mode = os.getenv(V4_CONFIDENCE_ENV_KEY, "").strip().lower()
    if runtime_mode in {"on", "off"}:
        return runtime_mode

    kwargs = kwargs or {}
    cfg_mode = str(kwargs.get("confidence_mode", "off")).strip().lower()
    return "on" if cfg_mode == "on" else "off"


def _resolve_format_mode(kwargs: dict | None) -> str:
    # Prefer the environment, then configuration; default to strict_final_line.
    runtime_mode = os.getenv(V4_FORMAT_MODE_ENV_KEY, "").strip().lower()
    if runtime_mode in {"legacy", "strict_final_line", "tagged"}:
        return runtime_mode

    kwargs = kwargs or {}
    cfg_mode = str(kwargs.get("answer_format_mode", "strict_final_line")).strip().lower()
    if cfg_mode in {"legacy", "strict_final_line", "tagged"}:
        return cfg_mode
    return "strict_final_line"


def _build_v4_post_prompt(kwargs: dict | None) -> str:
    # Build the suffix from the response, confidence, and format settings.
    kwargs = kwargs or {}
    response_style = str(kwargs.get("response_style", "direct")).strip().lower()
    confidence_mode = _resolve_confidence_mode(kwargs)
    format_mode = _resolve_format_mode(kwargs)

    if response_style == "cot":
        base_instruction = "Given the question above with the provided choices, reason step by step, then provide your final answer."
    else:
        base_instruction = "Given the question above with the provided choices, provide your answer directly without extra explanation."

    if format_mode == "tagged":
        if confidence_mode == "on":
            return (
                "\n"
                f"{base_instruction}\n"
                "Output format is mandatory.\n"
                "Your final two lines must be exactly:\n"
                "FINAL_ANSWER: X\n"
                "CONFIDENCE: NN\n"
                "X must be one uppercase option letter.\n"
                "NN must be an integer from 0 to 100.\n"
                "Do not output any text after these two lines.\n"
            )
        return "\n" f"{base_instruction}\n" "Output format is mandatory.\n" "Your final line must be exactly:\n" "FINAL_ANSWER: X\n" "X must be one uppercase option letter.\n" "Do not output any text after this line.\n"

    if format_mode == "strict_final_line":
        if confidence_mode == "on":
            return (
                "\n"
                f"{base_instruction}\n"
                "Please provide confidence level (0-100).\n"
                "Output format is mandatory.\n"
                "Your final three lines must be exactly:\n"
                "X\n"
                "The final answer is X\n"
                "Answer and Confidence (0-100): X, NN%\n"
                "X must be one uppercase option letter.\n"
                "NN must be an integer from 0 to 100.\n"
                "Do NOT use angle brackets around X.\n"
                "Do not output any text after these lines.\n"
            )
        return (
            "\n"
            f"{base_instruction}\n"
            "Output format is mandatory.\n"
            "Your final two lines must be exactly:\n"
            "X\n"
            "The final answer is X\n"
            "X must be one uppercase option letter.\n"
            "Do NOT use angle brackets around X.\n"
            "Do not output any text after these lines.\n"
        )

    # legacy
    if confidence_mode == "on":
        return (
            "\n"
            f"{base_instruction}\n"
            "Please also provide your confidence level (0-100%) that your answer is correct.\n"
            "Use this exact final line format:\n"
            "Answer and Confidence (0-100): <LETTER>, <NUMBER>%\n"
            "<LETTER> must be a single uppercase option letter.\n"
            "<NUMBER> must be an integer from 0 to 100.\n"
        )

    return "\n" f"{base_instruction}\n" "End your response with this exact final line:\n" "The final answer is <LETTER>\n" "<LETTER> must be a single uppercase option letter.\n"


def _build_v4_dynamic_pre_prompt(doc) -> str:
    task_type = str(doc.get("task_type", "")).strip()
    if task_type == "memory_interference":
        return (
            "Note: This video contains two separate videos played one after the other. "
            "Video 1 is played first, followed by Video 2.\n"
            "If the question mentions Video 1 or Video 2, they refer to the first and second video respectively.\n\n"
        )

    if task_type == "split_screen":
        prompt_version = str(doc.get("pre_prompt_version", "")).strip().lower()
        swap_count = int(doc.get("swap_count", 0) or 0)
        if prompt_version == "ss_swap_low_v1":
            return "Note: This video displays two separate videos side by side. " "At the beginning, Video 1 appears on the left and Video 2 appears on the right.\n\n"
        if prompt_version == "ss_swap_v1" or (not prompt_version and swap_count > 0):
            return (
                "Note: This video displays two separate videos side by side. "
                "At the beginning, Video 1 appears on the left and Video 2 appears on the right. "
                "Their positions may swap during playback. "
                "If the question mentions Video 1 or Video 2, keep tracking the underlying videos rather than their current position.\n\n"
            )
        return (
            "Note: This video displays two separate videos side by side. "
            "Video 1 is on the left and Video 2 is on the right throughout the video.\n"
            "If the question mentions Video 1 or Video 2, they refer to the left and right videos respectively.\n\n"
        )

    if task_type == "interleaved":
        prompt_version = str(doc.get("pre_prompt_version", "")).strip().lower()
        hint_mode = str(doc.get("hint_mode", "")).strip().lower()
        if prompt_version == "inter_source_nback_v1":
            return (
                "Note: This video is a single stream composed of segments drawn from multiple underlying source videos. "
                "Segment numbers always refer to the order in which segments appear during playback. "
                "When you are asked whether two segments come from the same source video, judge whether they originate from the same underlying video.\n\n"
            )
        if prompt_version == "inter_scene_nback_v1":
            return (
                "Note: This video is a single stream composed of segments drawn from multiple underlying source videos. "
                "Segment numbers always refer to the order in which segments appear during playback. "
                "When you are asked whether two segments take place in the same type of scene or environment, "
                "judge whether their dominant setting or environment type matches.\n\n"
            )
        if prompt_version == "inter_low_v1" or (not prompt_version and hint_mode == "none"):
            return "Note: This clip contains content from two videos. " "Video 1 refers to the video that appears first, and Video 2 refers to the one that appears second.\n\n"
        if prompt_version == "inter_full_v1" or (not prompt_version and hint_mode == "full"):
            segment_count = int(doc.get("segment_count", 0) or 0) or 4
            return (
                "Note: This video is composed by interleaving two separate videos. "
                f"The original two videos were each divided into {segment_count} segments by duration and then alternately concatenated together "
                "(A1, B1, A2, B2, ..., AN, BN). "
                "Please pay attention to distinguish content from different videos.\n"
                "If the question mentions Video 1 or Video 2: Video 1 refers to the video that appears first in the interleaved sequence; "
                "Video 2 refers to the one that appears second.\n\n"
            )
    return ""


def _v4_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # Combine the preamble, question/options, and answer-format suffix.
    kwargs = lmms_eval_specific_kwargs or {}
    dynamic_pre = _build_v4_dynamic_pre_prompt(doc)
    pre = dynamic_pre if dynamic_pre else kwargs.get("pre_prompt", "")
    post = _build_v4_post_prompt(kwargs)
    question = doc.get("question", "")
    for opt in doc.get("options", []):
        question += "\n" + opt
    return f"{pre}{question}{post}"


def v4_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    kwargs = dict(lmms_eval_specific_kwargs or {})
    kwargs.setdefault("response_style", "cot")
    return _v4_doc_to_text(doc, kwargs)


def v4_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = dict(lmms_eval_specific_kwargs or {})
    kwargs.setdefault("response_style", "direct")
    return _v4_doc_to_text(doc, kwargs)


def _v4_process_with_metrics(doc, results, metric_map):
    # Return the task score, parsed answer/confidence, and output-format status.
    pred = results[0]
    task_type = doc.get("task_type", "")
    metric_name = metric_map.get(task_type, "v4_process_results")
    if str(pred).startswith(VIDEO_READ_FAILURE_SENTINEL):
        return {
            metric_name: None,
            metric_name.replace("_score", "_missing_required_answer_format"): None,
            "v4_parsed_answer": "",
            "v4_parsed_confidence": -1,
            "v4_answer_format_status": "skipped_bad_video",
            "v4_answer_format_compliant": False,
            "v4_answer_resolution_status": "skipped_bad_video",
            "v4_answer_resolution_success": False,
            "v4_deepseek_checked": False,
            "v4_answer_resolution_evidence": "",
            "v4_deepseek_input_messages": [],
            "v4_deepseek_raw_output": "",
            "v4_deepseek_error": "",
            "v4_confidence_mode": _resolve_confidence_mode(None),
            "v4_skipped_due_to_bad_video": True,
            "v4_skip_reason": "video_read_failure",
        }

    allowed_letters = option_letters_from_options(doc.get("options", []))
    parsed = parse_mc_response_with_details(
        pred,
        allowed_letters=allowed_letters,
        strict_final_answer=True,
    )
    pred_ans = str(parsed.get("answer", "") or "")
    pred_conf = parsed.get("confidence")
    format_status = str(parsed.get("format_status", "") or "")
    format_compliant = bool(parsed.get("format_compliant", False))
    resolution_status = str(parsed.get("resolution_status", "") or "")
    resolution_success = bool(parsed.get("resolution_success", False))
    deepseek_checked = bool(parsed.get("deepseek_checked", False))
    resolution_evidence = str(parsed.get("resolution_evidence", "") or "")
    deepseek_input_messages = list(parsed.get("deepseek_input_messages") or [])
    deepseek_raw_output = str(parsed.get("deepseek_raw_output", "") or "")
    deepseek_error = str(parsed.get("deepseek_error", "") or "")
    gt_ans = str(doc.get("answer", "")).strip().upper()
    score = resolution_success and pred_ans == gt_ans

    return {
        metric_name: score,
        metric_name.replace("_score", "_missing_required_answer_format"): format_status == "missing_required_final_answer_format",
        "v4_parsed_answer": pred_ans,
        "v4_parsed_confidence": pred_conf if resolution_success and pred_conf is not None else -1,
        "v4_answer_format_status": format_status,
        "v4_answer_format_compliant": format_compliant,
        "v4_answer_resolution_status": resolution_status,
        "v4_answer_resolution_success": resolution_success,
        "v4_deepseek_checked": deepseek_checked,
        "v4_answer_resolution_evidence": resolution_evidence,
        "v4_deepseek_input_messages": deepseek_input_messages,
        "v4_deepseek_raw_output": deepseek_raw_output,
        "v4_deepseek_error": deepseek_error,
        "v4_confidence_mode": _resolve_confidence_mode(None),
        "v4_skipped_due_to_bad_video": False,
    }


def v4_process_results_cot(doc, results):
    metric_map = {
        "memory_interference": "v4_memory_interference_cot_score",
        "interleaved": "v4_interleaved_cot_score",
        "split_screen": "v4_split_screen_cot_score",
    }
    return _v4_process_with_metrics(doc, results, metric_map)


def v4_process_results(doc, results):
    metric_map = {
        "memory_interference": "v4_memory_interference_score",
        "interleaved": "v4_interleaved_score",
        "split_screen": "v4_split_screen_score",
    }
    return _v4_process_with_metrics(doc, results, metric_map)
