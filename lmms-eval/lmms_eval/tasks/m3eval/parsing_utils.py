"""Lightweight parsing utilities used by the public M3Eval release path."""

from __future__ import annotations

from lmms_eval.tasks.m3eval.m3eval_parsing import (
    VIDEO_READ_FAILURE_SENTINEL,
    option_letters_from_options,
    parse_strict_final_answer,
    parse_strict_final_yes_no_answer,
)

__all__ = [
    "VIDEO_READ_FAILURE_SENTINEL",
    "option_letters_from_options",
    "parse_strict_final_answer",
    "parse_strict_final_yes_no_answer",
]
