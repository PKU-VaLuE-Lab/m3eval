from __future__ import annotations

import re

VIDEO_READ_FAILURE_SENTINEL = "__VIDEO_INPUT_SKIPPED__"

_OPTION_LINE_RE = re.compile(r"^\s*([A-Z])[\.\)]\s+")
_STRICT_FINAL_ANSWER_RE = re.compile(r"(?m)^\s*The final answer is ([A-Z])(?:\s*[\.\!\?。！？]+)?\s*$")
_STRICT_FINAL_YES_NO_RE = re.compile(r"(?m)^\s*The final answer is (Yes|No)(?:\s*[\.\!\?。！？]+)?\s*$", flags=re.IGNORECASE)
_BARE_ANSWER_RE = re.compile(r"^\s*([A-Z])\s*$")
_BARE_YES_NO_RE = re.compile(r"^\s*(Yes|No)\s*$", flags=re.IGNORECASE)


def option_letters_from_options(options: list[str] | None) -> list[str]:
    letters: list[str] = []
    for option in options or []:
        match = _OPTION_LINE_RE.match(str(option))
        if not match:
            continue
        letter = match.group(1).upper()
        if letter not in letters:
            letters.append(letter)
    return letters or ["A", "B", "C", "D", "E"]


def parse_strict_final_answer(text: str, allowed_letters: list[str] | None = None) -> tuple[str, str]:
    allowed = {letter.upper() for letter in (allowed_letters or []) if str(letter).strip()}
    response = str(text or "").strip()
    matches = _STRICT_FINAL_ANSWER_RE.findall(response)
    if matches:
        answer = matches[-1].upper()
        if allowed and answer not in allowed:
            return "", "invalid_final_answer_option_letter"
        return answer, "ok"

    bare_match = _BARE_ANSWER_RE.fullmatch(response)
    if not bare_match:
        return "", "missing_required_final_answer_format"

    answer = bare_match.group(1).upper()
    if allowed and answer not in allowed:
        return "", "invalid_final_answer_option_letter"
    return answer, "ok_bare_answer"


def parse_strict_final_yes_no_answer(text: str) -> tuple[str, str]:
    response = str(text or "").strip()
    matches = _STRICT_FINAL_YES_NO_RE.findall(response)
    if matches:
        return matches[-1].capitalize(), "ok"

    bare_match = _BARE_YES_NO_RE.fullmatch(response)
    if not bare_match:
        return "", "missing_required_final_answer_format"
    return bare_match.group(1).capitalize(), "ok_bare_answer"


def parse_yes_no_answer(text: str) -> str:
    match = re.search(r"\b(yes|no)\b", str(text or "").strip(), flags=re.IGNORECASE)
    return match.group(1).capitalize() if match else ""
