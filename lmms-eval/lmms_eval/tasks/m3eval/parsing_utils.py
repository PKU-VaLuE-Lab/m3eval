"""Local multiple-choice parsing used by the frozen M3Eval paper profile.

Legacy empty diagnostic fields are retained for compatibility with archived
result records. No external answer resolver is available in this release.
"""

from __future__ import annotations

import re

_OPTION_LINE_RE = re.compile(r"^\s*([A-Z])[\.\)]\s+")
_THINK_END = "</think>"
_STRICT_FINAL_ANSWER_RE = re.compile(r"(?m)^\s*The final answer is ([A-Z])(?:\s*[\.\!\?。！？]+)?\s*$")
VIDEO_READ_FAILURE_SENTINEL = "__VIDEO_INPUT_SKIPPED__"

STRICT_FINAL_ANSWER_OK = "ok"
STRICT_FINAL_ANSWER_MISSING = "missing_required_final_answer_format"
STRICT_FINAL_ANSWER_INVALID = "invalid_final_answer_option_letter"

ANSWER_RESOLUTION_STRICT = "strict_final_answer"
ANSWER_RESOLUTION_SINGLE_LETTER = "single_uppercase_letter_only"
ANSWER_RESOLUTION_SKIPPED_BAD_VIDEO = "skipped_bad_video"


def _empty_deepseek_trace() -> dict[str, object]:
    return {
        "deepseek_input_messages": [],
        "deepseek_raw_output": "",
        "deepseek_error": "",
    }


def strip_think_content(text: str) -> str:
    s = (text or "").strip()
    think_end = s.rfind(_THINK_END)
    if think_end != -1:
        s = s[think_end + len(_THINK_END) :].strip()
    return s


def option_letters_from_options(options: list[str] | None) -> list[str]:
    letters: list[str] = []
    for opt in options or []:
        match = _OPTION_LINE_RE.match(str(opt))
        if match:
            letter = match.group(1).upper()
            if letter not in letters:
                letters.append(letter)
    if letters:
        return letters
    return ["A", "B", "C", "D", "E"]


def _letter_pattern(allowed_letters: list[str] | None) -> str:
    letters = [x.upper() for x in (allowed_letters or ["A", "B", "C", "D", "E"]) if x]
    if not letters:
        letters = ["A", "B", "C", "D", "E"]
    escaped = "".join(re.escape(ch) for ch in letters)
    return f"[{escaped}]"


def extract_option_letter(text: str, allowed_letters: list[str] | None = None) -> str:
    s = strip_think_content(text)
    pat_letter = _letter_pattern(allowed_letters)
    wrapped_letter = rf"[<\(\[]?\s*({pat_letter})\s*[>\)\]]?"

    patterns = [
        rf"Answer\s+and\s+Confidence\s*\(0-100\)\s*:\s*{wrapped_letter}\s*,\s*\d{{1,3}}(?:\.\d+)?\s*%?",
        rf"FINAL_ANSWER\s*[:=]\s*{wrapped_letter}",
        rf"[Tt]he\s+final\s+answer\s+is\s*:?\s*{wrapped_letter}",
        rf"[Tt]he\s+(?:best|correct)\s+answer\s+is\s*:?\s*{wrapped_letter}",
        rf"[Tt]he\s+answer\s+is\s*:?\s*{wrapped_letter}",
        rf"(?:Best|Correct)\s+(?:answer|option)\s*:\s*{wrapped_letter}",
    ]

    for pat in patterns:
        matches = re.findall(pat, s)
        if matches:
            return matches[-1].upper()

    # Fallback 1: explicit option prefix, e.g. "B. ..."
    option_prefix_matches = re.findall(rf"(?m)^\s*({pat_letter})[\.\)]\s+", s, flags=re.IGNORECASE)
    if option_prefix_matches:
        return option_prefix_matches[-1].upper()

    # Fallback 2: isolated single-letter line.
    single_line_matches = re.findall(rf"(?m)^\s*({pat_letter})\s*$", s, flags=re.IGNORECASE)
    if single_line_matches:
        return single_line_matches[-1].upper()

    # Fallback 3: conservative isolated single-letter token.
    token_matches = re.findall(rf"(?<![A-Z0-9])({pat_letter})(?![A-Z0-9])", s, flags=re.IGNORECASE)
    if token_matches:
        return token_matches[-1].upper()
    return ""


def extract_strict_final_answer_letter(
    text: str,
    allowed_letters: list[str] | None = None,
) -> tuple[str, str]:
    s = strip_think_content(text)
    matches = _STRICT_FINAL_ANSWER_RE.findall(s)
    if not matches:
        return "", STRICT_FINAL_ANSWER_MISSING

    answer = matches[-1].upper()
    allowed = [x.upper() for x in (allowed_letters or []) if x]
    if allowed and answer not in allowed:
        return "", STRICT_FINAL_ANSWER_INVALID
    return answer, STRICT_FINAL_ANSWER_OK


def extract_single_uppercase_letter_letter(
    text: str,
    allowed_letters: list[str] | None = None,
) -> tuple[str, str]:
    s = strip_think_content(text)
    if not re.fullmatch(r"[A-Z]", s):
        return "", STRICT_FINAL_ANSWER_MISSING

    answer = s.upper()
    allowed = [x.upper() for x in (allowed_letters or []) if x]
    if allowed and answer not in allowed:
        return "", STRICT_FINAL_ANSWER_INVALID
    return answer, STRICT_FINAL_ANSWER_OK


def _clamp_confidence(value: float) -> int:
    return max(0, min(100, int(round(value))))


def extract_confidence_percent(text: str) -> int | None:
    s = strip_think_content(text)
    if not s:
        return None

    patterns = [
        r"Answer\s+and\s+Confidence\s*\(0-100\)\s*:\s*[<\(\[]?\s*[A-Z]\s*[>\)\]]?\s*,\s*(\d{1,3}(?:\.\d+)?)\s*%?",
        r"CONFIDENCE\s*[:=]\s*(\d{1,3}(?:\.\d+)?)\s*%?",
        r"\bconfidence(?:\s+level)?\s*(?:is|:)?\s*(\d{1,3}(?:\.\d+)?)\s*%",
        r"\bconfidence(?:\s+level)?\s*(?:is|:)\s*(\d{1,3}(?:\.\d+)?)\b",
        r"(\d{1,3}(?:\.\d+)?)\s*%\s*confidence\b",
    ]

    for pat in patterns:
        matches = re.findall(pat, s, flags=re.IGNORECASE)
        if matches:
            return _clamp_confidence(float(matches[-1]))

    # fallback: the last plausible percentage in text
    percent_matches = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", s)
    if percent_matches:
        return _clamp_confidence(float(percent_matches[-1]))
    return None


def parse_mc_response(
    text: str,
    allowed_letters: list[str] | None = None,
    *,
    strict_final_answer: bool = False,
) -> tuple[str, int | None]:
    parsed = parse_mc_response_with_details(
        text,
        allowed_letters=allowed_letters,
        strict_final_answer=strict_final_answer,
    )
    return str(parsed.get("answer", "") or ""), parsed.get("confidence")


def parse_mc_response_with_details(
    text: str,
    allowed_letters: list[str] | None = None,
    *,
    strict_final_answer: bool = False,
) -> dict[str, object]:
    confidence = extract_confidence_percent(text)

    if not strict_final_answer:
        answer = extract_option_letter(text, allowed_letters=allowed_letters)
        format_status = STRICT_FINAL_ANSWER_OK if answer else "unparsed"
        return {
            "answer": answer,
            "confidence": confidence,
            "format_status": format_status,
            "format_compliant": format_status == STRICT_FINAL_ANSWER_OK,
            "resolution_status": "parsed" if answer else "unparsed",
            "resolution_success": bool(answer),
            "deepseek_checked": False,
            "resolution_evidence": "",
            **_empty_deepseek_trace(),
        }

    strict_answer, format_status = extract_strict_final_answer_letter(text, allowed_letters=allowed_letters)
    if strict_answer:
        return {
            "answer": strict_answer,
            "confidence": confidence,
            "format_status": format_status,
            "format_compliant": True,
            "resolution_status": ANSWER_RESOLUTION_STRICT,
            "resolution_success": True,
            "deepseek_checked": False,
            "resolution_evidence": f"The final answer is {strict_answer}",
            **_empty_deepseek_trace(),
        }

    if format_status == STRICT_FINAL_ANSWER_INVALID:
        return {
            "answer": "",
            "confidence": confidence,
            "format_status": format_status,
            "format_compliant": False,
            "resolution_status": STRICT_FINAL_ANSWER_INVALID,
            "resolution_success": False,
            "deepseek_checked": False,
            "resolution_evidence": "",
            **_empty_deepseek_trace(),
        }

    single_answer, single_status = extract_single_uppercase_letter_letter(text, allowed_letters=allowed_letters)
    if single_answer:
        return {
            "answer": single_answer,
            "confidence": confidence,
            "format_status": format_status,
            "format_compliant": False,
            "resolution_status": ANSWER_RESOLUTION_SINGLE_LETTER,
            "resolution_success": True,
            "deepseek_checked": False,
            "resolution_evidence": single_answer,
            **_empty_deepseek_trace(),
        }

    if single_status == STRICT_FINAL_ANSWER_INVALID:
        return {
            "answer": "",
            "confidence": confidence,
            "format_status": format_status,
            "format_compliant": False,
            "resolution_status": STRICT_FINAL_ANSWER_INVALID,
            "resolution_success": False,
            "deepseek_checked": False,
            "resolution_evidence": "",
            **_empty_deepseek_trace(),
        }

    if strict_final_answer:
        answer = ""
    else:
        answer = extract_option_letter(text, allowed_letters=allowed_letters)
        format_status = STRICT_FINAL_ANSWER_OK if answer else "unparsed"

    return {
        "answer": answer,
        "confidence": confidence,
        "format_status": format_status,
        "format_compliant": format_status == STRICT_FINAL_ANSWER_OK,
        "resolution_status": format_status if not answer else ANSWER_RESOLUTION_STRICT,
        "resolution_success": False,
        "deepseek_checked": False,
        "resolution_evidence": "",
        **_empty_deepseek_trace(),
    }
