import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from internvl_local_backend import call_internvl_batched_yes_no
from nback_eval_utils import (
    build_balanced_test_plan,
    build_random_test_plan,
    extract_test_order_from_selection_manifest,
    normalize_n_back_key,
    parse_batched_n_back_response,
    parse_yes_no_response,
    resolve_nback_video_path,
)
from PIL import Image
from qwen_vl_utils import fetch_video

# =========================
# Basic configuration
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/m3eval/videos/nback_paper_short_segments"
MIN_VIDEO_FILE_SIZE_BYTES = int(os.getenv("MIN_VIDEO_FILE_SIZE_BYTES", "10000") or "10000")


def _parse_choice_env(name: str, default: str, allowed: set[str]) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in allowed:
        return value
    raise ValueError(f"Unsupported {name}={raw!r}. Allowed: {sorted(allowed)}")


def _sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text).strip())
    return cleaned.strip("._-") or "run"


def _parse_optional_text_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _parse_int_list_env(name: str) -> list[int]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return []
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


MODEL_NAME = os.getenv("MODEL_NAME", "InternVL3.5-8B")
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "internvl_local").strip().lower()
if MODEL_BACKEND != "internvl_local":
    raise ValueError("This release supports only MODEL_BACKEND=internvl_local")
JSON_PATH = os.getenv("JSON_PATH", str(SCRIPT_DIR / "nback_test_action.json"))
DATA_ROOT = Path(os.getenv("NBACK_DATA_ROOT", str(DEFAULT_DATA_ROOT))).resolve()
RUN_NAME = os.getenv("RUN_NAME", _sanitize_name(MODEL_NAME))
RESULTS_DIR = Path(os.getenv("NBACK_RESULTS_DIR", str(SCRIPT_DIR / "results" / RUN_NAME))).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MAX_TASKS = int(os.getenv("MAX_TASKS", "0") or "0")
TARGET_QA_COUNT = int(os.getenv("TARGET_QA_COUNT", "0") or "0")
FRAMES_PER_CLIP = int(os.getenv("FRAMES_PER_CLIP", "8") or "8")
CLIP_IO_WORKERS = max(16, int(os.getenv("CLIP_IO_WORKERS", "16") or "16"))
NBACK_PROMPT_HINT_MODE = _parse_choice_env(
    "NBACK_PROMPT_HINT_MODE",
    default="structured",
    allowed={"structured", "minimal"},
)
NBACK_SELECTION_MODE = _parse_choice_env(
    "NBACK_SELECTION_MODE",
    default="balanced",
    allowed={"balanced", "random"},
)
NBACK_RANDOM_SEED = int(os.getenv("NBACK_RANDOM_SEED", "0") or "0")
NBACK_EVAL_MODE = _parse_choice_env(
    "NBACK_EVAL_MODE",
    default="fixed_last_clip",
    allowed={"fixed_last_clip", "prefix_last_clip_sweep"},
)
NBACK_CLIP_COUNT_SWEEP = _parse_int_list_env("NBACK_CLIP_COUNT_SWEEP")
NBACK_MAX_N = int(os.getenv("NBACK_MAX_N", "4") or "4")

INTERNVL_MODEL_PATH = str(Path(os.getenv("INTERNVL_MODEL_PATH", str(PROJECT_ROOT / "models/InternVL3_5-8B"))).resolve())
INTERNVL_INPUT_SIZE = int(os.getenv("INTERNVL_INPUT_SIZE", "280") or "280")
INTERNVL_MAX_NUM = int(os.getenv("INTERNVL_MAX_NUM", "1") or "1")
INTERNVL_TOTAL_MAX_NUM = int(os.getenv("INTERNVL_TOTAL_MAX_NUM", "96") or "96")
CHAT_MAX_RETRIES = int(os.getenv("CHAT_MAX_RETRIES", "3") or "3")
CHAT_BACKOFF = int(os.getenv("CHAT_BACKOFF_SECONDS", "2") or "2")
NBACK_SELECTION_SOURCE_MANIFEST = _parse_optional_text_env("NBACK_SELECTION_SOURCE_MANIFEST")
NBACK_TASK_ID_PREFIX = _parse_optional_text_env("NBACK_TASK_ID_PREFIX") or ""

# Resume configuration
PROGRESS_FILE = os.getenv("PROGRESS_FILE", str(RESULTS_DIR / "test_progress.json"))  # Progress checkpoint
OUTPUT_FILE = os.getenv("OUTPUT_FILE", str(RESULTS_DIR / "results.json"))  # Final results
SELECTION_FILE = os.getenv("SELECTION_FILE", str(RESULTS_DIR / "selection_manifest.json"))


def sample_video_frames_fixed(video_path: str) -> list[Image.Image]:
    """Sample a fixed number of frames as PIL images."""
    resolved_path = resolve_nback_video_path(video_path, data_root=DATA_ROOT)
    if not resolved_path.exists():
        print(f"❌ Video file not found: {video_path} -> {resolved_path}")
        return []
    if resolved_path.stat().st_size < MIN_VIDEO_FILE_SIZE_BYTES:
        print(f"❌ Video file too small to decode reliably: {video_path} -> {resolved_path}")
        return []

    try:
        frames = fetch_video(
            {
                "type": "video",
                "video": str(resolved_path),
                "nframes": FRAMES_PER_CLIP,
            }
        )
    except Exception as exc:
        print(f"❌ Failed to sample frames from {resolved_path}: {exc}")
        return []

    pil_frames = []
    for frame in frames:
        pil_frames.append(Image.fromarray(frame.permute(1, 2, 0).numpy().astype("uint8")))
    return pil_frames


def build_clip_payload(video_paths: list[str]) -> list[list[Image.Image]]:
    with ThreadPoolExecutor(max_workers=max(CLIP_IO_WORKERS, len(video_paths))) as executor:
        return list(executor.map(sample_video_frames_fixed, video_paths))


def call_vlm_n_back(clip_frames: list[list[Image.Image]], question: str) -> str:
    """Return the model response for a video sequence and question."""
    for attempt in range(1, CHAT_MAX_RETRIES + 1):
        try:
            return call_internvl_batched_yes_no(clip_frames, question)
        except Exception as e:
            print(f"⚠️ InternVL local retry {attempt} failed: {e}")
            if attempt < CHAT_MAX_RETRIES:
                time.sleep(CHAT_BACKOFF ** (attempt - 1))
    return "Error"


def load_progress():
    """Load completed test progress."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"finished_tasks": []}


def save_progress(finished_tasks):
    """Save completed test progress."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"finished_tasks": finished_tasks}, f, indent=4)


def load_existing_results():
    """Load existing test results."""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_selection_manifest(path: str | Path | None) -> dict | None:
    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def selection_manifest_matches_json(selection_manifest: dict | None, json_file_path: str | Path) -> bool:
    if not isinstance(selection_manifest, dict):
        return False
    manifest_json_path = selection_manifest.get("json_path")
    if not manifest_json_path:
        return True
    return Path(manifest_json_path).name == Path(json_file_path).name


def build_manifest_order_details(data: list[dict], ordered_tests: list[tuple[int, int]]) -> list[dict]:
    ordered_details: list[dict] = []
    for order_index, (domain_idx, test_idx) in enumerate(ordered_tests):
        domain_data = data[domain_idx]
        test = domain_data["tests"][test_idx]
        ordered_details.append(
            {
                "order_index": order_index,
                "domain_index": domain_idx,
                "domain": domain_data["domain"],
                "test_index": test_idx,
                "variation_id": test.get("variation_id"),
            }
        )
    return ordered_details


def build_question(label_key: str, label_sequence: list[str], idx_previous: int, idx_current: int, total_videos: int) -> tuple[str, str]:
    target_label = label_sequence[idx_current - 1]
    if label_key == "action_sequence":
        question = (
            f"There are {total_videos} videos. Please watch the sequence of videos provided. "
            f"Are the main actions/activities in the {idx_previous}th video and the {idx_current}th video "
            f"both '{target_label}'? Please answer with 'Yes' or 'No' only."
        )
    else:
        question = (
            f"There are {total_videos} videos. Please watch the sequence of videos provided. "
            f"Do the {idx_previous}th video and the {idx_current}th video depict the same scene/event, "
            f"namely '{target_label}'? Please answer with 'Yes' or 'No' only."
        )
    return target_label, question


def build_probe_items_fixed(label_key: str, label_sequence: list[str], n_back_eval: dict, total_videos: int) -> list[dict]:
    probe_items = []
    for n_back_key, expected_val in sorted(n_back_eval.items(), key=lambda item: normalize_n_back_key(item[0])):
        offset = normalize_n_back_key(n_back_key)
        idx_current = total_videos
        idx_previous = total_videos - offset
        target_label, single_question = build_question(label_key, label_sequence, idx_previous, idx_current, total_videos)
        probe_items.append(
            {
                "n_back_key": n_back_key,
                "n_value": offset,
                "expected_val": expected_val,
                "idx_current": idx_current,
                "idx_previous": idx_previous,
                "target_label": target_label,
                "single_question": single_question,
            }
        )
    return probe_items


def build_probe_items_prefix(label_key: str, label_sequence: list[str], total_videos: int) -> list[dict]:
    probe_items = []
    idx_current = total_videos
    max_n = min(NBACK_MAX_N, total_videos - 1)
    if max_n <= 0:
        return probe_items

    target_label = label_sequence[idx_current - 1]
    for offset in range(1, max_n + 1):
        idx_previous = total_videos - offset
        previous_label = label_sequence[idx_previous - 1]
        expected_val = 1 if previous_label == target_label else 0
        _, single_question = build_question(label_key, label_sequence, idx_previous, idx_current, total_videos)
        probe_items.append(
            {
                "n_back_key": f"N-{offset}_back",
                "n_value": offset,
                "expected_val": expected_val,
                "idx_current": idx_current,
                "idx_previous": idx_previous,
                "target_label": target_label,
                "single_question": single_question,
            }
        )
    return probe_items


def resolve_clip_counts(total_videos: int) -> list[int]:
    if NBACK_EVAL_MODE == "fixed_last_clip":
        return [total_videos]

    raw_counts = NBACK_CLIP_COUNT_SWEEP or [2, 3, 4, 5, 6]
    valid_counts = sorted({value for value in raw_counts if 2 <= value <= total_videos})
    if not valid_counts:
        return [total_videos]
    return valid_counts


def build_batched_question(label_key: str, probe_items: list[dict], total_videos: int) -> str:
    format_lines = [f"N-{probe['n_value']}: Yes or No" for probe in probe_items]
    question_lines = []
    for probe in probe_items:
        if label_key == "action_sequence":
            prompt_line = f"N-{probe['n_value']}: Are the main actions/activities in the " f"{probe['idx_previous']}th video and the {probe['idx_current']}th video both " f"'{probe['target_label']}'?"
        else:
            prompt_line = f"N-{probe['n_value']}: Do the {probe['idx_previous']}th video and the " f"{probe['idx_current']}th video depict the same scene/event, namely " f"'{probe['target_label']}'?"
        question_lines.append(prompt_line)

    return (
        f"There are {total_videos} videos. Please watch the sequence of videos provided carefully.\n"
        f"Answer the following N-back questions about the last clip.\n"
        f"Use exactly {len(probe_items)} lines in the format below and do not add any other text:\n" + "\n".join(format_lines) + "\nQuestions:\n" + "\n".join(question_lines)
    )


def save_selection_manifest(payload: dict):
    with open(SELECTION_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def collect_eligible_test_indices(data: list[dict]) -> tuple[dict[int, list[int]], dict[int, list[dict]]]:
    eligible_test_indices: dict[int, list[int]] = {}
    invalid_test_details: dict[int, list[dict]] = {}
    for domain_idx, domain_data in enumerate(data):
        domain_eligible = []
        domain_invalid = []
        for test_idx, test in enumerate(domain_data.get("tests", [])):
            invalid_videos = []
            for video_path in test.get("video_path_sequence", []):
                resolved_path = resolve_nback_video_path(video_path, data_root=DATA_ROOT)
                if not resolved_path.exists():
                    invalid_videos.append(
                        {
                            "video_path": video_path,
                            "resolved_path": str(resolved_path),
                            "reason": "missing",
                        }
                    )
                    continue
                if resolved_path.stat().st_size < MIN_VIDEO_FILE_SIZE_BYTES:
                    invalid_videos.append(
                        {
                            "video_path": video_path,
                            "resolved_path": str(resolved_path),
                            "reason": "file_too_small",
                            "file_size_bytes": resolved_path.stat().st_size,
                        }
                    )
            if invalid_videos:
                domain_invalid.append(
                    {
                        "test_index": test_idx,
                        "variation_id": test.get("variation_id"),
                        "invalid_videos": invalid_videos,
                    }
                )
            else:
                domain_eligible.append(test_idx)
        eligible_test_indices[domain_idx] = domain_eligible
        invalid_test_details[domain_idx] = domain_invalid
    return eligible_test_indices, invalid_test_details


def run_test(json_file_path):
    # Load progress and existing results.
    progress = load_progress()
    finished_tasks = set(progress["finished_tasks"])  # Use a set for completed-task lookups.
    all_results = load_existing_results()
    finished_tasks.update(item.get("task_id") for item in all_results if item.get("task_id"))
    completed_this_run = 0

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📦 JSON_PATH={json_file_path}")
    print(f"📦 DATA_ROOT={DATA_ROOT}")
    print(f"📦 MODEL_BACKEND={MODEL_BACKEND}")
    print(f"📦 MODEL_NAME={MODEL_NAME}")
    print(f"📦 OUTPUT_FILE={OUTPUT_FILE}")
    print(f"📦 PROGRESS_FILE={PROGRESS_FILE}")
    print(f"📦 NBACK_EVAL_MODE={NBACK_EVAL_MODE}")
    if NBACK_EVAL_MODE == "prefix_last_clip_sweep":
        print(f"📦 NBACK_CLIP_COUNT_SWEEP={resolve_clip_counts(12)}")
        print(f"📦 NBACK_MAX_N={NBACK_MAX_N}")
    print(f"📦 CLIP_IO_WORKERS={CLIP_IO_WORKERS}")
    print(f"📦 NBACK_PROMPT_HINT_MODE={NBACK_PROMPT_HINT_MODE}")
    print(f"📦 NBACK_SELECTION_MODE={NBACK_SELECTION_MODE}")
    if NBACK_SELECTION_MODE == "random":
        print(f"📦 NBACK_RANDOM_SEED={NBACK_RANDOM_SEED}")
    if NBACK_SELECTION_SOURCE_MANIFEST:
        print(f"📦 NBACK_SELECTION_SOURCE_MANIFEST={NBACK_SELECTION_SOURCE_MANIFEST}")
    if NBACK_TASK_ID_PREFIX:
        print(f"📦 NBACK_TASK_ID_PREFIX={NBACK_TASK_ID_PREFIX}")
    print(f"📦 MIN_VIDEO_FILE_SIZE_BYTES={MIN_VIDEO_FILE_SIZE_BYTES}")
    print(f"📦 CHAT_MAX_RETRIES={CHAT_MAX_RETRIES}")
    print(f"📦 CHAT_BACKOFF_SECONDS={CHAT_BACKOFF}")
    print(f"📦 FRAMES_PER_CLIP={FRAMES_PER_CLIP}")
    if NBACK_EVAL_MODE == "prefix_last_clip_sweep":
        print(f"📦 MAX_TOTAL_FRAMES_TARGET={FRAMES_PER_CLIP * max(resolve_clip_counts(12))}")
    else:
        print(f"📦 TOTAL_FRAMES_TARGET={FRAMES_PER_CLIP * 12}")
    if MODEL_BACKEND == "internvl_local":
        print(f"📦 INTERNVL_MODEL_PATH={INTERNVL_MODEL_PATH}")
        print(f"📦 INTERNVL_INPUT_SIZE={INTERNVL_INPUT_SIZE}")
        print(f"📦 INTERNVL_MAX_NUM={INTERNVL_MAX_NUM}")
        print(f"📦 INTERNVL_TOTAL_MAX_NUM={INTERNVL_TOTAL_MAX_NUM}")
    if MAX_TASKS > 0:
        print(f"📦 MAX_TASKS={MAX_TASKS}")
    if TARGET_QA_COUNT > 0:
        print(f"📦 TARGET_QA_COUNT={TARGET_QA_COUNT}")

    questions_per_test = len(data[0]["tests"][0]["n_back_evaluation"])
    eligible_test_indices, invalid_test_details = collect_eligible_test_indices(data)
    valid_test_count = sum(len(indices) for indices in eligible_test_indices.values())
    invalid_test_count = sum(len(items) for items in invalid_test_details.values())
    print(f"📦 VALID_TESTS={valid_test_count}")
    print(f"📦 INVALID_TESTS_FILTERED={invalid_test_count}")

    existing_selection_manifest = load_selection_manifest(SELECTION_FILE)
    if existing_selection_manifest and not selection_manifest_matches_json(existing_selection_manifest, json_file_path):
        print(f"⚠️ Ignore mismatched existing selection manifest: {SELECTION_FILE}")
        existing_selection_manifest = None
    source_selection_manifest = None
    if NBACK_SELECTION_SOURCE_MANIFEST:
        source_manifest_path = Path(NBACK_SELECTION_SOURCE_MANIFEST)
        if source_manifest_path.resolve() != Path(SELECTION_FILE).resolve():
            source_selection_manifest = load_selection_manifest(source_manifest_path)
            if source_selection_manifest and not selection_manifest_matches_json(source_selection_manifest, json_file_path):
                print(f"⚠️ Ignore mismatched source selection manifest: {source_manifest_path}")
                source_selection_manifest = None

    if NBACK_SELECTION_MODE == "random":
        precomputed_test_order = extract_test_order_from_selection_manifest(existing_selection_manifest)
        locked_prefix_tests = None
        reused_selection_source = None
        if precomputed_test_order:
            reused_selection_source = str(Path(SELECTION_FILE).resolve())
        else:
            locked_prefix_tests = extract_test_order_from_selection_manifest(source_selection_manifest)
            if locked_prefix_tests and NBACK_SELECTION_SOURCE_MANIFEST:
                reused_selection_source = str(Path(NBACK_SELECTION_SOURCE_MANIFEST).resolve())

        selection_plan = build_random_test_plan(
            data,
            target_qa_count=TARGET_QA_COUNT,
            questions_per_test=questions_per_test,
            eligible_test_indices=eligible_test_indices,
            random_seed=NBACK_RANDOM_SEED,
            locked_prefix_tests=locked_prefix_tests,
            precomputed_test_order=precomputed_test_order,
        )
    else:
        reused_selection_source = None
        selection_plan = build_balanced_test_plan(
            data,
            target_qa_count=TARGET_QA_COUNT,
            questions_per_test=questions_per_test,
            eligible_test_indices=eligible_test_indices,
        )
    selection_manifest = {
        "json_path": str(json_file_path),
        "model_name": MODEL_NAME,
        "evaluation_mode": NBACK_EVAL_MODE,
        "clip_count_sweep": resolve_clip_counts(12) if NBACK_EVAL_MODE == "prefix_last_clip_sweep" else [],
        "target_qa_count": TARGET_QA_COUNT,
        "selected_qa_count": selection_plan["selected_qa_count"],
        "selected_test_count": selection_plan["selected_test_count"],
        "valid_test_count": valid_test_count,
        "invalid_test_count": invalid_test_count,
        "questions_per_test": selection_plan["questions_per_test"],
        "selection_mode": NBACK_SELECTION_MODE,
        "random_seed": NBACK_RANDOM_SEED if NBACK_SELECTION_MODE == "random" else None,
        "selection_strategy": selection_plan["selection_strategy"],
        "selection_source_manifest": reused_selection_source,
        "ordered_valid_tests": build_manifest_order_details(data, selection_plan.get("ordered_valid_tests", [])),
        "selected_tests": [],
    }
    for domain_idx, domain_data in enumerate(data):
        selected_test_indices = selection_plan["selected_test_indices"].get(domain_idx, [])
        selection_manifest["selected_tests"].append(
            {
                "domain_index": domain_idx,
                "domain": domain_data["domain"],
                "eligible_test_indices": eligible_test_indices.get(domain_idx, []),
                "invalid_tests": invalid_test_details.get(domain_idx, []),
                "selected_test_indices": selected_test_indices,
                "selected_variation_ids": [domain_data["tests"][test_idx]["variation_id"] for test_idx in selected_test_indices],
            }
        )
    save_selection_manifest(selection_manifest)

    for domain_idx, domain_data in enumerate(data):
        domain = domain_data["domain"]
        print(f"\n🚀 Testing Domain: {domain}")
        selected_test_indices = set(selection_plan["selected_test_indices"].get(domain_idx, []))

        for test_id, test in enumerate(domain_data["tests"]):
            if TARGET_QA_COUNT > 0 and test_id not in selected_test_indices:
                continue

            v_id = test["variation_id"]
            video_seq = test["video_path_sequence"]
            label_key = "action_sequence" if "action_sequence" in test else "scene_sequence"
            label_seq = test[label_key]
            n_back_eval = test["n_back_evaluation"]

            total_videos = len(video_seq)
            full_clip_frames = build_clip_payload(video_seq)

            for clip_count in resolve_clip_counts(total_videos):
                clip_frames = full_clip_frames[:clip_count]
                prefix_label_seq = label_seq[:clip_count]
                if NBACK_EVAL_MODE == "prefix_last_clip_sweep":
                    probe_items = build_probe_items_prefix(label_key, prefix_label_seq, clip_count)
                else:
                    probe_items = build_probe_items_fixed(label_key, prefix_label_seq, n_back_eval, clip_count)

                pending_probe_items = []
                for probe in probe_items:
                    task_id_core = f"{domain_idx}_{test_id}_{domain}_{v_id}_K{clip_count}_{probe['n_back_key']}"
                    task_id = f"{NBACK_TASK_ID_PREFIX}_{task_id_core}" if NBACK_TASK_ID_PREFIX else task_id_core
                    if task_id in finished_tasks:
                        print(f"⏭️  Skipping {task_id} (already completed)")
                        continue
                    pending_probe_items.append({**probe, "task_id": task_id, "clip_count": clip_count})

                if not pending_probe_items:
                    continue

                if MAX_TASKS > 0:
                    remaining_budget = MAX_TASKS - completed_this_run
                    if remaining_budget <= 0:
                        print(f"\n🛑 Reached MAX_TASKS={MAX_TASKS}, stop early.")
                        print(f"📄 Partial results saved to {OUTPUT_FILE}")
                        return
                    pending_probe_items = pending_probe_items[:remaining_budget]

                for probe in pending_probe_items:
                    print(f"  - Var {v_id} | K={clip_count} | {probe['n_back_key']} | Target: {probe['target_label']}")

                batch_question = build_batched_question(label_key, pending_probe_items, clip_count)
                batch_response = call_vlm_n_back(clip_frames, batch_question)
                parsed_batch = parse_batched_n_back_response(
                    batch_response,
                    expected_n_values=[probe["n_value"] for probe in pending_probe_items],
                )

                for probe in pending_probe_items:
                    n_back_key = probe["n_back_key"]
                    expected_val = probe["expected_val"]
                    parse_payload = parsed_batch.get(probe["n_value"]) or parse_yes_no_response(batch_response)
                    parsed_answer = parse_payload["parsed_answer"]
                    predicted_value = 1 if parsed_answer == "yes" else 0 if parsed_answer == "no" else None
                    is_correct = predicted_value is not None and predicted_value == expected_val

                    print(f"    [Result] K={clip_count} | {n_back_key} | Parsed: {parsed_answer or 'UNRESOLVED'} " f"| Expected: {'Yes(1)' if expected_val==1 else 'No(0)'} | Match: {'✅' if is_correct else '❌'}")

                    result_item = {
                        "model_name": MODEL_NAME,
                        "domain": domain,
                        "domain_index": domain_idx,
                        "test_index": test_id,
                        "variation_id": v_id,
                        "clip_count": clip_count,
                        "evaluation_mode": NBACK_EVAL_MODE,
                        "current_clip_index": probe["idx_current"],
                        "previous_clip_index": probe["idx_previous"],
                        "n_back_type": n_back_key,
                        "n_value": probe["n_value"],
                        "label_key": label_key,
                        "target_label": probe["target_label"],
                        "question": probe["single_question"],
                        "batch_question": batch_question,
                        "batch_model_response": batch_response,
                        "model_response": parse_payload.get("raw_text") or batch_response,
                        "parsed_answer": parsed_answer,
                        "predicted_value": predicted_value,
                        "resolved_prediction": parse_payload["resolved"],
                        "format_compliant": parse_payload["format_compliant"],
                        "parse_status": parse_payload["parse_status"],
                        "ground_truth": expected_val,
                        "is_correct": is_correct,
                        "task_id": probe["task_id"],
                        "request_mode": "batched_yes_no",
                    }
                    for extra_key in (
                        "pair_controlled",
                        "pair_controlled_mode",
                        "source_domain_index",
                        "source_test_index",
                        "source_variation_id",
                        "source_clip_count",
                        "target_pair_original_indices",
                        "target_pair_reordered_indices",
                        "distractor_original_indices",
                        "pair_ground_truth",
                        "pair_label_a",
                        "pair_label_b",
                    ):
                        if extra_key in test:
                            result_item[extra_key] = test[extra_key]
                    if label_key == "action_sequence":
                        result_item["target_action"] = probe["target_label"]
                    else:
                        result_item["target_scene"] = probe["target_label"]
                    all_results.append(result_item)
                    completed_this_run += 1
                    finished_tasks.add(probe["task_id"])
                    save_progress(list(finished_tasks))

                    with open(OUTPUT_FILE, "w", encoding="utf-8") as rf:
                        json.dump(all_results, rf, indent=4, ensure_ascii=False)

    print(f"\n✨ All tests completed! Results saved to {OUTPUT_FILE}")
    if all_results:
        correct = sum(1 for item in all_results if item.get("is_correct"))
        print(f"📊 Accuracy so far: {correct}/{len(all_results)} = {correct / len(all_results):.4f}")
    # Remove the checkpoint after all tests complete.
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("🗑️ Progress file cleared")


if __name__ == "__main__":
    if os.path.exists(JSON_PATH):
        run_test(JSON_PATH)
    else:
        print(f"❌ Cannot find input file: {JSON_PATH}")
