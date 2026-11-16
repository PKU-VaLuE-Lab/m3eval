#!/usr/bin/env bash
# InternVL3.5-8B N-back runner.
# This is the complete paper-style N-back workflow for the public release.
set -euo pipefail

# Configuration:
# - K sweep: 2,3,4,5,6,8,10,12.
# - N max: 4.
# - Eight frames per N-back clip.
# - Original paper prompt, including the named target attribute.
# - One InternVL model with one process per shard.

cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

source "${PROJECT_ROOT}/scripts/open_source/python_runtime.sh"

quote_cmd() {
  local quoted=""
  printf -v quoted '%q ' "$@"
  quoted="${quoted% }"
  printf '%s' "${quoted}"
}

sum_questions_per_test() {
  python3 - <<'PY' "$1" "$2"
import sys

counts = [int(part) for part in sys.argv[1].split(",") if part.strip()]
max_n = int(sys.argv[2])
print(sum(min(max(count - 1, 0), max_n) for count in counts))
PY
}

OUTPUT_ROOT=""
GPU_LIST="${FINAL_PAPER_GPUS:-0,1,2,4,5}"
PORT_LIST="${FINAL_PAPER_PORTS:-8210,8211,8212,8213,8214}"
TESTS_PER_SPLIT="32"
PROMPT_HINT_MODE="minimal"
CLIP_COUNT_SWEEP="2,3,4,5,6,8,10,12"
NBACK_MAX_N="4"
SELECTION_MODE="balanced"
FRAMES_PER_CLIP="8"
CLIP_IO_WORKERS="16"
MIN_VIDEO_FILE_SIZE_BYTES="10000"
INTERNVL_INPUT_SIZE="280"
INTERNVL_MAX_NUM="1"
INTERNVL_TOTAL_MAX_NUM="96"
INTERNVL_DEVICE_MAP="${FINAL_PAPER_INTERNVL_DEVICE_MAP:-cuda:0}"
INTERNVL_VISION_BATCH_SIZE="${FINAL_PAPER_INTERNVL_VISION_BATCH_SIZE:-0}"
INTERNVL_LLM_ATTENTION="${FINAL_PAPER_INTERNVL_LLM_ATTENTION:-}"
INTERNVL_PAIR_GPUS="${FINAL_PAPER_INTERNVL_PAIR_GPUS:-0}"
ACTION_JSON="${PROJECT_ROOT}/n-back/nback_test_action.json"
SCENE_JSON="${PROJECT_ROOT}/n-back/nback_test_scene.json"
DEFAULT_NBACK_DATA_ROOT="${PROJECT_ROOT}/data/m3eval/videos/nback_paper_short_segments"
if [[ -n "${M3EVAL_DATA_ROOT:-}" ]]; then
  if [[ -d "${M3EVAL_DATA_ROOT}/videos/nback_paper_short_segments" ]]; then
    DEFAULT_NBACK_DATA_ROOT="${M3EVAL_DATA_ROOT}/videos/nback_paper_short_segments"
  elif [[ -d "${M3EVAL_DATA_ROOT}/releases/legacy_nback_paper_v1/runtime/precut_full_20pairs_per_dataset_seg10" ]]; then
    DEFAULT_NBACK_DATA_ROOT="${M3EVAL_DATA_ROOT}/releases/legacy_nback_paper_v1/runtime/precut_full_20pairs_per_dataset_seg10"
  fi
fi
NBACK_DATA_ROOT="${FINAL_PAPER_NBACK_DATA_ROOT:-${DEFAULT_NBACK_DATA_ROOT}}"
MODEL_FILTER_CSV="${FINAL_PAPER_MODEL_ALIASES:-}"
STATUS_CSV=""
QUEUE_MANIFEST=""

MODEL_NAMES=("InternVL3.5-8B")
MODEL_ALIASES=("internvl3_5_8b")
MODEL_PATHS=("${FINAL_PAPER_INTERNVL_MODEL_PATH:-${PROJECT_ROOT}/models/InternVL3_5-8B}")
MODEL_BACKENDS=("internvl_local")

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/open_source/run_nback_internvl35_8b.sh \
    [--output_root PATH] \
    [--gpus 0,1,2,4,5] [--ports 8210,8211,8212,8213,8214] \
    [--tests_per_split 32] \
    [--prompt_hint_mode minimal] \
    [--clip_count_sweep 2,3,4,5,6,8,10,12]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPU_LIST="$2"; shift 2 ;;
    --ports) PORT_LIST="$2"; shift 2 ;;
    --tests_per_split) TESTS_PER_SPLIT="$2"; shift 2 ;;
    --model_aliases) MODEL_FILTER_CSV="$2"; shift 2 ;;
    --prompt_hint_mode) PROMPT_HINT_MODE="$2"; shift 2 ;;
    --clip_count_sweep) CLIP_COUNT_SWEEP="$2"; shift 2 ;;
    --nback_max_n) NBACK_MAX_N="$2"; shift 2 ;;
    --selection_mode) SELECTION_MODE="$2"; shift 2 ;;
    --frames_per_clip) FRAMES_PER_CLIP="$2"; shift 2 ;;
    --clip_io_workers) CLIP_IO_WORKERS="$2"; shift 2 ;;
    --min_video_file_size_bytes) MIN_VIDEO_FILE_SIZE_BYTES="$2"; shift 2 ;;
    --internvl_input_size) INTERNVL_INPUT_SIZE="$2"; shift 2 ;;
    --internvl_max_num) INTERNVL_MAX_NUM="$2"; shift 2 ;;
    --internvl_total_max_num) INTERNVL_TOTAL_MAX_NUM="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -n "${MODEL_FILTER_CSV}" ]]; then
  IFS=',' read -r -a REQUESTED_MODEL_ALIASES <<< "${MODEL_FILTER_CSV}"
  FILTERED_MODEL_NAMES=()
  FILTERED_MODEL_ALIASES=()
  FILTERED_MODEL_PATHS=()
  FILTERED_MODEL_BACKENDS=()
  for requested_alias in "${REQUESTED_MODEL_ALIASES[@]}"; do
    requested_alias="${requested_alias//[[:space:]]/}"
    [[ -z "${requested_alias}" ]] && continue
    found="0"
    for idx in "${!MODEL_ALIASES[@]}"; do
      if [[ "${MODEL_ALIASES[idx]}" == "${requested_alias}" ]]; then
        FILTERED_MODEL_NAMES+=("${MODEL_NAMES[idx]}")
        FILTERED_MODEL_ALIASES+=("${MODEL_ALIASES[idx]}")
        FILTERED_MODEL_PATHS+=("${MODEL_PATHS[idx]}")
        FILTERED_MODEL_BACKENDS+=("${MODEL_BACKENDS[idx]}")
        found="1"
        break
      fi
    done
    if [[ "${found}" != "1" ]]; then
      echo "[ERROR] unknown model alias in --model_aliases: ${requested_alias}"
      exit 1
    fi
  done
  if (( ${#FILTERED_MODEL_ALIASES[@]} == 0 )); then
    echo "[ERROR] --model_aliases selected no models"
    exit 1
  fi
  MODEL_NAMES=("${FILTERED_MODEL_NAMES[@]}")
  MODEL_ALIASES=("${FILTERED_MODEL_ALIASES[@]}")
  MODEL_PATHS=("${FILTERED_MODEL_PATHS[@]}")
  MODEL_BACKENDS=("${FILTERED_MODEL_BACKENDS[@]}")
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
IFS=',' read -r -a PORTS <<< "${PORT_LIST}"
if (( ${#GPUS[@]} == 0 )); then
  echo "[ERROR] empty --gpus"
  exit 1
fi
if (( ${#GPUS[@]} != ${#PORTS[@]} )); then
  echo "[ERROR] --gpus and --ports must have the same length"
  exit 1
fi
if [[ "${SELECTION_MODE}" != "balanced" ]]; then
  echo "[ERROR] prefix formal queue only supports --selection_mode balanced"
  exit 1
fi

SHARD_COUNT="${#GPUS[@]}"
QUESTIONS_PER_TEST="$(sum_questions_per_test "${CLIP_COUNT_SWEEP}" "${NBACK_MAX_N}")"
EXPECTED_ROWS_PER_SPLIT="$(( TESTS_PER_SPLIT * QUESTIONS_PER_TEST ))"

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${PROJECT_ROOT}/${FINAL_PAPER_OUTPUT_ROOT:-lmms-eval/output/final_paper_rerun}/nback_open_models_full_$(date +%Y-%m-%d_%H-%M-%S)"
fi
mkdir -p "${OUTPUT_ROOT}/logs"
QUEUE_LOG="${OUTPUT_ROOT}/logs/queue.log"
QUEUE_MANIFEST="${OUTPUT_ROOT}/queue_manifest.json"
STATUS_CSV="${OUTPUT_ROOT}/queue_status.csv"
SHARED_SELECTION_DIR="${OUTPUT_ROOT}/shared_selection"
FINAL_DIR="${OUTPUT_ROOT}/final"

exec > >(tee -a "${QUEUE_LOG}") 2>&1

log() {
  echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*"
}

write_manifest() {
  python3 - <<'PY' "${QUEUE_MANIFEST}" "${OUTPUT_ROOT}" "${TESTS_PER_SPLIT}" "${CLIP_COUNT_SWEEP}" "${NBACK_MAX_N}" "${EXPECTED_ROWS_PER_SPLIT}" "${GPU_LIST}" "${PORT_LIST}" "${PROMPT_HINT_MODE}" "${SELECTION_MODE}" "${FRAMES_PER_CLIP}" "${SHARD_COUNT}" "${INTERNVL_DEVICE_MAP}" "${INTERNVL_VISION_BATCH_SIZE}" "${INTERNVL_LLM_ATTENTION}" "${INTERNVL_PAIR_GPUS}" "$@"
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
payload = {
    "output_root": sys.argv[2],
    "tests_per_split": int(sys.argv[3]),
    "clip_count_sweep": [int(part) for part in sys.argv[4].split(",") if part.strip()],
    "nback_max_n": int(sys.argv[5]),
    "expected_rows_per_split": int(sys.argv[6]),
    "gpus": [int(part) for part in sys.argv[7].split(",") if part.strip()],
    "ports": [int(part) for part in sys.argv[8].split(",") if part.strip()],
    "prompt_hint_mode": sys.argv[9],
    "selection_mode": sys.argv[10],
    "frames_per_clip": int(sys.argv[11]),
    "shard_count": int(sys.argv[12]),
    "internvl_device_map": sys.argv[13],
    "internvl_vision_batch_size": int(sys.argv[14]),
    "internvl_llm_attention": sys.argv[15] or None,
    "internvl_pair_gpus": sys.argv[16] == "1",
    "models": [json.loads(item) for item in sys.argv[17:]],
}
manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
}

append_status_row() {
  local model_alias="$1"
  local model_name="$2"
  local backend="$3"
  local started_at="$4"
  local finished_at="$5"
  local shard_status="$6"
  local action_status="$7"
  local scene_status="$8"
  local overall_status="$9"

  if [[ ! -f "${STATUS_CSV}" ]]; then
    echo "model_alias,model_name,backend,started_at,finished_at,shard_status,action_status,scene_status,overall_status" > "${STATUS_CSV}"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${model_alias}" "${model_name}" "${backend}" "${started_at}" "${finished_at}" \
    "${shard_status}" "${action_status}" "${scene_status}" "${overall_status}" >> "${STATUS_CSV}"
}

shared_json_path() {
  local split_name="$1"
  local shard_index="$2"
  echo "${SHARED_SELECTION_DIR}/${split_name}/shards/${split_name}_shard${shard_index}.json"
}

shared_manifest_path() {
  local split_name="$1"
  local shard_index="$2"
  echo "${SHARED_SELECTION_DIR}/${split_name}/shards/${split_name}_shard${shard_index}_manifest.json"
}

build_shared_selection_split() {
  local split_name="$1"
  local source_json="$2"
  local split_dir="${SHARED_SELECTION_DIR}/${split_name}"
  mkdir -p "${split_dir}/shards"

  local full_json="${split_dir}/${split_name}_full.json"
  local full_manifest="${split_dir}/${split_name}_full_manifest.json"
  log "[BUILD] shared selection split=${split_name}"
  python3 n-back/build_prefix_nback_subset_json.py \
    --source_json "${source_json}" \
    --output_json "${full_json}" \
    --selection_manifest_out "${full_manifest}" \
    --tests_per_split "${TESTS_PER_SPLIT}" \
    --selection_mode "${SELECTION_MODE}" \
    --data_root "${NBACK_DATA_ROOT}" \
    --min_video_file_size_bytes "${MIN_VIDEO_FILE_SIZE_BYTES}"

  local shard_index
  for (( shard_index=0; shard_index<SHARD_COUNT; shard_index++ )); do
    python3 n-back/build_prefix_nback_subset_json.py \
      --source_json "${source_json}" \
      --output_json "$(shared_json_path "${split_name}" "${shard_index}")" \
      --selection_manifest_out "$(shared_manifest_path "${split_name}" "${shard_index}")" \
      --tests_per_split "${TESTS_PER_SPLIT}" \
      --selection_mode "${SELECTION_MODE}" \
      --data_root "${NBACK_DATA_ROOT}" \
      --min_video_file_size_bytes "${MIN_VIDEO_FILE_SIZE_BYTES}" \
      --shard_count "${SHARD_COUNT}" \
      --shard_index "${shard_index}"
  done

  python3 - <<'PY' "${full_manifest}" "${SHARD_COUNT}" "${split_name}" "${SHARED_SELECTION_DIR}"
import json
import sys
from pathlib import Path

full_manifest = Path(sys.argv[1]).resolve()
shard_count = int(sys.argv[2])
split_name = sys.argv[3]
shared_root = Path(sys.argv[4]).resolve()

payload = json.loads(full_manifest.read_text(encoding="utf-8"))
full_pairs = {
    (int(item["domain_index"]), int(item["test_index"]))
    for item in payload.get("selected_pairs", [])
}
union_pairs = set()
for shard_idx in range(shard_count):
    shard_manifest = shared_root / split_name / "shards" / f"{split_name}_shard{shard_idx}_manifest.json"
    shard_payload = json.loads(shard_manifest.read_text(encoding="utf-8"))
    shard_pairs = {
        (int(item["domain_index"]), int(item["test_index"]))
        for item in shard_payload.get("selected_pairs", [])
    }
    overlap = union_pairs & shard_pairs
    if overlap:
        raise SystemExit(f"Shard overlap detected in {split_name}: {sorted(overlap)[:3]}")
    union_pairs |= shard_pairs

if union_pairs != full_pairs:
    missing = sorted(full_pairs - union_pairs)[:3]
    extra = sorted(union_pairs - full_pairs)[:3]
    raise SystemExit(f"Shard coverage mismatch in {split_name}: missing={missing} extra={extra}")

print(f"[DONE] validated shared shards split={split_name} count={len(full_pairs)}")
PY
}

build_shared_selection() {
  build_shared_selection_split "action" "${ACTION_JSON}"
  build_shared_selection_split "scene" "${SCENE_JSON}"
}

model_root() {
  echo "${OUTPUT_ROOT}/model_runs/$1"
}

shard_root() {
  local model_alias="$1"
  local split_name="$2"
  local shard_index="$3"
  echo "$(model_root "${model_alias}")/shards/${split_name}_shard${shard_index}"
}

merged_results_path() {
  local model_alias="$1"
  local split_name="$2"
  echo "$(model_root "${model_alias}")/final/${split_name}/results.json"
}

merged_summary_path() {
  local model_alias="$1"
  local split_name="$2"
  echo "$(model_root "${model_alias}")/final/${split_name}/analysis/summary.json"
}

split_final_complete() {
  local model_alias="$1"
  local split_name="$2"
  python3 - <<'PY' "$(merged_results_path "$1" "$2")" "$(merged_summary_path "$1" "$2")" "${EXPECTED_ROWS_PER_SPLIT}"
import json
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
expected = int(sys.argv[3])
if not results_path.exists() or not summary_path.exists():
    raise SystemExit(1)
try:
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if len(rows) != expected:
    raise SystemExit(1)
if int(summary.get("total_questions", 0) or 0) != expected:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

model_final_complete() {
  local model_alias="$1"
  split_final_complete "${model_alias}" "action" && split_final_complete "${model_alias}" "scene"
}

run_shard_job() {
  local model_alias="$1"
  local model_name="$2"
  local model_path="$3"
  local backend="$4"
  local gpu="$5"
  local port="$6"
  local split_name="$7"
  local shard_index="$8"
  local job_root
  job_root="$(shard_root "${model_alias}" "${split_name}" "${shard_index}")"
  local selected_count
  selected_count="$("${PYTHON_BIN}" - "$(shared_json_path "${split_name}" "${shard_index}")" <<'PY'
import json
import sys
from pathlib import Path
print(sum(len(domain['tests']) for domain in json.loads(Path(sys.argv[1]).read_text())))
PY
)"
  if [[ "${selected_count}" == "0" ]]; then
    log "[SKIP] empty shard split=${split_name} shard=${shard_index}"
    return 0
  fi

  local -a cmd=(
    bash scripts/open_source/run_prefix_nback_local_split.sh
    --output_root "${job_root}"
    --gpu "${gpu}"
    --port "${port}"
    --model_name "${model_name}"
    --model_path "${model_path}"
    --model_backend "${backend}"
    --split_name "${split_name}"
    --json_path "$(shared_json_path "${split_name}" "${shard_index}")"
    --data_root "${NBACK_DATA_ROOT}"
    --task_id_prefix "${model_alias}_${split_name}_shard${shard_index}"
    --prompt_hint_mode "${PROMPT_HINT_MODE}"
    --clip_count_sweep "${CLIP_COUNT_SWEEP}"
    --nback_max_n "${NBACK_MAX_N}"
    --frames_per_clip "${FRAMES_PER_CLIP}"
    --clip_io_workers "${CLIP_IO_WORKERS}"
    --min_video_file_size_bytes "${MIN_VIDEO_FILE_SIZE_BYTES}"
  )

  cmd+=(
    --internvl_input_size "${INTERNVL_INPUT_SIZE}"
    --internvl_max_num "${INTERNVL_MAX_NUM}"
    --internvl_total_max_num "${INTERNVL_TOTAL_MAX_NUM}"
    --internvl_device_map "${INTERNVL_DEVICE_MAP}"
    --internvl_vision_batch_size "${INTERNVL_VISION_BATCH_SIZE}"
    --internvl_llm_attention "${INTERNVL_LLM_ATTENTION}"
  )

  log "[RUN] model=${model_alias} backend=${backend} gpu=${gpu} split=${split_name} shard=${shard_index}"
  "${cmd[@]}"
}

run_gpu_worker() {
  local model_alias="$1"
  local model_name="$2"
  local model_path="$3"
  local backend="$4"
  local worker_index="$5"
  local gpu="$6"
  local port="$7"
  local worker_status=0

  run_shard_job "${model_alias}" "${model_name}" "${model_path}" "${backend}" "${gpu}" "${port}" "action" "${worker_index}" || worker_status=1
  run_shard_job "${model_alias}" "${model_name}" "${model_path}" "${backend}" "${gpu}" "${port}" "scene" "${worker_index}" || worker_status=1

  return "${worker_status}"
}

merge_model_split() {
  local model_alias="$1"
  local split_name="$2"
  local final_root
  final_root="$(model_root "${model_alias}")/final/${split_name}"
  mkdir -p "${final_root}"

  local inputs=()
  local shard_index
  for (( shard_index=0; shard_index<SHARD_COUNT; shard_index++ )); do
    local candidate
    candidate="$(shard_root "${model_alias}" "${split_name}" "${shard_index}")/results/results.json"
    if [[ -f "${candidate}" ]]; then
      inputs+=("${candidate}")
    fi
  done

  if (( ${#inputs[@]} == 0 )); then
    log "[ERROR] no shard results to merge model=${model_alias} split=${split_name}"
    return 1
  fi

  python3 n-back/merge_n_back_results.py \
    --inputs "${inputs[@]}" \
    --output "$(merged_results_path "${model_alias}" "${split_name}")"

  python3 n-back/analyze_n_back_results.py \
    --results "$(merged_results_path "${model_alias}" "${split_name}")"

  if ! split_final_complete "${model_alias}" "${split_name}"; then
    log "[ERROR] merged split incomplete model=${model_alias} split=${split_name} expected_rows=${EXPECTED_ROWS_PER_SPLIT}"
    return 1
  fi
  log "[DONE] merged split model=${model_alias} split=${split_name}"
  return 0
}


build_final_outputs() {
  mkdir -p "${FINAL_DIR}"
  local summary_paths=()
  local model_alias
  for model_alias in "${MODEL_ALIASES[@]}"; do
    local action_summary scene_summary
    action_summary="$(merged_summary_path "${model_alias}" "action")"
    scene_summary="$(merged_summary_path "${model_alias}" "scene")"
    [[ -f "${action_summary}" ]] && summary_paths+=("${action_summary}")
    [[ -f "${scene_summary}" ]] && summary_paths+=("${scene_summary}")
  done


  if (( ${#summary_paths[@]} > 0 )) && [[ -f "${PROJECT_ROOT}/n-back/build_prefix_nback_formal_report.py" ]]; then
    python3 n-back/build_prefix_nback_formal_report.py \
      --summary "${summary_paths[@]}" \
      --output_dir "${FINAL_DIR}"
  fi

  if (( ${#summary_paths[@]} > 0 )) && [[ -f "${PROJECT_ROOT}/n-back/build_prefix_nback_clip_count_figure.py" ]]; then
    python3 n-back/build_prefix_nback_clip_count_figure.py \
      --summary "${summary_paths[@]}" \
      --output "${FINAL_DIR}/prefix_nback_clip_count_curves.png"
    log "[DONE] final figure=${FINAL_DIR}/prefix_nback_clip_count_curves.png"
  fi
}

run_model() {
  local model_alias="$1"
  local model_name="$2"
  local model_path="$3"
  local backend="$4"
  local model_dir
  model_dir="$(model_root "${model_alias}")"
  mkdir -p "${model_dir}/logs"

  if model_final_complete "${model_alias}"; then
    log "[SKIP] model=${model_alias} already complete"
    return 0
  fi

  local started_at finished_at
  started_at="$(date +%Y-%m-%dT%H:%M:%S)"
  log "[MODEL START] alias=${model_alias} name=${model_name} backend=${backend}"

  local worker_pids=()
  local worker_statuses=()
  local worker_index
  local worker_count="${SHARD_COUNT}"
  if [[ "${backend}" == "internvl_local" && "${INTERNVL_PAIR_GPUS}" == "1" ]]; then
    if (( SHARD_COUNT % 2 != 0 )); then
      log "[ERROR] paired InternVL workers require an even GPU count; got ${SHARD_COUNT}"
      return 1
    fi
    worker_count=$(( SHARD_COUNT / 2 ))
    for (( worker_index=0; worker_index<worker_count; worker_index++ )); do
      local gpu_pair="${GPUS[worker_index * 2]},${GPUS[worker_index * 2 + 1]}"
      local port="${PORTS[worker_index * 2]}"
      local gpu_slug="${gpu_pair//,/_}"
      local worker_log="${model_dir}/logs/worker_gpus${gpu_slug}.log"
      (
        pair_status=0
        for (( shard_index=worker_index; shard_index<SHARD_COUNT; shard_index+=worker_count )); do
          run_gpu_worker "${model_alias}" "${model_name}" "${model_path}" "${backend}" "${shard_index}" "${gpu_pair}" "${port}" || pair_status=1
        done
        exit "${pair_status}"
      ) > >(tee -a "${worker_log}") 2>&1 &
      worker_pids+=("$!")
      worker_statuses+=("0")
    done
  else
    for (( worker_index=0; worker_index<SHARD_COUNT; worker_index++ )); do
      local gpu="${GPUS[worker_index]}"
      local port="${PORTS[worker_index]}"
      local worker_log="${model_dir}/logs/worker_gpu${gpu}.log"
      (
        run_gpu_worker "${model_alias}" "${model_name}" "${model_path}" "${backend}" "${worker_index}" "${gpu}" "${port}"
      ) > >(tee -a "${worker_log}") 2>&1 &
      worker_pids+=("$!")
      worker_statuses+=("0")
    done
  fi

  local shard_status=0
  for (( worker_index=0; worker_index<worker_count; worker_index++ )); do
    if ! wait "${worker_pids[worker_index]}"; then
      worker_statuses[worker_index]="1"
      shard_status=1
    fi
  done

  local action_status=0
  local scene_status=0
  merge_model_split "${model_alias}" "action" || action_status=1
  merge_model_split "${model_alias}" "scene" || scene_status=1

  finished_at="$(date +%Y-%m-%dT%H:%M:%S)"
  local overall_status=0
  if [[ "${shard_status}" != "0" || "${action_status}" != "0" || "${scene_status}" != "0" ]]; then
    overall_status=1
  fi
  append_status_row "${model_alias}" "${model_name}" "${backend}" "${started_at}" "${finished_at}" "${shard_status}" "${action_status}" "${scene_status}" "${overall_status}"
  log "[MODEL END] alias=${model_alias} shard_status=${shard_status} action_status=${action_status} scene_status=${scene_status}"
  return "${overall_status}"
}

log "[START] output_root=${OUTPUT_ROOT}"
log "[CONFIG] models=${MODEL_ALIASES[*]} gpus=${GPU_LIST} ports=${PORT_LIST} tests_per_split=${TESTS_PER_SPLIT} clip_count_sweep=${CLIP_COUNT_SWEEP} expected_rows_per_split=${EXPECTED_ROWS_PER_SPLIT}"

manifest_models=()
for idx in "${!MODEL_NAMES[@]}"; do
  manifest_models+=("{\"alias\":\"${MODEL_ALIASES[idx]}\",\"name\":\"${MODEL_NAMES[idx]}\",\"path\":\"${MODEL_PATHS[idx]}\",\"backend\":\"${MODEL_BACKENDS[idx]}\"}")
done
write_manifest "${manifest_models[@]}"

build_shared_selection
if [[ "${M3EVAL_PAPER_PROFILE:-}" == "internvl35-paper-2026" ]]; then
  "${PYTHON_BIN}" scripts/open_source/validate_paper_nback.py --selection-root "${SHARED_SELECTION_DIR}" --tests-per-split "${TESTS_PER_SPLIT}"
fi

QUEUE_OVERALL_STATUS=0
for idx in "${!MODEL_NAMES[@]}"; do
  if ! run_model "${MODEL_ALIASES[idx]}" "${MODEL_NAMES[idx]}" "${MODEL_PATHS[idx]}" "${MODEL_BACKENDS[idx]}"; then
    QUEUE_OVERALL_STATUS=1
  fi
done

build_final_outputs

log "[EXIT] done output_root=${OUTPUT_ROOT} overall_status=${QUEUE_OVERALL_STATUS}"
exit "${QUEUE_OVERALL_STATUS}"
