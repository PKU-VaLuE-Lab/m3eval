#!/usr/bin/env bash
set -euo pipefail

# Run one action or scene shard for the local InternVL backend.

cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

source "${PROJECT_ROOT}/scripts/open_source/python_runtime.sh"

quote_cmd() {
  local quoted=""
  printf -v quoted '%q ' "$@"
  quoted="${quoted% }"
  printf '%s' "${quoted}"
}

sanitize_name() {
  python3 - <<'PY' "$1"
import re
import sys
text = re.sub(r"[^0-9A-Za-z._-]+", "_", sys.argv[1].strip())
text = text.strip("._-") or "run"
print(text)
PY
}

OUTPUT_ROOT=""
GPU="0"
PORT="8020"
MODEL_NAME="InternVL3.5-8B"
MODEL_PATH="${PROJECT_ROOT}/models/InternVL3_5-8B"
MODEL_BACKEND_NAME="internvl_local"
SPLIT_NAME="action"
JSON_PATH=""
PROMPT_HINT_MODE="minimal"
CLIP_COUNT_SWEEP="2,3,4,5,6,8,10,12"
NBACK_MAX_N="4"
NBACK_EVAL_MODE="prefix_last_clip_sweep"
FRAMES_PER_CLIP="8"
CLIP_IO_WORKERS="16"
MIN_VIDEO_FILE_SIZE_BYTES="10000"
TASK_ID_PREFIX=""
INTERNVL_INPUT_SIZE="280"
INTERNVL_MAX_NUM="1"
INTERNVL_TOTAL_MAX_NUM="96"
INTERNVL_DEVICE_MAP="${FINAL_PAPER_INTERNVL_DEVICE_MAP:-cuda:0}"
INTERNVL_VISION_BATCH_SIZE="${FINAL_PAPER_INTERNVL_VISION_BATCH_SIZE:-0}"
INTERNVL_LLM_ATTENTION="${FINAL_PAPER_INTERNVL_LLM_ATTENTION:-}"
NBACK_DATA_ROOT_VALUE="${NBACK_DATA_ROOT:-}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/open_source/run_prefix_nback_local_split.sh \
    --output_root PATH --gpu 0 --port 8020 \
    --model_name NAME --model_path PATH \
    --model_backend internvl_local \
    --split_name action|scene --json_path PATH \
    [--task_id_prefix PREFIX] [--prompt_hint_mode minimal] \
    [--clip_count_sweep 2,3,4,5,6,8,10,12] [--nback_max_n 4] \
    [--nback_eval_mode prefix_last_clip_sweep|fixed_last_clip] \
    [--internvl_device_map cuda:0|split] [--internvl_vision_batch_size N] \
    [--internvl_llm_attention eager|sdpa|flash_attention_2]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --model_backend) MODEL_BACKEND_NAME="$2"; shift 2 ;;
    --split_name) SPLIT_NAME="$2"; shift 2 ;;
    --json_path) JSON_PATH="$2"; shift 2 ;;
    --data_root) NBACK_DATA_ROOT_VALUE="$2"; shift 2 ;;
    --task_id_prefix) TASK_ID_PREFIX="$2"; shift 2 ;;
    --prompt_hint_mode) PROMPT_HINT_MODE="$2"; shift 2 ;;
    --clip_count_sweep) CLIP_COUNT_SWEEP="$2"; shift 2 ;;
    --nback_max_n) NBACK_MAX_N="$2"; shift 2 ;;
    --nback_eval_mode) NBACK_EVAL_MODE="$2"; shift 2 ;;
    --frames_per_clip) FRAMES_PER_CLIP="$2"; shift 2 ;;
    --clip_io_workers) CLIP_IO_WORKERS="$2"; shift 2 ;;
    --min_video_file_size_bytes) MIN_VIDEO_FILE_SIZE_BYTES="$2"; shift 2 ;;
    --internvl_input_size) INTERNVL_INPUT_SIZE="$2"; shift 2 ;;
    --internvl_max_num) INTERNVL_MAX_NUM="$2"; shift 2 ;;
    --internvl_total_max_num) INTERNVL_TOTAL_MAX_NUM="$2"; shift 2 ;;
    --internvl_device_map) INTERNVL_DEVICE_MAP="$2"; shift 2 ;;
    --internvl_vision_batch_size) INTERNVL_VISION_BATCH_SIZE="$2"; shift 2 ;;
    --internvl_llm_attention) INTERNVL_LLM_ATTENTION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${OUTPUT_ROOT}" || -z "${JSON_PATH}" ]]; then
  usage
  exit 1
fi

if [[ "${MODEL_BACKEND_NAME}" != "internvl_local" ]]; then
  echo "[ERROR] unsupported --model_backend: ${MODEL_BACKEND_NAME}"
  exit 1
fi
if [[ "${NBACK_EVAL_MODE}" != "prefix_last_clip_sweep" && "${NBACK_EVAL_MODE}" != "fixed_last_clip" ]]; then
  echo "[ERROR] unsupported --nback_eval_mode: ${NBACK_EVAL_MODE}"
  exit 1
fi

MODEL_SLUG="$(sanitize_name "${MODEL_NAME}")"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/results"
QUEUE_LOG="${OUTPUT_ROOT}/logs/queue.log"
RUN_LOG="${OUTPUT_ROOT}/logs/run.log"
RUN_MANIFEST="${OUTPUT_ROOT}/run_manifest.json"
RESULTS_DIR="${OUTPUT_ROOT}/results"
RESULTS_FILE="${RESULTS_DIR}/results.json"
SUMMARY_FILE="${RESULTS_DIR}/analysis/summary.json"

exec > >(tee -a "${QUEUE_LOG}") 2>&1

log() {
  echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*"
}

write_manifest() {
  python3 - <<'PY' "${RUN_MANIFEST}" "${OUTPUT_ROOT}" "${GPU}" "${PORT}" "${MODEL_NAME}" "${MODEL_PATH}" "${MODEL_BACKEND_NAME}" "${SPLIT_NAME}" "${JSON_PATH}" "${PROMPT_HINT_MODE}" "${CLIP_COUNT_SWEEP}" "${NBACK_MAX_N}" "${NBACK_EVAL_MODE}" "${TASK_ID_PREFIX}" "${INTERNVL_INPUT_SIZE}" "${INTERNVL_MAX_NUM}" "${INTERNVL_TOTAL_MAX_NUM}" "${INTERNVL_DEVICE_MAP}" "${INTERNVL_VISION_BATCH_SIZE}" "${INTERNVL_LLM_ATTENTION}" "${NBACK_DATA_ROOT_VALUE}"
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
payload = {
    "output_root": sys.argv[2],
    "gpu": int(sys.argv[3]) if sys.argv[3].isdigit() else sys.argv[3],
    "gpu_ids": [int(part) for part in sys.argv[3].split(",") if part.strip()],
    "port": int(sys.argv[4]),
    "model_name": sys.argv[5],
    "model_path": sys.argv[6],
    "model_backend": sys.argv[7],
    "split_name": sys.argv[8],
    "json_path": sys.argv[9],
    "prompt_hint_mode": sys.argv[10],
    "clip_count_sweep": [int(part) for part in sys.argv[11].split(",") if part.strip()],
    "nback_max_n": int(sys.argv[12]),
    "nback_eval_mode": sys.argv[13],
    "task_id_prefix": sys.argv[14] or None,
    "internvl_input_size": int(sys.argv[15]),
    "internvl_max_num": int(sys.argv[16]),
    "internvl_total_max_num": int(sys.argv[17]),
    "internvl_device_map": sys.argv[18],
    "internvl_vision_batch_size": int(sys.argv[19]),
    "internvl_llm_attention": sys.argv[20] or None,
    "nback_data_root": sys.argv[21] or None,
}
manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
PY
}

write_manifest
log "Using local InternVL inference"

set +e
(
  export CUDA_VISIBLE_DEVICES="${GPU}"
  export MODEL_NAME="${MODEL_NAME}"
  export MODEL_BACKEND="${MODEL_BACKEND_NAME}"
  export FRAMES_PER_CLIP="${FRAMES_PER_CLIP}"
  export NBACK_PROMPT_HINT_MODE="${PROMPT_HINT_MODE}"
  export NBACK_EVAL_MODE="${NBACK_EVAL_MODE}"
  export NBACK_CLIP_COUNT_SWEEP="${CLIP_COUNT_SWEEP}"
  export NBACK_MAX_N="${NBACK_MAX_N}"
  export CLIP_IO_WORKERS="${CLIP_IO_WORKERS}"
  export MIN_VIDEO_FILE_SIZE_BYTES="${MIN_VIDEO_FILE_SIZE_BYTES}"
  export TARGET_QA_COUNT="0"
  export MAX_TASKS="0"
  export JSON_PATH="${JSON_PATH}"
  if [[ -n "${NBACK_DATA_ROOT_VALUE}" ]]; then
    export NBACK_DATA_ROOT="${NBACK_DATA_ROOT_VALUE}"
  fi
  export NBACK_RESULTS_DIR="${RESULTS_DIR}"
  export OUTPUT_FILE="${RESULTS_FILE}"
  export PROGRESS_FILE="${RESULTS_DIR}/test_progress.json"
  export SELECTION_FILE="${RESULTS_DIR}/selection_manifest.json"
  export NBACK_TASK_ID_PREFIX="${TASK_ID_PREFIX}"
  export INTERNVL_MODEL_PATH="${MODEL_PATH}"
  export INTERNVL_INPUT_SIZE="${INTERNVL_INPUT_SIZE}"
  export INTERNVL_MAX_NUM="${INTERNVL_MAX_NUM}"
  export INTERNVL_TOTAL_MAX_NUM="${INTERNVL_TOTAL_MAX_NUM}"
  export INTERNVL_DEVICE_MAP="${INTERNVL_DEVICE_MAP}"
  export INTERNVL_VISION_BATCH_SIZE="${INTERNVL_VISION_BATCH_SIZE}"
  export INTERNVL_LLM_ATTENTION="${INTERNVL_LLM_ATTENTION}"
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  "${PYTHON_BIN}" -u n-back/test_N_back.py
) 2>&1 | tee "${RUN_LOG}"
run_status=${PIPESTATUS[0]}
set -e

if [[ ! -f "${RESULTS_FILE}" ]]; then
  log "[ERROR] missing results: ${RESULTS_FILE}"
  exit 1
fi

"${PYTHON_BIN}" -u n-back/analyze_n_back_results.py --results "${RESULTS_FILE}"
analyze_status=$?
if [[ ${run_status} -ne 0 || ${analyze_status} -ne 0 ]]; then
  log "[ERROR] run_status=${run_status} analyze_status=${analyze_status}"
  exit 1
fi

log "[DONE] split=${SPLIT_NAME} summary=${SUMMARY_FILE}"
