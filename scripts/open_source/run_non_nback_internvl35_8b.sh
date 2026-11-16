#!/usr/bin/env bash
# Run the three non-N-back M3Eval task families with InternVL3.5-8B.
set -euo pipefail

# The public wrapper passes finalized QA files, uses 96 frames, and writes
# runtime outputs below lmms-eval/output/.

cd "$(dirname "$0")/../.."
PROJECT_ROOT="$(pwd)"

source "${PROJECT_ROOT}/scripts/open_source/python_runtime.sh"

QA_ROOT="${FINAL_PAPER_QA_ROOT:-}"
OUTPUT_DIR="${FINAL_PAPER_OUTPUT_ROOT:-lmms-eval/output/final_paper_rerun}/non_nback_open_model_739"
BASE_MODEL_PATH="${FINAL_PAPER_MODEL_PATH:-models/InternVL3_5-8B}"
MODEL_TYPE=""
MAIN_TASK=""
VERSION="current"
GPUS="${FINAL_PAPER_GPUS:-0,1}"
BATCH_SIZE="1"
GPU_MEM="0.85"
MAX_MODEL_LEN="65536"
MAX_FRAMES_NUM="96"
EFFECTIVE_MAX_MODEL_LEN=""
ENFORCE_EAGER="yes"
DISABLE_THINKING="no"
INTERNVL_NUM_FRAME="96"
INTERNVL_MAX_NUM="1"
INTERNVL_TOTAL_MAX_NUM="96"
INTERNVL_INPUT_SIZE="280"
INTERNVL_MODALITY="video"
INTERNVL_DEVICE_MAP="${FINAL_PAPER_INTERNVL_DEVICE_MAP:-cuda:0}"
INTERNVL_VISION_BATCH_SIZE="${FINAL_PAPER_INTERNVL_VISION_BATCH_SIZE:-0}"
INTERNVL_LLM_ATTENTION="${FINAL_PAPER_INTERNVL_LLM_ATTENTION:-}"
INTERNVL_NUM_FRAME_EXPLICIT="no"
GEN_KWARGS=""
LIMIT=""
EFFECTIVE_LIMIT=""
EFFECTIVE_GPUS=""
EFFECTIVE_NUM_PROCESSES=""
VERBOSITY="INFO"
CONFIDENCE="off"
MAIN_PROCESS_PORT=""
DRY_RUN="no"
CUSTOM_TASK_DIR=""
MAIN_TASK="v4_video_memory_eval_instruct"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/open_source/run_non_nback_internvl35_8b.sh \
    [--qa_root PATH] [--output PATH] [--base_model_path PATH] [--version current|six_src_50] [--model_type NAME] \
    [--main_task NAME] \
    [--gpus IDS] [--batch_size N] [--gpu_mem F] [--max_model_len N] \
    [--max_frames_num N] \
    [--enforce_eager yes|no] [--disable_thinking yes|no] \
    [--internvl_num_frame N] [--internvl_max_num N] [--internvl_total_max_num N] [--internvl_input_size N] \
    [--internvl_modality video|image] [--internvl_device_map cuda:0|split|auto|balanced|balanced_low_0|sequential] \
    [--internvl_vision_batch_size N] \
    [--internvl_llm_attention eager|sdpa|flash_attention_2] \
    [--gen_kwargs KEY=VALUE[,KEY=VALUE...]] \
    [--limit N] [--verbosity LEVEL] [--confidence on|off] \
    [--main_process_port PORT] [--dry_run yes|no]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qa_root) QA_ROOT="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --base_model_path) BASE_MODEL_PATH="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --model_type) MODEL_TYPE="$2"; shift 2 ;;
    --main_task) MAIN_TASK="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --gpu_mem) GPU_MEM="$2"; shift 2 ;;
    --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max_frames_num) MAX_FRAMES_NUM="$2"; shift 2 ;;
    --enforce_eager) ENFORCE_EAGER="$2"; shift 2 ;;
    --disable_thinking) DISABLE_THINKING="$2"; shift 2 ;;
    --internvl_num_frame) INTERNVL_NUM_FRAME="$2"; INTERNVL_NUM_FRAME_EXPLICIT="yes"; shift 2 ;;
    --internvl_max_num) INTERNVL_MAX_NUM="$2"; shift 2 ;;
    --internvl_total_max_num) INTERNVL_TOTAL_MAX_NUM="$2"; shift 2 ;;
    --internvl_input_size) INTERNVL_INPUT_SIZE="$2"; shift 2 ;;
    --internvl_modality) INTERNVL_MODALITY="$2"; shift 2 ;;
    --internvl_device_map) INTERNVL_DEVICE_MAP="$2"; shift 2 ;;
    --internvl_vision_batch_size) INTERNVL_VISION_BATCH_SIZE="$2"; shift 2 ;;
    --internvl_llm_attention) INTERNVL_LLM_ATTENTION="$2"; shift 2 ;;
    --gen_kwargs) GEN_KWARGS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --verbosity) VERBOSITY="$2"; shift 2 ;;
    --confidence) CONFIDENCE="$2"; shift 2 ;;
    --main_process_port) MAIN_PROCESS_PORT="$2"; shift 2 ;;
    --dry_run) DRY_RUN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${QA_ROOT}" ]]; then
  QA_ROOT="${FINAL_PAPER_QA_ROOT:-lmms-eval/output/hf_release_full_2026-05-01/qa_root}"
fi

qa_json_has_rows() {
  python3 - "$1" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
except FileNotFoundError:
    print("0")
    raise SystemExit(0)
print("1" if isinstance(payload, list) and len(payload) > 0 else "0")
PY
}

qa_json_row_count() {
  python3 - "$1" <<'PY'
import json
import sys
path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
except FileNotFoundError:
    print("0")
    raise SystemExit(0)
print(len(payload) if isinstance(payload, list) else 0)
PY
}

TASKS_PRESENT=()
TOTAL_DOCS_PRESENT=0
MIN_TASK_DOCS=0
for task in memory_interference split_screen interleaved; do
  path="${QA_ROOT}/${task}/${task}_qa.json"
  row_count="$(qa_json_row_count "${path}")"
  if [[ "${row_count}" =~ ^[0-9]+$ ]] && (( row_count > 0 )); then
    TASKS_PRESENT+=("${task}")
    TOTAL_DOCS_PRESENT=$((TOTAL_DOCS_PRESENT + row_count))
    if (( MIN_TASK_DOCS == 0 || row_count < MIN_TASK_DOCS )); then
      MIN_TASK_DOCS="${row_count}"
    fi
  fi
done

if [[ "${#TASKS_PRESENT[@]}" -eq 0 ]]; then
  echo "[ERROR] No non-empty task QA files found under ${QA_ROOT}"
  exit 1
fi

if [[ -z "${GPUS}" ]]; then
  echo "[ERROR] --gpus must not be empty"
  exit 1
fi

IFS=',' read -ra GPU_ARRAY <<< "${GPUS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "[ERROR] Could not parse a valid GPU count from --gpus"
  exit 1
fi

EFFECTIVE_GPUS="${GPUS}"
EFFECTIVE_NUM_PROCESSES="${NUM_GPUS}"
EFFECTIVE_LIMIT="${LIMIT}"
EFFECTIVE_TOTAL_DOCS="${TOTAL_DOCS_PRESENT}"
if [[ -n "${LIMIT}" ]]; then
  if [[ ! "${LIMIT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --limit must be a positive integer"
    exit 1
  fi
  if (( LIMIT < EFFECTIVE_TOTAL_DOCS )); then
    EFFECTIVE_TOTAL_DOCS="${LIMIT}"
  fi
  if (( LIMIT < NUM_GPUS )); then
    EFFECTIVE_NUM_PROCESSES="${LIMIT}"
    EFFECTIVE_GPUS="$(IFS=,; echo "${GPU_ARRAY[*]:0:${EFFECTIVE_NUM_PROCESSES}}")"
    echo "[WARN] --limit=${LIMIT} is below num_gpus=${NUM_GPUS}; using ${EFFECTIVE_NUM_PROCESSES} workers on GPUs ${EFFECTIVE_GPUS} to avoid empty ranks"
  fi
fi

if (( MIN_TASK_DOCS < EFFECTIVE_NUM_PROCESSES )); then
  EFFECTIVE_NUM_PROCESSES="${MIN_TASK_DOCS}"
  EFFECTIVE_GPUS="$(IFS=,; echo "${GPU_ARRAY[*]:0:${EFFECTIVE_NUM_PROCESSES}}")"
  echo "[INFO] Smallest task contains ${MIN_TASK_DOCS} samples; using ${EFFECTIVE_NUM_PROCESSES} data-parallel workers on GPUs ${EFFECTIVE_GPUS}"
fi

if [[ -z "${MODEL_TYPE}" ]]; then
  MODEL_TYPE="internvl3_5"
fi

if [[ "${MODEL_TYPE}" == "internvl3" || "${MODEL_TYPE}" == "internvl3_5" ]]; then
  if [[ "${INTERNVL_DEVICE_MAP}" != cuda:* ]]; then
    EFFECTIVE_GPUS="${GPUS}"
    EFFECTIVE_NUM_PROCESSES="1"
    echo "[INFO] InternVL device_map=${INTERNVL_DEVICE_MAP}: keep GPUs ${EFFECTIVE_GPUS} visible and use one model-parallel process"
  fi
fi

EFFECTIVE_MAX_MODEL_LEN="${MAX_MODEL_LEN}"

if [[ -z "${MAIN_TASK}" ]]; then
  case "${BASE_MODEL_PATH}" in
    *Thinking*|*thinking*)
      MAIN_TASK="v4_video_memory_eval" ;;
    *)
      MAIN_TASK="v4_video_memory_eval_instruct" ;;
  esac
fi

case "${MAIN_TASK}" in
  v4_video_memory_eval_instruct|v4_video_memory_eval) ;;
  *)
    echo "[ERROR] --main_task must be v4_video_memory_eval_instruct or v4_video_memory_eval"
    exit 1 ;;
esac

if [[ "${CONFIDENCE}" != "on" && "${CONFIDENCE}" != "off" ]]; then
  echo "[ERROR] --confidence must be on or off"
  exit 1
fi

if [[ "${ENFORCE_EAGER}" != "yes" && "${ENFORCE_EAGER}" != "no" ]]; then
  echo "[ERROR] --enforce_eager must be yes or no"
  exit 1
fi

if [[ "${DISABLE_THINKING}" != "yes" && "${DISABLE_THINKING}" != "no" ]]; then
  echo "[ERROR] --disable_thinking must be yes or no"
  exit 1
fi

if [[ "${INTERNVL_MODALITY}" != "video" && "${INTERNVL_MODALITY}" != "image" ]]; then
  echo "[ERROR] --internvl_modality must be video or image"
  exit 1
fi

if [[ "${INTERNVL_NUM_FRAME_EXPLICIT}" != "yes" ]]; then
  INTERNVL_NUM_FRAME="${MAX_FRAMES_NUM}"
fi

RUN_TS="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${OUTPUT_DIR}/logs"
ANALYSIS_DIR="${OUTPUT_DIR}/analysis"
mkdir -p "${LOG_DIR}"
export VIDEO_MEMORY_EVAL_SHARED_BAD_VIDEOS_REGISTRY="${VIDEO_MEMORY_EVAL_SHARED_BAD_VIDEOS_REGISTRY:-${ANALYSIS_DIR}/shared_bad_videos.json}"
LOG_FILE="${LOG_DIR}/run_${RUN_TS}.log"

abs_path() {
  python3 - "$1" <<'PY'
import os
import sys
print(os.path.realpath(sys.argv[1]))
PY
}

write_runtime_task() {
  local path="$1"
  local task_name="$2"
  local qa_json="$3"
  local metric_name="$4"
  local doc_to_text_fn="$5"
  local process_results_fn="$6"
  local total_frames="$7"
  local response_style="$8"
  local missing_format_metric_name="${metric_name/_score/_missing_required_answer_format}"
  cat > "${path}" <<EOF
task: "${task_name}"
include: ${PROJECT_ROOT}/lmms-eval/lmms_eval/tasks/m3eval/_generation_kwargs_yaml
dataset_path: json
dataset_kwargs:
  data_files:
    test: ${qa_json}
test_split: test
output_type: generate_until

doc_to_visual: !function lmms_eval.tasks.m3eval.utils.v4_doc_to_visual
doc_to_text: !function ${doc_to_text_fn}
doc_to_target: "answer"

process_results: !function ${process_results_fn}
metric_list:
  - metric: ${metric_name}
    aggregation: mean
    higher_is_better: true
  - metric: ${missing_format_metric_name}
    aggregation: mean
    higher_is_better: false

lmms_eval_specific_kwargs:
  default:
    total_frames: ${total_frames}
    pre_prompt: ""
    response_style: "${response_style}"

metadata:
  version: 0.4
  total_frames: ${total_frames}
EOF
}

runtime_task_spec() {
  local task="$1"
  local suffix=""
  local doc_to_text_fn="lmms_eval.tasks.m3eval.utils.v4_doc_to_text"
  local process_results_fn="lmms_eval.tasks.m3eval.utils.v4_process_results"
  local response_style="direct"
  local resolved_total_frames=""
  if [[ "${MAIN_TASK}" == "v4_video_memory_eval" ]]; then
    suffix="_cot"
    doc_to_text_fn="lmms_eval.tasks.m3eval.utils.v4_doc_to_text_cot"
    process_results_fn="lmms_eval.tasks.m3eval.utils.v4_process_results_cot"
    response_style="cot"
  fi

  case "${task}" in
    memory_interference)
      resolved_total_frames="${MAX_FRAMES_NUM:-96}"
      printf '%s\n' \
        "v4_runtime_memory_interference${suffix}" \
        "v4_memory_interference${suffix}_score" \
        "${doc_to_text_fn}" \
        "${process_results_fn}" \
        "${resolved_total_frames}" \
        "${response_style}"
      ;;
    split_screen)
      resolved_total_frames="${MAX_FRAMES_NUM:-96}"
      printf '%s\n' \
        "v4_runtime_split_screen${suffix}" \
        "v4_split_screen${suffix}_score" \
        "${doc_to_text_fn}" \
        "${process_results_fn}" \
        "${resolved_total_frames}" \
        "${response_style}"
      ;;
    interleaved)
      resolved_total_frames="${MAX_FRAMES_NUM:-96}"
      printf '%s\n' \
        "v4_runtime_interleaved${suffix}" \
        "v4_interleaved${suffix}_score" \
        "${doc_to_text_fn}" \
        "${process_results_fn}" \
        "${resolved_total_frames}" \
        "${response_style}"
      ;;
    *)
      echo "[ERROR] unsupported runtime task: ${task}" >&2
      exit 1
      ;;
  esac
}

prepare_runtime_tasks() {
  local qa_root_abs
  qa_root_abs="$(abs_path "${QA_ROOT}")"
  CUSTOM_TASK_DIR="${OUTPUT_DIR}/runtime_tasks_${RUN_TS}"
  mkdir -p "${CUSTOM_TASK_DIR}"

  local runtime_tasks=()
  for task in "${TASKS_PRESENT[@]}"; do
    mapfile -t task_spec < <(runtime_task_spec "${task}")
    write_runtime_task \
      "${CUSTOM_TASK_DIR}/${task_spec[0]}.yaml" \
      "${task_spec[0]}" \
      "${qa_root_abs}/${task}/${task}_qa.json" \
      "${task_spec[1]}" \
      "${task_spec[2]}" \
      "${task_spec[3]}" \
      "${task_spec[4]}" \
      "${task_spec[5]}"
    runtime_tasks+=("${task_spec[0]}")
  done

  if [[ "${#runtime_tasks[@]}" -eq 1 ]]; then
    MAIN_TASK="${runtime_tasks[0]}"
    return
  fi

  cat > "${CUSTOM_TASK_DIR}/runtime_group.yaml" <<EOF
group: v4_runtime_group
task:
$(for task_name in "${runtime_tasks[@]}"; do echo "  - ${task_name}"; done)
EOF

  MAIN_TASK="v4_runtime_group"
}

prepare_runtime_tasks

echo "[INFO] use finalized QA as-is; skip finalize/migrate mutations"

if [[ "${MODEL_TYPE}" != "internvl3" && "${MODEL_TYPE}" != "internvl3_5" ]]; then
  echo "[ERROR] this release supports only InternVL3/InternVL3.5" >&2
  exit 1
fi
if [[ "${BATCH_SIZE}" != "1" ]]; then
  echo "[ERROR] InternVL requires --batch_size 1" >&2
  exit 1
fi
MODEL_ARGS="pretrained=${BASE_MODEL_PATH},modality=${INTERNVL_MODALITY},num_frame=${INTERNVL_NUM_FRAME},max_num=${INTERNVL_MAX_NUM},total_max_num=${INTERNVL_TOTAL_MAX_NUM},input_size=${INTERNVL_INPUT_SIZE},device_map=${INTERNVL_DEVICE_MAP},vision_batch_size=${INTERNVL_VISION_BATCH_SIZE},use_flash_attn=False"
if [[ -n "${INTERNVL_LLM_ATTENTION}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},llm_attention_implementation=${INTERNVL_LLM_ATTENTION}"
fi

if [[ -z "${MAIN_PROCESS_PORT}" ]]; then
  MAIN_PROCESS_PORT="$(shuf -i 29500-29999 -n 1)"
fi

limit_args=()
if [[ -n "${EFFECTIVE_LIMIT}" ]]; then
  limit_args=(--limit "${EFFECTIVE_LIMIT}")
fi

include_args=()
if [[ -n "${CUSTOM_TASK_DIR}" ]]; then
  include_args=(--include_path "${CUSTOM_TASK_DIR}")
fi

CMD=(
  "${PYTHON_BIN}" -m accelerate.commands.launch
  --num_processes "${EFFECTIVE_NUM_PROCESSES}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  -m lmms_eval
  --model "${MODEL_TYPE}"
  --model_args "${MODEL_ARGS}"
  --tasks "${MAIN_TASK}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${OUTPUT_DIR}"
  --log_samples
  --verbosity "${VERBOSITY}"
)
if [[ "${#include_args[@]}" -gt 0 ]]; then
  CMD+=("${include_args[@]}")
fi
if [[ -n "${GEN_KWARGS}" ]]; then
  CMD+=(--gen_kwargs "${GEN_KWARGS}")
fi
if [[ "${#limit_args[@]}" -gt 0 ]]; then
  CMD+=("${limit_args[@]}")
fi

{
  echo "[INFO] qa_root=${QA_ROOT}"
  echo "[INFO] output=${OUTPUT_DIR}"
  echo "[INFO] model=${BASE_MODEL_PATH}"
  echo "[INFO] version=${VERSION}"
  echo "[INFO] model_type=${MODEL_TYPE}"
  echo "[INFO] active_tasks=${TASKS_PRESENT[*]}"
  echo "[INFO] gpus=${EFFECTIVE_GPUS}"
  echo "[INFO] num_processes=${EFFECTIVE_NUM_PROCESSES}"
  echo "[INFO] batch_size=${BATCH_SIZE}"
  echo "[INFO] gpu_mem=${GPU_MEM}"
  echo "[INFO] max_model_len=${MAX_MODEL_LEN}"
  if [[ -n "${MAX_FRAMES_NUM}" ]]; then
    echo "[INFO] max_frames_num=${MAX_FRAMES_NUM}"
  fi
  if [[ "${EFFECTIVE_MAX_MODEL_LEN}" != "${MAX_MODEL_LEN}" ]]; then
    echo "[INFO] effective_max_model_len=${EFFECTIVE_MAX_MODEL_LEN}"
  fi
  echo "[INFO] enforce_eager=${ENFORCE_EAGER}"
  echo "[INFO] disable_thinking=${DISABLE_THINKING}"
  if [[ "${MODEL_TYPE}" == "internvl3" || "${MODEL_TYPE}" == "internvl3_5" ]]; then
    echo "[INFO] internvl_modality=${INTERNVL_MODALITY}"
    echo "[INFO] internvl_num_frame=${INTERNVL_NUM_FRAME}"
    echo "[INFO] internvl_max_num=${INTERNVL_MAX_NUM}"
    echo "[INFO] internvl_total_max_num=${INTERNVL_TOTAL_MAX_NUM}"
    echo "[INFO] internvl_input_size=${INTERNVL_INPUT_SIZE}"
    echo "[INFO] internvl_device_map=${INTERNVL_DEVICE_MAP}"
    echo "[INFO] internvl_vision_batch_size=${INTERNVL_VISION_BATCH_SIZE}"
    echo "[INFO] internvl_llm_attention=${INTERNVL_LLM_ATTENTION:-checkpoint_default}"
  fi
  if [[ -n "${GEN_KWARGS}" ]]; then
    echo "[INFO] gen_kwargs=${GEN_KWARGS}"
  fi
  echo "[INFO] confidence=${CONFIDENCE}"
  if [[ -n "${LIMIT}" && "${EFFECTIVE_LIMIT}" != "${LIMIT}" ]]; then
    echo "[INFO] requested_limit=${LIMIT}"
  fi
  if [[ -n "${EFFECTIVE_LIMIT}" ]]; then
    echo "[INFO] effective_limit=${EFFECTIVE_LIMIT}"
  fi
  echo "[INFO] main_task=${MAIN_TASK}"
  echo "[INFO] main_process_port=${MAIN_PROCESS_PORT}"
  if [[ -n "${CUSTOM_TASK_DIR}" ]]; then
    echo "[INFO] custom_task_dir=${CUSTOM_TASK_DIR}"
  fi

  echo "[CHECK] GPU status"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

  echo "[RUN] lmms-eval multi-gpu eval"
  printf '[CMD] CUDA_VISIBLE_DEVICES=%s ' "${EFFECTIVE_GPUS}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} | tee -a "${LOG_FILE}"

if [[ "${DRY_RUN}" == "yes" ]]; then
  echo "[DONE] dry run only" | tee -a "${LOG_FILE}"
  exit 0
fi

exec > >(tee -a "${LOG_FILE}") 2>&1

export CUDA_VISIBLE_DEVICES="${EFFECTIVE_GPUS}"
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export VIDEO_MEMORY_EVAL_CONFIDENCE="${CONFIDENCE}"

"${CMD[@]}"

RESULT_COUNT="$(find "${OUTPUT_DIR}" -maxdepth 2 -name '*_results.json' | wc -l | tr -d ' ')"
if [[ "${RESULT_COUNT}" == "0" ]]; then
  echo "[ERROR] lmms-eval produced no results.json; stopping postprocessing"
  exit 1
fi

echo "[DONE] Multi-GPU evaluation shard completed"
echo "[INFO] output=${OUTPUT_DIR}"
echo "[INFO] log=${LOG_FILE}"
echo "[DONE] raw evaluation outputs are available under ${OUTPUT_DIR}"
