#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL_PATH=""
MODEL_TYPE="vllm"
TASK="m3eval"
OUTPUT_DIR="lmms-eval/output/m3eval"
BATCH_SIZE=1
LIMIT=""
OFFSET=""
GPUS=""
NUM_PROCESSES=""
TP="1"
GPU_MEM="0.90"
MAX_MODEL_LEN="65536"
MAX_FRAME_NUM="96"
DISABLE_THINKING="true"
ENFORCE_EAGER="true"

usage() {
    cat <<'EOF'
Usage: bash lmms-eval/scripts/run_m3eval_sharded_vllm.sh --model_path /path/to/model [options]

Options:
  --model_path PATH       Local or Hugging Face model path. Required.
  --model_type NAME       lmms-eval model backend. Default: vllm.
  --task NAME             Task name. Default: m3eval.
  --output DIR            Output directory. Default: lmms-eval/output/m3eval.
  --batch_size N          Batch size per process. Default: 1.
  --gpus LIST             CUDA_VISIBLE_DEVICES value, e.g. 0,1,2,3.
  --num_processes N       Number of accelerate worker processes. Defaults to number of --gpus entries, or 1.
  --limit N               Optional subset size for debugging.
  --offset N              Optional dataset offset for debugging/resume shards.
  --tp N                  vLLM tensor parallel size. Default: 1. Must stay 1 when num_processes > 1.
  --gpu_mem F             vLLM gpu_memory_utilization. Default: 0.90.
  --max_model_len N       vLLM max_model_len. Default: 65536.
  --max_frame_num N       Maximum video frames. Default: 96.
  --disable_thinking BOOL Default: true.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --model_type) MODEL_TYPE="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --offset) OFFSET="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --num_processes) NUM_PROCESSES="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --gpu_mem) GPU_MEM="$2"; shift 2 ;;
        --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max_frame_num) MAX_FRAME_NUM="$2"; shift 2 ;;
        --disable_thinking) DISABLE_THINKING="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown arg: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "${MODEL_PATH}" ]]; then
    usage
    exit 1
fi

if [[ -n "${GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS}"
fi

if [[ -z "${NUM_PROCESSES}" ]]; then
    if [[ -n "${GPUS}" ]]; then
        IFS=',' read -ra GPU_ITEMS <<< "${GPUS}"
        NUM_PROCESSES="${#GPU_ITEMS[@]}"
    else
        NUM_PROCESSES="1"
    fi
fi

if [[ "${NUM_PROCESSES}" -gt 1 && "${TP}" != "1" ]]; then
    echo "This data-parallel runner requires --tp 1 when --num_processes > 1."
    exit 1
fi

if [[ -n "${LIMIT}" && "${LIMIT}" =~ ^[0-9]+$ && "${NUM_PROCESSES}" -gt "${LIMIT}" ]]; then
    echo "For multi-GPU data-parallel smoke tests, --limit must be at least --num_processes."
    echo "Current values: --limit ${LIMIT}, --num_processes ${NUM_PROCESSES}."
    echo "Use --limit ${NUM_PROCESSES} or larger, or set --num_processes 1 for a single-sample smoke run."
    exit 1
fi

export PYTHONPATH="${PWD}/lmms-eval:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ -n "${LIMIT}" ]]; then
    EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${OFFSET}" ]]; then
    EXTRA_ARGS+=(--offset "${OFFSET}")
fi

mkdir -p "${OUTPUT_DIR}"
ACCEL_PORT=${ACCEL_PORT:-29571}

accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --main_process_port="${ACCEL_PORT}" \
    -m lmms_eval \
    --model "${MODEL_TYPE}" \
    --model_args "model=${MODEL_PATH},tensor_parallel_size=${TP},gpu_memory_utilization=${GPU_MEM},max_model_len=${MAX_MODEL_LEN},max_frame_num=${MAX_FRAME_NUM},disable_thinking=${DISABLE_THINKING},enforce_eager=${ENFORCE_EAGER},disable_log_stats=True" \
    --tasks "${TASK}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_DIR}" \
    --log_samples \
    --verbosity INFO \
    "${EXTRA_ARGS[@]}"
