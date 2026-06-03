#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL_PATH=""
MODEL_TYPE="vllm"
TASK="m3eval"
OUTPUT_DIR="lmms-eval/output/m3eval"
BATCH_SIZE=1
LIMIT=""
GPUS=""
TP="1"
GPU_MEM="0.90"
MAX_MODEL_LEN="65536"
MAX_FRAME_NUM="96"
DISABLE_THINKING="true"
ENFORCE_EAGER="true"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --model_type) MODEL_TYPE="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --gpu_mem) GPU_MEM="$2"; shift 2 ;;
        --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max_frame_num) MAX_FRAME_NUM="$2"; shift 2 ;;
        --disable_thinking) DISABLE_THINKING="$2"; shift 2 ;;
        *)
            echo "Unknown arg: $1"
            exit 1
            ;;
    esac
done

if [[ -z "${MODEL_PATH}" ]]; then
    echo "Usage: bash lmms-eval/scripts/run_m3eval_vllm.sh --model_path /path/to/model [--task m3eval]"
    exit 1
fi

if [[ -n "${GPUS}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS}"
fi

export PYTHONPATH="${PWD}/lmms-eval:${PYTHONPATH:-}"

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

mkdir -p "${OUTPUT_DIR}"
ACCEL_PORT=${ACCEL_PORT:-29571}

accelerate launch \
    --num_processes=1 \
    --main_process_port="${ACCEL_PORT}" \
    -m lmms_eval \
    --model "${MODEL_TYPE}" \
    --model_args "model=${MODEL_PATH},tensor_parallel_size=${TP},gpu_memory_utilization=${GPU_MEM},max_model_len=${MAX_MODEL_LEN},max_frame_num=${MAX_FRAME_NUM},disable_thinking=${DISABLE_THINKING},enforce_eager=${ENFORCE_EAGER},disable_log_stats=True" \
    --tasks "${TASK}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_DIR}" \
    --log_samples \
    --verbosity INFO \
    "${LIMIT_ARGS[@]}"
