#!/usr/bin/env bash
# Run the main tasks and supplements with the pinned release inputs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${ROOT}"
DATA_ROOT="${M3EVAL_DATA_ROOT:-${ROOT}/data/m3eval}"
MODEL_PATH="${INTERNVL_MODEL_PATH:-${ROOT}/models/InternVL3_5-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/lmms-eval/output/open_source_runs/internvl35_8b}"
GPUS="${GPU_IDS:-0,1}"
LIMIT_PER_TASK="0"
NBACK_TESTS_PER_SPLIT="32"
STAGE="all"
TASK="m3eval"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --model-path|--model_path) MODEL_PATH="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --limit-per-task) LIMIT_PER_TASK="$2"; shift 2 ;;
    --nback-tests-per-split) NBACK_TESTS_PER_SPLIT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash lmms-eval/scripts/run_m3eval_internvl.sh --model-path PATH [--data-root PATH] [--output-root PATH] [--gpus 0,1] [--task m3eval|m3eval_memory_interference|m3eval_split_screen|m3eval_interleaved|m3eval_nback] [--stage all|main|supplement|nback] [--limit-per-task N] [--nback-tests-per-split N]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "${STAGE}" in all|main|supplement|nback) ;; *) echo "Invalid --stage: ${STAGE}" >&2; exit 2 ;; esac
case "${TASK}" in
  m3eval|m3eval_memory_interference|m3eval_split_screen|m3eval_interleaved|m3eval_nback) ;;
  *) echo "Invalid --task: ${TASK}" >&2; exit 2 ;;
esac
if [[ "${TASK}" == "m3eval_nback" && "${STAGE}" != "all" && "${STAGE}" != "nback" ]] || \
   [[ "${TASK}" != "m3eval" && "${TASK}" != "m3eval_nback" && "${STAGE}" == "nback" ]] || \
   [[ "${TASK}" == "m3eval_memory_interference" && "${STAGE}" == "supplement" ]]; then
  echo "The selected task has no questions in --stage ${STAGE}" >&2
  exit 2
fi
if [[ ! "${LIMIT_PER_TASK}" =~ ^[0-9]+$ || ! "${NBACK_TESTS_PER_SPLIT}" =~ ^[0-9]+$ ]] || \
   (( NBACK_TESTS_PER_SPLIT < 1 || NBACK_TESTS_PER_SPLIT > 32 )); then
  echo "--limit-per-task must be non-negative; --nback-tests-per-split must be in 1..32" >&2
  exit 2
fi
if [[ ! "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "--gpus must be a comma-separated list of GPU indices" >&2
  exit 2
fi
source "${ROOT}/scripts/open_source/python_runtime.sh"
DATA_ROOT="$("${PYTHON_BIN}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "${DATA_ROOT}")"
MODEL_PATH="$("${PYTHON_BIN}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "${MODEL_PATH}")"
OUTPUT_ROOT="$("${PYTHON_BIN}" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "${OUTPUT_ROOT}")"
cd "${ROOT}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
for (( i=0; i<${#GPU_ARRAY[@]}; i++ )); do
  for (( j=0; j<i; j++ )); do
    if [[ "${GPU_ARRAY[i]}" == "${GPU_ARRAY[j]}" ]]; then
      echo "--gpus must not contain duplicate GPU indices" >&2
      exit 2
    fi
  done
done
if (( ${#GPU_ARRAY[@]} == 0 )); then
  echo "--gpus must select at least one GPU" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}/run_profile.json" ]]; then
  echo "[ERROR] Output already contains a run. Choose a new --output-root: ${OUTPUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
export M3EVAL_DATA_ROOT="${DATA_ROOT}"
export VIDEO_MEMORY_EVAL_DATA_ROOT="${DATA_ROOT}"
export VIDEO_MEMORY_EVAL_CONFIDENCE="off"
export VIDEO_MEMORY_EVAL_FORMAT_MODE="strict_final_line"
export M3EVAL_PAPER_PROFILE="internvl35-paper-2026"
export INTERNVL_USE_FLASH_ATTN="0"
export FINAL_PAPER_MODEL_PATH="${MODEL_PATH}"
export FINAL_PAPER_INTERNVL_MODEL_PATH="${MODEL_PATH}"
export FINAL_PAPER_GPUS="${GPUS}"
export TOKENIZERS_PARALLELISM="false"

"${PYTHON_BIN}" - "${OUTPUT_ROOT}/run_profile.json" "${STAGE}" "${DATA_ROOT}" "${MODEL_PATH}" "${GPUS}" "${TASK}" "${LIMIT_PER_TASK}" "${NBACK_TESTS_PER_SPLIT}" <<'PY'
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path
packages = ('torch', 'torchvision', 'transformers', 'accelerate', 'decord', 'numpy', 'pillow', 'datasets')
gpus = [int(gpu) for gpu in sys.argv[5].split(',')]
payload = {
    'profile': 'internvl35-paper-2026', 'stage': sys.argv[2],
    'data_root': sys.argv[3], 'model_path': sys.argv[4], 'gpus': sys.argv[5],
    'task': sys.argv[6], 'limit_per_task': int(sys.argv[7]), 'nback_tests_per_split': int(sys.argv[8]),
    'python': sys.version, 'packages': {p: metadata.version(p) for p in packages},
    'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
    'main': {'input_size': 280, 'frames': 96, 'max_num': 1, 'attention': 'eager', 'vision_batch_size': 0, 'gpus': gpus},
    'supplement': {'input_size': 280, 'frames': 96, 'max_num': 1, 'attention': 'sdpa', 'vision_batch_size': 8, 'gpus': gpus[:2]},
    'nback': {'input_size': 280, 'frames_per_clip': 8, 'max_num': 1, 'attention': 'sdpa', 'vision_batch_size': 8, 'gpus': gpus,
              'gpu_groups': [gpus[i:i+2] for i in range(0, len(gpus), 2)] if len(gpus) > 1 and len(gpus) % 2 == 0 else [[g] for g in gpus]},
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + '\n')
PY

run_non_nback_stage() {
  local stage="$1" attention="$2" vision_batch="$3" device_map="$4"
  local stage_root="${OUTPUT_ROOT}/${stage}"
  local stage_gpus="${GPUS}"
  if [[ "${stage}" == "supplement" && ${#GPU_ARRAY[@]} -gt 2 ]]; then
    stage_gpus="${GPU_ARRAY[0]},${GPU_ARRAY[1]}"
  fi
  "${PYTHON_BIN}" "${ROOT}/scripts/open_source/prepare_m3eval_qa.py" \
    --data-root "${DATA_ROOT}" --output-qa-root "${stage_root}/qa_root" \
    --profile paper-2026 --stage "${stage}" --task "${TASK}" --limit-per-task "${LIMIT_PER_TASK}"
  "${PYTHON_BIN}" "${ROOT}/scripts/open_source/validate_paper_inputs.py" --qa-root "${stage_root}/qa_root"
  bash "${ROOT}/scripts/open_source/run_non_nback_internvl35_8b.sh" \
    --qa_root "${stage_root}/qa_root" --output "${stage_root}/eval" \
    --base_model_path "${MODEL_PATH}" --model_type internvl3_5 --gpus "${stage_gpus}" \
    --internvl_device_map "${device_map}" --internvl_vision_batch_size "${vision_batch}" \
    --internvl_llm_attention "${attention}" --max_frames_num 96 \
    --internvl_max_num 1 --internvl_total_max_num 96 --internvl_input_size 280 --dry_run no
}

if [[ "${TASK}" != "m3eval_nback" && ( "${STAGE}" == "all" || "${STAGE}" == "main" ) ]]; then
  run_non_nback_stage main eager 0 cuda:0
fi
if [[ "${TASK}" != "m3eval_nback" && "${TASK}" != "m3eval_memory_interference" && ( "${STAGE}" == "all" || "${STAGE}" == "supplement" ) ]]; then
  supplement_map="cuda:0"
  if (( ${#GPU_ARRAY[@]} > 1 )); then supplement_map="split"; fi
  run_non_nback_stage supplement sdpa 8 "${supplement_map}"
fi
if [[ ( "${TASK}" == "m3eval" || "${TASK}" == "m3eval_nback" ) && ( "${STAGE}" == "all" || "${STAGE}" == "nback" ) ]]; then
  export FINAL_PAPER_INTERNVL_DEVICE_MAP="cuda:0"
  export FINAL_PAPER_INTERNVL_PAIR_GPUS="0"
  if (( ${#GPU_ARRAY[@]} > 1 && ${#GPU_ARRAY[@]} % 2 == 0 )); then
    export FINAL_PAPER_INTERNVL_DEVICE_MAP="split"
    export FINAL_PAPER_INTERNVL_PAIR_GPUS="1"
  fi
  export FINAL_PAPER_INTERNVL_VISION_BATCH_SIZE="8"
  export FINAL_PAPER_INTERNVL_LLM_ATTENTION="sdpa"
  ports=()
  for (( i=0; i<${#GPU_ARRAY[@]}; i++ )); do ports+=("$((8210+i))"); done
  port_csv="$(IFS=,; echo "${ports[*]}")"
  bash "${ROOT}/scripts/open_source/run_nback_internvl35_8b.sh" \
    --output_root "${OUTPUT_ROOT}/nback" --gpus "${GPUS}" --ports "${port_csv}" \
    --tests_per_split "${NBACK_TESTS_PER_SPLIT}" --clip_count_sweep 2,3,4,5,6,8,10,12 \
    --nback_max_n 4 --frames_per_clip 8
fi

summary_args=()
if [[ "${TASK}" == "m3eval" && "${STAGE}" == "all" && "${LIMIT_PER_TASK}" == "0" && "${NBACK_TESTS_PER_SPLIT}" == "32" ]]; then
  summary_args+=(--require-full)
fi
"${PYTHON_BIN}" "${ROOT}/scripts/open_source/summarize_paper_run.py" \
  --run-root "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/verified_metrics.json" "${summary_args[@]}"
