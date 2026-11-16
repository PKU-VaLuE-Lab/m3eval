#!/usr/bin/env bash
# Sourced by evaluation entry points; select one explicit, isolated environment.
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
  elif [[ -n "${FINAL_PAPER_CONDA_ENV:-}" && -x "${FINAL_PAPER_CONDA_ENV}/bin/python" ]]; then
    PYTHON_BIN="${FINAL_PAPER_CONDA_ENV}/bin/python"
  else
    echo "[ERROR] Run 'uv sync --locked' in the repository root, or set PYTHON_BIN to an isolated environment." >&2
    exit 1
  fi
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
export PYTHON_BIN
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${PROJECT_ROOT}/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"
