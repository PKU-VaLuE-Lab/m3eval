#!/usr/bin/env bash
# Public M3Eval evaluation entry point; inference uses the frozen InternVL profile.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${ROOT}/scripts/open_source/run_internvl35_8b_paper.sh" "$@"
