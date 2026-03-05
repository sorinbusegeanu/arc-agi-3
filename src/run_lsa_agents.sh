#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXEC="python"
if [[ -x "/home/zodrak/zod/.venv/bin/python" ]]; then
  PYTHON_EXEC="/home/zodrak/zod/.venv/bin/python"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 --agent bootstrap_explorer [args...]" >&2
  exit 1
fi

"$PYTHON_EXEC" -m llm_stack_agentic.lsa_entrypoint "$@"
