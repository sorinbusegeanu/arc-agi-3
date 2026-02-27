#!/usr/bin/env bash
set -euo pipefail

AGENT=""
INPUT=""
OUTDIR=""
SAVE_VIZ=""
FORMAT="both"
GAME=""
SEED="0"
MAX_STEPS="80"
PROBE_STEPS="10"
OP_MODE="offline"
DEBUG=""
SNAPSHOT_EVERY_STEPS="0"
FP_SAVE_MODE="buffer"
ACTION_SCHEMA=""
SIMPLE_REPORT=""
FULL_REPORT=""
TRACE_PATH=""
MECHANIC_REPORT=""
HYPOTHESES_REPORT=""
GOAL_REPORT=""
PLANNER_TRACE=""
SIMPLE_TRACE=""
FULL_TRACE=""
FP_DIR=""
ACTION_SCHEMA_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent)
      AGENT="$2"
      shift 2
      ;;
    --input)
      INPUT="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --save-viz)
      SAVE_VIZ="--save-viz"
      shift 1
      ;;
    --format)
      FORMAT="$2"
      shift 2
      ;;
    --action-schema)
      ACTION_SCHEMA="$2"
      shift 2
      ;;
    --simple)
      SIMPLE_REPORT="$2"
      shift 2
      ;;
    --full)
      FULL_REPORT="$2"
      shift 2
      ;;
    --trace)
      TRACE_PATH="$2"
      shift 2
      ;;
    --mechanic)
      MECHANIC_REPORT="$2"
      shift 2
      ;;
    --hypotheses)
      HYPOTHESES_REPORT="$2"
      shift 2
      ;;
    --goal)
      GOAL_REPORT="$2"
      shift 2
      ;;
    --planner-trace)
      PLANNER_TRACE="$2"
      shift 2
      ;;
    --simple-trace)
      SIMPLE_TRACE="$2"
      shift 2
      ;;
    --full-trace)
      FULL_TRACE="$2"
      shift 2
      ;;
    --fp-dir)
      FP_DIR="$2"
      shift 2
      ;;
    --action-schema)
      ACTION_SCHEMA_PATH="$2"
      shift 2
      ;;
    --game)
      GAME="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --probe-steps)
      PROBE_STEPS="$2"
      shift 2
      ;;
    --op-mode)
      OP_MODE="$2"
      shift 2
      ;;
    --debug)
      DEBUG="--debug"
      shift 1
      ;;
    --snapshot-every-steps)
      SNAPSHOT_EVERY_STEPS="$2"
      shift 2
      ;;
    --fp-save-mode)
      FP_SAVE_MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$AGENT" ]]; then
  echo "Usage: $0 --agent fp_analyst|simple_explorer|full_explorer|rule_proposer|mechanic_classifier|goal_detector|planner|trajectory_summarizer|executable_hypothesis_engine|test_selector|mechanic_synthesizer|swarm [args...]" >&2
  exit 1
fi

PYTHON_EXEC="python"
if [[ -x "/home/zodrak/zod/.venv/bin/python" ]]; then
  PYTHON_EXEC="/home/zodrak/zod/.venv/bin/python"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$AGENT" == "fp_analyst" ]]; then
  if [[ -z "$INPUT" || -z "$OUTDIR" ]]; then
    echo "Usage: $0 --agent fp_analyst --input <frame.json> --outdir <dir> [--save-viz] [--format ascii|json|both]" >&2
    exit 1
  fi
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_EXEC" -m arc_agi_agent.cli --agent "$AGENT" --input "$INPUT" --outdir "$OUTDIR" $SAVE_VIZ --format "$FORMAT"
  exit 0
fi

if [[ "$AGENT" == "simple_explorer" ]]; then
  if [[ -z "$GAME" ]]; then
    echo "Usage: $0 --agent simple_explorer --game <id> [--seed <n>] [--max-steps <n>] [--outdir <dir>] [--save-viz]" >&2
    exit 1
  fi
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_EXEC" -m arc_agi_agent.cli --agent "$AGENT" --game "$GAME" --seed "$SEED" --max-steps "$MAX_STEPS" --op-mode "$OP_MODE" ${OUTDIR:+--outdir "$OUTDIR"} $SAVE_VIZ
  exit 0
fi

if [[ "$AGENT" == "full_explorer" ]]; then
  if [[ -z "$GAME" ]]; then
    echo "Usage: $0 --agent full_explorer --game <id> [--seed <n>] [--max-steps <n>] [--outdir <dir>] [--save-viz]" >&2
    exit 1
  fi
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_EXEC" -m arc_agi_agent.cli --agent "$AGENT" --game "$GAME" --seed "$SEED" --max-steps "$MAX_STEPS" --op-mode "$OP_MODE" ${OUTDIR:+--outdir "$OUTDIR"} $SAVE_VIZ
  exit 0
fi

if [[ "$AGENT" == "rule_proposer" ]]; then
  if [[ -z "$OUTDIR" || -z "$INPUT" ]]; then
    echo "Usage: $0 --agent rule_proposer --input <fp_report.json> --outdir <dir> --action-schema <schema.json> [--simple <report.json>] [--full <report.json>]" >&2
    exit 1
  fi
  if [[ -z "${ACTION_SCHEMA:-}" ]]; then
    echo "Usage: $0 --agent rule_proposer --input <fp_report.json> --outdir <dir> --action-schema <schema.json> [--simple <report.json>] [--full <report.json>]" >&2
    exit 1
  fi
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_EXEC" -m arc_agi_agent.cli --agent "$AGENT" --input_fp "$INPUT" --outdir "$OUTDIR" --action-schema "$ACTION_SCHEMA" ${SIMPLE_REPORT:+--simple "$SIMPLE_REPORT"} ${FULL_REPORT:+--full "$FULL_REPORT"}
  exit 0
fi

if [[ "$AGENT" == "mechanic_classifier" ]]; then
  if [[ -z "$OUTDIR" || -z "$INPUT" ]]; then
    echo "Usage: $0 --agent mechanic_classifier --input <fp_report.json> --outdir <dir> [--action-schema <schema.json>] [--simple <report.json>] [--full <report.json>]" >&2
    exit 1
  fi
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_EXEC" -m arc_agi_agent.cli --agent "$AGENT" --input_fp "$INPUT" --outdir "$OUTDIR" ${ACTION_SCHEMA:+--action-schema "$ACTION_SCHEMA"} ${SIMPLE_REPORT:+--simple "$SIMPLE_REPORT"} ${FULL_REPORT:+--full "$FULL_REPORT"}
  exit 0
fi

if [[ "$AGENT" == "goal_detector" ]]; then
  if [[ -z "$OUTDIR" || -z "$INPUT" ]]; then
    echo "Usage: $0 --agent goal_detector --input <fp_report.json> --outdir <dir> [--trace <trace.jsonl>]" >&2
    exit 1
  fi
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_EXEC" -m arc_agi_agent.cli --agent "$AGENT" --input_fp "$INPUT" --outdir "$OUTDIR" ${TRACE_PATH:+--trace "$TRACE_PATH"}
  exit 0
fi

if [[ "$AGENT" == "planner" ]]; then
  if [[ -z "$GAME" || -z "$OUTDIR" ]]; then
    echo "Usage: $0 --agent planner --game <id> --outdir <dir> [--seed <n>] [--max-steps <n>] [--op-mode online|normal] [--mechanic <report.json>] [--hypotheses <report.json>] [--simple <report.json>] [--full <report.json>] [--goal <report.json>]" >&2
    exit 1
  fi
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_EXEC" -m arc_agi_agent.cli --agent "$AGENT" --game "$GAME" --seed "$SEED" --max-steps "$MAX_STEPS" --op-mode "$OP_MODE" --outdir "$OUTDIR" ${MECHANIC_REPORT:+--mechanic "$MECHANIC_REPORT"} ${HYPOTHESES_REPORT:+--hypotheses "$HYPOTHESES_REPORT"} ${SIMPLE_REPORT:+--simple "$SIMPLE_REPORT"} ${FULL_REPORT:+--full "$FULL_REPORT"} ${GOAL_REPORT:+--goal "$GOAL_REPORT"}
  exit 0
fi

if [[ "$AGENT" == "executable_hypothesis_engine" ]]; then
  if [[ -z "$OUTDIR" || -z "$INPUT" || -z "${ACTION_SCHEMA:-}" ]]; then
    echo "Usage: $0 --agent executable_hypothesis_engine --input <fp_report.json>[,<fp_report2.json>...] --outdir <dir> --action-schema <schema.json> [--simple <report.json>] [--full <report.json>] [--hypotheses <report.json>]" >&2
    exit 1
  fi
  INPUT_ARGS=()
  IFS=',' read -ra FP_PATHS <<< "$INPUT"
  for path in "${FP_PATHS[@]}"; do
    if [[ -n "$path" ]]; then
      INPUT_ARGS+=(--input_fp "$path")
    fi
  done
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_EXEC" -m arc_agi_agent.cli --agent "$AGENT" "${INPUT_ARGS[@]}" --outdir "$OUTDIR" --action-schema "$ACTION_SCHEMA" ${SIMPLE_REPORT:+--simple "$SIMPLE_REPORT"} ${FULL_REPORT:+--full "$FULL_REPORT"} ${HYPOTHESES_REPORT:+--hypotheses "$HYPOTHESES_REPORT"}
  exit 0
fi

if [[ "$AGENT" == "test_selector" ]]; then
  if [[ -z "$OUTDIR" || -z "$INPUT" || -z "${ACTION_SCHEMA:-}" || -z "${HYPOTHESES_REPORT:-}" ]]; then
    echo "Usage: $0 --agent test_selector --input <fp_report.json>[,<fp_report2.json>...] --outdir <dir> --action-schema <schema.json> --hypotheses <report.json> [--simple <report.json>] [--full <report.json>]" >&2
    exit 1
  fi
  INPUT_ARGS=()
  IFS=',' read -ra FP_PATHS <<< "$INPUT"
  for path in "${FP_PATHS[@]}"; do
    if [[ -n "$path" ]]; then
      INPUT_ARGS+=(--input_fp "$path")
    fi
  done
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_EXEC" -m arc_agi_agent.cli --agent "$AGENT" "${INPUT_ARGS[@]}" --outdir "$OUTDIR" --action-schema "$ACTION_SCHEMA" --hypotheses "$HYPOTHESES_REPORT" ${SIMPLE_REPORT:+--simple "$SIMPLE_REPORT"} ${FULL_REPORT:+--full "$FULL_REPORT"}
  exit 0
fi

if [[ "$AGENT" == "mechanic_synthesizer" ]]; then
  if [[ -z "$OUTDIR" || -z "$INPUT" ]]; then
    echo "Usage: $0 --agent mechanic_synthesizer --input <fp_report.json>[,<fp_report2.json>...] --outdir <dir> [--hypotheses <report.json>] [--trace <trace.jsonl>]" >&2
    exit 1
  fi
  INPUT_ARGS=()
  IFS=',' read -ra FP_PATHS <<< "$INPUT"
  for path in "${FP_PATHS[@]}"; do
    if [[ -n "$path" ]]; then
      INPUT_ARGS+=(--input_fp "$path")
    fi
  done
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_EXEC" -m arc_agi_agent.cli --agent "$AGENT" "${INPUT_ARGS[@]}" --outdir "$OUTDIR" ${HYPOTHESES_REPORT:+--hypotheses "$HYPOTHESES_REPORT"} ${TRACE_PATH:+--trace "$TRACE_PATH"}
  exit 0
fi

if [[ "$AGENT" == "swarm" ]]; then
  if [[ -z "$GAME" || -z "$OUTDIR" ]]; then
    echo "Usage: $0 --agent swarm --game <id> --outdir <dir> [--seed <n>] [--max-steps <n>] [--probe-steps <n>] [--snapshot-every-steps <n>] [--fp-save-mode buffer|files] [--op-mode offline|online] [--debug]" >&2
    exit 1
  fi
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_EXEC" -m arc_agi_agent.run_swarm --game "$GAME" --seed "$SEED" --max-steps "$MAX_STEPS" --probe-steps "$PROBE_STEPS" --snapshot-every-steps "$SNAPSHOT_EVERY_STEPS" --fp-save-mode "$FP_SAVE_MODE" --op-mode "$OP_MODE" --outdir "$OUTDIR" $DEBUG
  exit 0
fi

if [[ "$AGENT" == "trajectory_summarizer" ]]; then
  if [[ -z "$OUTDIR" ]]; then
    echo "Usage: $0 --agent trajectory_summarizer --outdir <dir> [--planner-trace <...>] [--simple-trace <...>] [--full-trace <...>] [--fp-dir <dir>] [--action-schema <schema.json>] [--hypotheses <report.json>] [--mechanic <report.json>] [--goal <report.json>] [--memory <memory.json>]" >&2
    exit 1
  fi
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_EXEC" -m arc_agi_agent.cli --agent "$AGENT" --outdir "$OUTDIR" ${PLANNER_TRACE:+--planner-trace "$PLANNER_TRACE"} ${SIMPLE_TRACE:+--simple-trace "$SIMPLE_TRACE"} ${FULL_TRACE:+--full-trace "$FULL_TRACE"} ${FP_DIR:+--fp-dir "$FP_DIR"} ${ACTION_SCHEMA_PATH:+--action-schema "$ACTION_SCHEMA_PATH"} ${HYPOTHESES_REPORT:+--hypotheses "$HYPOTHESES_REPORT"} ${MECHANIC_REPORT:+--mechanic "$MECHANIC_REPORT"} ${GOAL_REPORT:+--goal "$GOAL_REPORT"} ${MEMORY_PATH:+--memory "$MEMORY_PATH"}
  exit 0
fi
