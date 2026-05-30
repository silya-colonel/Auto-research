#!/usr/bin/env bash
# Entrypoint for remote training launched via SSH from Mac.
# Usage: run_experiments.sh <task-name> [train_yolo.py args...]
#
# Accepts train_yolo.py positional flags plus these extras (consumed here):
#   --data-yaml PATH   (relative to ~/ar or absolute)
#   --model NAME       (default: yolo11n.pt)
#   --clearml-project NAME  (default: yolo-steel-defect)
#
# All other arguments (--imgsz, --epochs, --batch, --device, --workers,
#   --seed, --runs-dir, --extra ...) are forwarded to train_yolo.py.
#
# Example:
#   ssh server03 "~/ar/run_experiments.sh baseline_... --data-yaml data/steel-defect-mixed/data.yaml --model yolo11n.pt --imgsz 640 --epochs 100 --device 0 --seed 42"

set -euo pipefail

ROOT="${HOME}/ar"
VENV="${HOME}/train-venv/bin/python"

if [ $# -lt 1 ]; then
    echo "Usage: run_experiments.sh <task-name> [args...]" >&2
    exit 1
fi

TASK_NAME="$1"
shift

DATA_YAML=""
MODEL="yolo11n.pt"
CLEARML_PROJECT="yolo-steel-defect"
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --data-yaml)
            DATA_YAML="$2"; shift 2 ;;
        --data-yaml=*)
            DATA_YAML="${1#*=}"; shift
            if [ -z "${DATA_YAML}" ]; then
                echo "ERROR: --data-yaml= requires a non-empty value" >&2
                exit 1
            fi
            ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --model=*)
            MODEL="${1#*=}"; shift
            if [ -z "${MODEL}" ]; then
                echo "ERROR: --model= requires a non-empty value" >&2
                exit 1
            fi
            ;;
        --clearml-project)
            CLEARML_PROJECT="$2"; shift 2 ;;
        --clearml-project=*)
            CLEARML_PROJECT="${1#*=}"; shift
            if [ -z "${CLEARML_PROJECT}" ]; then
                echo "ERROR: --clearml-project= requires a non-empty value" >&2
                exit 1
            fi
            ;;
        --extra)
            shift
            EXTRA_ARGS+=("--extra")
            for extra_arg in "$@"; do
                EXTRA_ARGS+=("${extra_arg}")
                shift
            done
            break ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "$DATA_YAML" ]; then
    echo "ERROR: --data-yaml is required" >&2
    exit 1
fi

# Resolve data-yaml path
if [[ "$DATA_YAML" != /* ]]; then
    DATA_YAML="${ROOT}/${DATA_YAML}"
fi

echo "=== Starting training ==="
echo "task:    ${TASK_NAME}"
echo "data:    ${DATA_YAML}"
echo "model:   ${MODEL}"
echo "project: ${CLEARML_PROJECT}"
echo "extra:   ${EXTRA_ARGS[*]:-none}"
echo ""

cd "${ROOT}"

exec "${VENV}" train_yolo.py train \
    --task-name "${TASK_NAME}" \
    --data-yaml "${DATA_YAML}" \
    --model "${MODEL}" \
    --enable-clearml \
    --clearml-project "${CLEARML_PROJECT}" \
    "${EXTRA_ARGS[@]}"
