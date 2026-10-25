#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHONPATH=src:other_repos/arc-agi:other_repos/ARCEngine python -m v10.cli live "$@"
