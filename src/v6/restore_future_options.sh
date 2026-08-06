#!/usr/bin/env bash
set -euo pipefail

git checkout 8c4e3c295fb6a35e1931bf95a096f70c4d621fad -- src/v6/future_options.py
rm -f spec/v6/h01_add.py spec/v6/r3
python -m py_compile src/v6/future_options.py
