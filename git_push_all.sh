#!/usr/bin/env bash
#python3 src/utils/describe_current_state.py
git add .
#git add \
#  arc_lang_v25 arc_models docs main models scripts tests \
#  Makefile current_state.txt git_push_all.sh pyproject.toml \
#  rule_memory.json test_rule_model.py
git commit -m "update"
git push

#cp docs/* /mnt/c/Users/sorin/ARCv3/docs/
