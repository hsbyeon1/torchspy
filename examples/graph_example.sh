#!/usr/bin/env bash

uv run python scripts/trace_to_graph.py \
    --trace-file examples/output/traces/forward.txt \
    --out examples/output/traces/forward_graph \
    --show-repetition \
    --prune-leaves \
    --save-json examples/output/traces/forward_graph.json
