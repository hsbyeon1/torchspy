# Trace Graph

Utilities to convert a module call trace (text file) into a top-down graph.

Usage (CLI):

```bash
uv run python scripts/trace_to_graph.py --trace-file traces/forward.txt --out out.png
```

Options:
- `--show-repetition` (default: True): show repetition counts for nodes
- `--prune-leaves` (default: True): prune terminal leaf nodes
- `--squash-name` (default: False): collapse nodes with the same name
- `--save-json`: save parsed tree as JSON

The tool uses Graphviz (dot) for layout and outputs PNG images.
