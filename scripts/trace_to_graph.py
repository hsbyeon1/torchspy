"""CLI to convert a module call trace into a graph image.

Usage example:
    uv run python scripts/trace_to_graph.py --trace-file traces/forward.txt --out graph.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from torchspy.trace_graph import (
    apply_graph_render_options,
    build_tree,
    construct_graph,
    parse_trace,
    save_tree_json,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert trace text to graph image")
    p.add_argument(
        "--trace-file", type=Path, required=True, help="Path to trace text file"
    )
    p.add_argument("--out", type=Path, required=True, help="Output image path (png)")
    p.add_argument(
        "--show-repetition",
        action="store_true",
        default=True,
        help="Show repetition counts on nodes (default: True)",
    )
    p.add_argument(
        "--no-show-repetition",
        action="store_false",
        dest="show_repetition",
        help="Do not show repetition counts",
    )
    p.add_argument(
        "--prune-leaves",
        action="store_true",
        default=True,
        help="Prune terminal leaf nodes (default: True)",
    )
    p.add_argument(
        "--no-prune-leaves",
        action="store_false",
        dest="prune_leaves",
        help="Do not prune leaves",
    )
    p.add_argument(
        "--squash-name",
        action="store_true",
        default=False,
        help="Collapse nodes with the same name across the tree (default: False)",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--figwidth", type=float, default=None)
    p.add_argument("--figheight", type=float, default=None)
    p.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save the parsed tree as JSON",
    )
    p.add_argument("--verbose", action="store_true", default=False)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Configure logging based on verbosity
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    trace_file: Path = args.trace_file
    out: Path = args.out

    logger.info("Parsing trace: %s", trace_file)
    try:
        paths = parse_trace(trace_file)
    except (ValueError, FileNotFoundError) as e:
        logger.error(str(e))
        raise

    tree = build_tree(paths)
    if args.save_json:
        save_tree_json(tree, args.save_json)
        logger.info("Saved JSON to %s", args.save_json)

    dot = construct_graph(
        tree,
        show_repetition=args.show_repetition,
        prune_leaves_flag=args.prune_leaves,
        squash_name=args.squash_name,
    )

    # Apply rendering options in the CLI so the graph construction remains
    apply_graph_render_options(
        dot, dpi=args.dpi, figwidth=args.figwidth, figheight=args.figheight
    )

    render_path = Path(dot.render(filename=str(out), cleanup=True))
    logger.info("Wrote graph to: %s", render_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
