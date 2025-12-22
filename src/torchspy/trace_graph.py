"""Utilities to build and render a module call trace as a hierarchical graph.

This module parses a trace (text file with one module path per line),
builds a hierarchical tree, supports squashing numeric-indexed layers
(e.g., `layers.0`, `layers.1` -> `layers x 2`), optional pruning of
leaf nodes, and optional name-based squashing to merge nodes with the
same basename (or class name when a class map is supplied).

The graph is rendered using Graphviz (via the Python `graphviz` package)
and saved as a PNG file.

The implementation is intentionally small and raises explicit errors for
malformed inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from graphviz import Digraph


@dataclass
class Node:
    name: str
    children: Dict[str, "Node"] = field(default_factory=dict)
    count: int = 1  # repetition count for squashed numeric indices
    first_seen: int = 0  # index in trace for stable ordering
    occurrences: int = 0  # how many times node is visited in trace
    parents: List["Node"] = field(default_factory=list)

    def __hash__(self) -> int:  # provide hash based on object identity
        return id(self)


def parse_trace(path: Path) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Trace file not found: {}".format(path))
    lines: List[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    if not lines:
        # Empty trace is considered invalid input for downstream processing
        raise ValueError("trace file is empty")
    return lines


def _is_int_segment(seg: str) -> bool:
    return seg.isdigit()


def build_tree(paths: List[str]) -> Node:
    """Build a tree from dot-delimited paths.

    Numeric segments (pure digits) are treated as indices and absorbed
    into the previous name (so "layers.0.adaln" becomes "layers" -> "adaln").
    The tree preserves first-seen order information.
    """
    root = Node("root", first_seen=0)
    for idx, p in enumerate(paths):
        segs = p.split(".")
        # normalize numeric segments by absorbing them into previous segment
        new_segs: List[str] = []
        for seg in segs:
            if _is_int_segment(seg):
                if not new_segs:
                    # leading numeric index is unexpected; treat as name
                    new_segs.append(seg)
                else:
                    # absorb into previous segment by incrementing a suffix marker
                    # we don't change the previous name; instead we track repetition
                    # at the tree insertion time
                    # store a special marker that will be ignored here
                    # simply skip the numeric segment
                    continue
            else:
                new_segs.append(seg)

        node = root
        node.occurrences += 1
        for seg in new_segs:
            if seg not in node.children:
                node.children[seg] = Node(seg, first_seen=idx)
                node.children[seg].parents.append(node)
            child = node.children[seg]
            child.occurrences += 1
            node = child
    return root


def squash_numeric_indices(tree: Node) -> Node:
    """Collapse repeated numeric-indexed instances by counting distinct
    occurrences of an immediate child name at the same path level.

    This implementation is conservative: it inspects child names only and
    records counts using occurrences; it does not rename nodes (the caller
    may choose to present label as "{name} x {count}" when rendering).
    """

    # nothing to modify structurally; counts are inferred from occurrences
    # but we'll also traverse children recursively to ensure their counts
    for child in tree.children.values():
        squash_numeric_indices(child)
    return tree


def prune_leaves(tree: Node) -> Node:
    """Prune the last segment from each recorded path and rebuild the tree.

    For each path that was recorded in the original tree, remove its final
    segment and construct a new tree from the trimmed paths. This implements
    the "prune last element" behavior described in the spec.
    """

    def _collect_paths(node: Node, prefix: List[str], out: List[List[str]]):
        if not node.children:
            out.append(prefix.copy())
            return
        for child in node.children.values():
            prefix.append(child.name)
            _collect_paths(child, prefix, out)
            prefix.pop()

    all_paths: List[List[str]] = []
    for child in tree.children.values():
        _collect_paths(child, [child.name], all_paths)

    # Trim last segment from each path (if path has length > 1)
    trimmed = [p[:-1] if len(p) > 1 else p for p in all_paths]

    # Rebuild tree from trimmed paths
    new_root = Node("root")
    for idx, p in enumerate(trimmed):
        node = new_root
        node.occurrences += 1
        for seg in p:
            if seg not in node.children:
                node.children[seg] = Node(seg, first_seen=idx)
                node.children[seg].parents.append(node)
            child = node.children[seg]
            child.occurrences += 1
            node = child

    return new_root


def _collect_nodes_for_graph(
    node: Node, squash_name: bool, seen: Dict[str, str]
) -> List[Tuple[str, Node]]:
    """Collect nodes in pre-order and return list of (path_str, node).

    `path_str` is a unique identifier constructed from the path from the
    root to this node (dot-separated). If `squash_name` is True the returned
    path_str will be the node name alone (intended to collapse nodes by name).
    """
    items: List[Tuple[str, Node]] = []

    def _dfs(n: Node, prefix: List[str]):
        if n.name != "root":
            prefix = prefix + [n.name]
            path_str = ".".join(prefix)
            key = n.name if squash_name else path_str
            items.append((key, n))
        for c in n.children.values():
            _dfs(c, prefix)

    _dfs(node, [])
    return items


def construct_graph(
    tree: Node,
    show_repetition: bool = True,
    prune_leaves_flag: bool = True,
    squash_name: bool = False,
) -> Digraph:
    """Construct a Graphviz Digraph for the tree and return it to the caller.

    This function builds the graph structure (nodes and edges) but does NOT
    set rendering-specific attributes like DPI or size; callers (for example
    a CLI) should set those attributes before calling `digraph.render(...)`.
    """
    if prune_leaves_flag:
        tree = prune_leaves(tree)

    dot = Digraph(format="png")
    dot.attr(rankdir="TB")

    # Assign stable ids based on first_seen ordering
    node_items = _collect_nodes_for_graph(tree, squash_name, {})

    # Map key -> list[Node]
    nodes_by_key: Dict[str, List[Node]] = {}
    for key, node in node_items:
        nodes_by_key.setdefault(key, []).append(node)

    # Sort keys by earliest first_seen among constituent nodes for stable ordering
    def key_first_seen(k: str) -> int:
        return min(n.first_seen for n in nodes_by_key[k])

    sorted_keys = sorted(nodes_by_key.keys(), key=key_first_seen)

    node_to_id: Dict[str, str] = {}
    counter = 0

    # Create nodes: one Graphviz node per key (merging when squash_name=True)
    for key in sorted_keys:
        nid = f"n{counter}"
        # Label should omit parents' names (always use basename)
        # When squash_name is True, aggregate occurrences across nodes with same name
        nodes = nodes_by_key[key]
        if squash_name:
            total_occ = sum(n.occurrences for n in nodes)
            label_base = nodes[0].name
            label = (
                f"{label_base} x {total_occ}"
                if (show_repetition and total_occ > 1)
                else label_base
            )
        else:
            # key is full path_str; find the corresponding single node
            node = nodes[0]
            label_base = node.name
            label = (
                f"{label_base} x {node.occurrences}"
                if (show_repetition and node.occurrences > 1)
                else label_base
            )
        dot.node(nid, label)
        node_to_id[key] = nid
        counter += 1

    # Create edges (preserve insertion order of children to respect call order)
    seen_edges = set()
    for key in sorted_keys:
        parent_id = node_to_id[key]
        for node in nodes_by_key[key]:
            for child in node.children.values():
                if squash_name:
                    child_key = child.name
                else:
                    child_key = f"{key}.{child.name}"
                if child_key in node_to_id:
                    child_id = node_to_id[child_key]
                    edge = (parent_id, child_id)
                    if edge not in seen_edges:
                        dot.edge(parent_id, child_id)
                        seen_edges.add(edge)

    return dot


def apply_graph_render_options(
    dot: Digraph,
    dpi: int = 150,
    figwidth: Optional[float] = None,
    figheight: Optional[float] = None,
) -> None:
    """Apply DPI and size attributes to an existing Graphviz `Digraph`.

    This keeps rendering concerns separate from graph construction so callers
    (for example CLIs) can decide when and where to write files.
    """
    dot.attr(dpi=str(dpi))
    if figwidth is not None or figheight is not None:
        w = "" if figwidth is None else str(figwidth)
        h = "" if figheight is None else str(figheight)
        # Use '!' to force Graphviz to respect the size exactly
        dot.attr(size=f"{w},{h}!", ratio="fill")


def render_graph(
    tree: Node,
    show_repetition: bool = True,
    prune_leaves_flag: bool = True,
    squash_name: bool = False,
    dpi: int = 150,
    figwidth: Optional[float] = None,
    figheight: Optional[float] = None,
) -> Digraph:
    """Backward-compatible wrapper that builds the graph and applies render options.

    Prefer using `construct_graph` + `apply_graph_render_options` for clearer
    separation of concerns; this wrapper remains for convenience and tests.
    """
    dot = construct_graph(
        tree,
        show_repetition=show_repetition,
        prune_leaves_flag=prune_leaves_flag,
        squash_name=squash_name,
    )
    apply_graph_render_options(dot, dpi=dpi, figwidth=figwidth, figheight=figheight)
    return dot


def save_tree_json(tree: Node, json_path: Path) -> None:
    def node_to_dict(n: Node) -> Dict:
        return {
            "name": n.name,
            "count": n.occurrences,
            "children": [node_to_dict(c) for c in n.children.values()],
        }

    with open(json_path, "w") as f:
        json.dump(node_to_dict(tree), f, indent=2)
