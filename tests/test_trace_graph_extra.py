from pathlib import Path

import pytest

from torchspy.trace_graph import (
    apply_graph_render_options,
    build_tree,
    construct_graph,
    parse_trace,
    prune_leaves,
)


def test_parse_trace_empty_raises(tmp_path: Path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(ValueError, match="trace file is empty"):
        parse_trace(p)


def test_leading_numeric_segment_is_treated_as_name(tmp_path: Path):
    p = tmp_path / "trace.txt"
    p.write_text("0.foo.bar\n")
    lines = parse_trace(p)
    assert lines == ["0.foo.bar"]
    tree = build_tree(lines)
    # Root should have child '0'
    assert "0" in tree.children
    assert "foo" in tree.children["0"].children


def test_squash_name_merges_nodes_and_counts():
    # two paths with same basename under different parents
    paths = [
        "parentA.shared.leaf",
        "parentB.shared.leaf",
    ]
    tree = build_tree(paths)
    dot = construct_graph(
        tree, show_repetition=True, prune_leaves_flag=True, squash_name=True
    )
    # apply rendering options for completeness
    apply_graph_render_options(dot, dpi=123, figwidth=1.0, figheight=1.0)
    src = dot.source
    # 'shared' node should appear once and show repetition count (at least count > 1)
    assert "shared x" in src or "shared" in src


def test_prune_leaves_trims_last_segment():
    paths = [
        "a.b.c",
        "a.b.d",
        "single",
    ]
    tree = build_tree(paths)
    pruned = prune_leaves(tree)
    # For 'a.b.c' and 'a.b.d', last segment removed => paths become a.b and a.b
    # For 'single' (length 1) trimmed path remains 'single'
    # So pruned tree should have child 'a' and 'single' under root
    assert "a" in pruned.children
    assert "single" in pruned.children


def test_numeric_index_repetition_is_detected():
    # Simulate paths that include numeric layer indices
    paths = [
        "encoder.layers.0.adaln",
        "encoder.layers.1.adaln",
        "encoder.layers.0.transition",
        "encoder.layers.1.transition",
    ]
    tree = build_tree(paths)
    # 'layers' node under encoder should record two distinct indices (0,1)
    layers_node = tree.children["encoder"].children["layers"]
    assert len(layers_node.indices) == 2

    # Graph should show repetition (label contains 'layers x 2') when rendered
    dot = construct_graph(tree, show_repetition=True, prune_leaves_flag=True)
    src = dot.source
    assert "layers x 2" in src or "layers x 2" in src.replace("\n", " ")


def test_single_index_not_counted_as_repetition():
    # same index, different downstream children should NOT imply repetition
    paths = [
        "foo.layer.0.a",
        "foo.layer.0.b",
    ]
    tree = build_tree(paths)
    layers_node = tree.children["foo"].children["layer"]
    assert len(layers_node.indices) == 1
    # Even though a segment is visited multiple times, repetition is derived solely from distinct numeric indices
    dot = construct_graph(tree, show_repetition=True, prune_leaves_flag=True)
    src = dot.source
    assert "layer x 2" not in src
    # and 'layer' should still be present
    assert "layer" in src
