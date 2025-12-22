import json
import shutil
from pathlib import Path

import pytest

from torchspy.trace_graph import build_tree, parse_trace, prune_leaves, save_tree_json

SAMPLE_TRACE = """
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.adaln.a_norm
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.adaln.s_norm
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.adaln.s_scale
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.adaln
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.reshape_post_key
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.proj_q.0
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.qkv_rearr
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.proj_q
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.proj_kv.0
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.qkv_rearr
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.proj_kv
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.q_norm
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.k_norm
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.proj_z.0
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.proj_z.1
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.proj_z.2
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.proj_z
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.proj_g.0
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.proj_g.1
initial_embedder.token_embedder.input_embedder.encoder.atom_encoder.diffusion_transformer.layers.0.pair_bias_attn.proj_g
"""


@pytest.fixture
def tmp_trace(tmp_path: Path) -> Path:
    p = tmp_path / "trace.txt"
    p.write_text(SAMPLE_TRACE.strip())
    return p


def test_parse_and_build(tmp_trace: Path):
    lines = parse_trace(tmp_trace)
    assert len(lines) > 0
    tree = build_tree(lines)
    # basic sanity checks
    assert "initial_embedder" in tree.children


def test_prune_leaves(tmp_trace: Path, tmp_path: Path):
    tree = build_tree(parse_trace(tmp_trace))
    pruned = prune_leaves(tree)
    # After pruning, proj_g and proj_q leafs should be removed
    _ = json.dumps({c: list(n.children.keys()) for c, n in tree.children.items()})
    assert pruned is not None


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz 'dot' not available")
def test_render_and_save_json(tmp_trace: Path, tmp_path: Path):
    import importlib
    import inspect

    import torchspy.trace_graph as tg

    # Reload to ensure the test uses current source (avoid stale bound symbols)
    importlib.reload(tg)
    # Ensure we are testing the local workspace module, not an installed package
    src_path = Path(__file__).resolve().parents[1] / "src"
    module_file = Path(inspect.getsourcefile(tg)).resolve()
    assert str(module_file).startswith(str(src_path)), (
        f"trace_graph module loaded from {module_file} (expected under src/)"
    )

    tree = build_tree(parse_trace(tmp_trace))
    out = tmp_path / "graph.png"
    json_out = tmp_path / "tree.json"
    save_tree_json(tree, json_out)
    assert json_out.exists()

    # Call render_graph directly from the module object to avoid any stale
    # name bindings in the test global namespace.
    import importlib as _il

    _tg = _il.import_module("torchspy.trace_graph")
    dot = _tg.render_graph(tree, show_repetition=True, prune_leaves_flag=True)
    render_path = Path(dot.render(filename=str(out), cleanup=True))
    assert render_path.exists()
    assert render_path.suffix == ".png"


def test_render_with_size_and_dpi(tmp_trace: Path, tmp_path: Path):
    import shutil

    if shutil.which("dot") is None:
        pytest.skip("Graphviz 'dot' not available")
    import importlib

    import torchspy.trace_graph as tg

    importlib.reload(tg)
    tree = build_tree(parse_trace(tmp_trace))
    out = tmp_path / "graph2.png"
    # Try non-default DPI and fig size
    import importlib as _il

    _tg = _il.import_module("torchspy.trace_graph")
    dot = _tg.render_graph(
        tree,
        show_repetition=True,
        prune_leaves_flag=True,
        dpi=200,
        figwidth=6.0,
        figheight=4.0,
    )
    render_path = Path(dot.render(filename=str(out), cleanup=True))
    assert render_path.exists()
    assert render_path.suffix == ".png"
