#!/usr/bin/env python
"""Example script demonstrating torchspy usage."""

import logging
import shutil
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn

from torchspy.context import DebugContext
from torchspy.saver import TensorSaver, spy_save
from torchspy.tracer import CallTracer


# =============================================================================
# Define a simple model with spy_save calls
# =============================================================================
class Attention(nn.Module):
    """Simple attention module."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Save intermediate tensors for debugging
        spy_save("q", q, self)
        spy_save("k", k, self)
        spy_save("v", v, self)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        spy_save("attn", attn, self)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class MLP(nn.Module):
    """Simple MLP module."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        spy_save("hidden", x, self)
        x = self.act(x)
        return self.fc2(x)


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleTransformer(nn.Module):
    """Simple transformer model."""

    def __init__(self, dim: int = 64, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1))


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    pl.seed_everything(42)
    """Run the example."""
    output_dir = Path(__file__).parent / "output"

    # Clean up previous runs
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Create model and input
    model = SimpleTransformer(dim=64, num_heads=4, num_layers=2)
    x = torch.randn(2, 16, 64)  # (batch, seq_len, dim)

    logger.info("=" * 60)
    logger.info("TensorSaver Example")
    logger.info("=" * 60)

    # Create saver and register modules
    saver = TensorSaver(output_dir / "tensors")
    saver.register_modules(model, target_classes=(Attention, MLP))

    # Run forward pass with debug context
    with DebugContext(saver, prefix="step0"):
        _ = model(x)

    # List saved tensors
    saved_files = sorted((output_dir / "tensors").glob("*.pt"))
    logger.info("Saved %d tensors:", len(saved_files))
    for f in saved_files:
        tensor = torch.load(f, weights_only=True)
        logger.info("  %s: shape=%s", f.name, list(tensor.shape))

    logger.info("")
    logger.info("=" * 60)
    logger.info("CallTracer Example")
    logger.info("=" * 60)

    # Create tracer and register hooks
    tracer = CallTracer(output_dir / "traces")
    tracer.register_hooks(model, target_classes=(nn.Linear,))

    # Run forward pass - hooks automatically record call order
    _ = model(x)

    # Save and display trace
    tracer.save_trace("forward.txt")
    logger.info("Module call order:")
    for i, path in enumerate(tracer.get_trace(), 1):
        logger.info("  %d. %s", i, path)

    # Clean up hooks
    tracer.remove_hooks()

    logger.info("")
    logger.info("Output saved to: %s", output_dir.absolute())


if __name__ == "__main__":
    main()
