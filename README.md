# torchspy

[![Release](https://img.shields.io/github/v/release/hsbyeon1/torchspy)](https://img.shields.io/github/v/release/hsbyeon1/torchspy)
[![Build status](https://img.shields.io/github/actions/workflow/status/hsbyeon1/torchspy/main.yml?branch=main)](https://github.com/hsbyeon1/torchspy/actions/workflows/main.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/hsbyeon1/torchspy)](https://img.shields.io/github/license/hsbyeon1/torchspy)

A lightweight tool for saving intermediate tensors and tracing module execution during PyTorch forward passes.

## Installation

```bash
cd /path/to/clone
git clone https://github.com/hsbyeon1/torchspy.git
cd /path/to/your/project
uv add <--editable> /path/to/clone/torchspy
```

## Features

- **TensorSaver**: Save intermediate tensors with `spy_save()` calls
- **CallTracer**: Record module execution order via forward hooks (no code changes needed)
- **Trace Graph**: Generate a top-down graph visualization from a saved trace (`scripts/trace_to_graph.py`), with options for squashing repeated layers and pruning leaves.
- **Context-aware**: Organize saved tensors by prefix (batch/step)
- **Zero overhead**: `spy_save()` is a no-op outside debug context

## Usage

### Saving Tensors

```python
from torchspy import TensorSaver, DebugContext,
from torchspy.saver import spy_save

# In your module's forward():
class MyModule(nn.Module):
    def forward(self, x):
        q = self.proj_q(x)
        spy_save("q", q, self)  # Only saves when inside DebugContext
        return q

# Setup and run
saver = TensorSaver("./debug_tensors")
saver.register_modules(model, target_classes=(MyModule,))

with DebugContext(saver, prefix="step0"):
    output = model(inputs)
# Saves as: step0.{module_path}.q.call0.pt
```

### Tracing Module Calls

```python
from torchspy import CallTracer

tracer = CallTracer("./traces")
tracer.register_hooks(model, target_classes=(nn.Linear,))

output = model(inputs)  # Hooks record call order automatically

tracer.save_trace("forward.txt")
# Output:
# encoder.layer.0.linear1
# encoder.layer.0.linear2
# ...
```

## License

MIT
