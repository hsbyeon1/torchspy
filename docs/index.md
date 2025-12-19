# TorchSpy

A lightweight debug tool for saving and inspecting intermediate tensors during PyTorch model forward passes.

## Features

- **Non-intrusive debugging**: Add `spy_save()` calls to your model without changing the forward logic
- **Call tracing**: Automatically record module execution order using forward hooks
- **Context-aware saving**: Use `DebugContext` to add prefixes and organize saved tensors
- **Module path tracking**: Automatically tracks module hierarchy paths for meaningful tensor names
- **Thread-safe**: Uses context variables for safe concurrent execution
- **Flexible registration**: Register modules by class type or name pattern

## Installation

```bash
pip install torchspy
```

## Quick Start

### Saving Tensors

```python
from torchspy.saver import TensorSaver
from torchspy.context import DebugContext
from torchspy.saver import spy_save

# Setup
saver = TensorSaver("./debug_tensors")
saver.register_modules(model, target_classes=(AttentionLayer,))

# In your module's forward():
spy_save("q", q, self)

# Run with context
with DebugContext(saver, prefix="step0"):
    output = model(inputs)
```

### Tracing Module Calls

```python
from torchspy import CallTracer

# Setup tracer
tracer = CallTracer("./debug_traces")
tracer.register_hooks(model, target_classes=(nn.Linear,))

# Run forward pass - hooks record call order automatically
output = model(inputs)

# Save trace to file
tracer.save_trace("call_order.txt")
```

See [API Reference](modules.md) for detailed documentation.
