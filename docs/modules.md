# API Reference

## Tensor Saving

::: torchspy.saver.TensorSaver
    options:
        show_root_heading: true
        members:
            - __init__
            - register_modules
            - save
            - get_module_path
            - reset_counts

::: torchspy.context.DebugContext
    options:
        show_root_heading: true
        members:
            - __init__
            - __enter__
            - __exit__

::: torchspy.saver.spy_save
    options:
        show_root_heading: true

::: torchspy.context.get_debug_context
    options:
        show_root_heading: true

## Call Tracing

::: torchspy.tracer.CallTracer
    options:
        show_root_heading: true
        members:
            - __init__
            - register_hooks
            - save_trace
            - get_trace
            - reset_trace
            - remove_hooks

## Base Class

::: torchspy._base.BaseDebugger
    options:
        show_root_heading: true
        members:
            - __init__
            - get_module_path

::: torchspy.trace_graph
    options:
        show_root_heading: true
        members:
            - parse_trace
            - build_tree
            - render_graph
            - save_tree_json
