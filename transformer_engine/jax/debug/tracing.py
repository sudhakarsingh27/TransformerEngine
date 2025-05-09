from dataclasses import dataclass
import inspect
import sys

_GLOBAL_TRACING_CONTEXTS = []

__all__ = ["TracingContext"]

@dataclass
class CallableName:
    module: str
    name: str

class FunctionWrapper:
    def __init__(self, func, name: CallableName):
        self._func = func
        self._name = name

    def __call__(self, *args, **kwargs):
        if len(_GLOBAL_TRACING_CONTEXTS) == 0:
            # No tracing context, call the function directly
            return self._func(*args, **kwargs)

        outer_ctx = _GLOBAL_TRACING_CONTEXTS[-1]
        with outer_ctx._create_nested_context(self._name, args, kwargs) as ctx:
            return ctx._callback(self._func, args, kwargs, ctx)

class TracingContext:

    def __init__(self, call_stack, module_filter, callback):
        self._call_stack = call_stack
        self._module_filter = module_filter
        self._callback = callback

    def __enter__(self):
        _GLOBAL_TRACING_CONTEXTS.append(self)
        _wrap_modules(self._module_filter)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        _GLOBAL_TRACING_CONTEXTS.pop()
        if exc_type is not None:
            # Handle exception if needed
            pass
        return False
    
    def _create_nested_context(self, callable_name: CallableName, args, kwargs):
        # Create a new tracing context for the nested call
        return TracingContext(
            call_stack=self._call_stack + [callable_name],
            module_filter=self._module_filter,
            callback=self._callback
        )
    
def _wrap_modules(module_filter):
    for module_name, module in sys.modules.items():
        if module_filter(module_name):
            # Wrap the module with the tracing functionality
            _wrap_module(module)

def _wrap_module(module):
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if callable(attr):
            # Wrap the callable with tracing functionality
            setattr(module, attr_name, _wrap_callable(attr, CallableName(module.__name__, attr_name)))

def _wrap_callable(callable, name):
    if isinstance(callable, FunctionWrapper):
        # Already wrapped
        return callable

    if inspect.isfunction(callable):
        # Only functions are supported for wrapping currently
        return FunctionWrapper(callable, name)
    return callable