from jax_array_info import sharding_info, sharding_vis

from .tracing import TracingContext

__all__ = ["JaxTracingContext"]

class JaxTracingContext(TracingContext):

    def __init__(self, call_stack = None):
        if call_stack is None:
            call_stack = []
        super().__init__(call_stack=call_stack, module_filter=module_filter, callback=log_callback)

    def _create_nested_context(self, callable_name, args, kwargs):
        # Create a new tracing context for the nested call
        return JaxTracingContext(
            call_stack=self._call_stack + [callable_name],
        )

def module_filter(module_name):
    return module_name.startswith("transformer_engine.jax") and not module_name.startswith("transformer_engine.jax.debug")

def callable_name_to_str(callable_name):
    module_name = callable_name.module
    name = callable_name.name
    module_name = module_name.replace('transformer_engine.jax.', 'te.')
    return f"{module_name}.{name}"

def log_debug_info(value, call_stack, name):
    # This function should log the debug information
    # For now, we will just print it
    print(f"Debug Info: {name} in {call_stack} - {value}")
    if type(value).__name__ == 'DynamicJaxprTracer':
        full_name = "->".join([callable_name_to_str(x) for x in call_stack]) + f":{name}"
        sharding_info(value, name=full_name)
        sharding_vis(value)

def log_callback(func, args, kwargs, ctx: JaxTracingContext):
#     return jax.tree_util.tree_map(
#       lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x,
#       boxed_pytree,
#       is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
#   )
    for i, arg in enumerate(args):
        # Handle the case where the argument is a Jax tracer
        log_debug_info(arg, ctx._call_stack, f'args[{i}]')
    
    for key, value in kwargs.items():
        log_debug_info(value, ctx._call_stack, f'kwargs["{key}"]')

    result = func(*args, **kwargs)

    log_debug_info(result, ctx._call_stack, "result")
    return result
