from functools import wraps

from nautic.context import Context
from nautic.config import flowx_cfg

# Cache the flow import (initially None)
_cached_flow = None

# cached import prefect, so that we load it only when necessary (e.g: prefect not needed in testing)
def __get_flow():
    global _cached_flow
    if _cached_flow is None:
        from prefect import flow
        _cached_flow = flow
    return _cached_flow

# a context-passing function that optionally becomes a Prefect flow
def flowx(_func=None, **f_kwargs):
    def decorator(fn):
        # optional marker used for testing/introspection
        fn._is_flowx = True

        # wraps function so that it returns the context and keeps the function's documentation intact
        @wraps(fn)
        def wrapper(ctx: Context, *args, **kwargs):
            fn(ctx, *args, **kwargs)
            return ctx

        if flowx_cfg.disable_nautic:
            return wrapper  # just call the function
        else:
            __flow = __get_flow()

            # applies the actual @flow wrapper
            return __flow(**f_kwargs)(wrapper)

    if _func is None:
        return decorator
    else:
        return decorator(_func)
