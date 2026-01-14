from functools import wraps

from nautic.context import Context
from nautic.config import taskx_cfg

# Cache the task import (initially None)
_cached_task = None

# cached import prefect
def __get_task():
    global _cached_task
    if _cached_task is None:
        from prefect import task
        _cached_task = task
    return _cached_task

def taskx(_func=None, **t_kwargs):
    t_kwargs.setdefault("cache_policy", None)

    def decorator(fn):
        fn._is_taskx = True  # 👈 marker for taskx detection
        @wraps(fn)
        def wrapper(ctx: Context, *args, **kwargs):
            fn(ctx, *args, **kwargs)
            return ctx
        if taskx_cfg.disable_nautic:
            return staticmethod(wrapper)
        else:
            __cached_task = __get_task()
            return staticmethod(__cached_task(**t_kwargs)(wrapper))

    if _func is None:
        return decorator
    else:
        return decorator(_func)

