import time
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from loguru import logger

T = TypeVar("T")
P = ParamSpec("P")


class MetricsUtils:
    @staticmethod
    def execution_time(func: Callable[P, T]):
        @wraps(func)
        def execution_time_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            total_time = end_time - start_time
            logger.info(f"Function: {func.__name__} Took {total_time:.4f} seconds")
            return result

        return execution_time_wrapper
