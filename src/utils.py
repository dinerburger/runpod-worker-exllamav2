import os
import logging
from functools import wraps
from time import time
def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        logging.info(f"{func.__name__} completed in {end - start:.2f} seconds")
        return result
    return wrapper
