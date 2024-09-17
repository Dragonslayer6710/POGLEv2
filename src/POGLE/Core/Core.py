from POGLE.OGL.OpenGLContext import *
import os, sys, random, copy
from typing import Union, List, Optional, Tuple, TypeVar, Generic, Type, Dict, Any
from enum import Enum, auto, unique
from collections import deque, namedtuple
import ctypes
from ctypes import sizeof as c_sizeof
import struct

class RestrictedEnum(type(Enum)):
    def __call__(cls, enum_obj: Enum):
        raise RuntimeError(f"Calling {cls.__name__} is not allowed.")

REnum = RestrictedEnum

STRUCT_FORMAT_UNSIGNED_INT = "I"
STRUCT_FORMAT_SHORT = "h"
STRUCT_FORMAT_UNSIGNED_SHORT = "H"

import glm
import numpy as np


def clamp(n, min, max):
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n


EPSILON: float = 1e-8


class Color:
    BLACK = glm.vec3(0.0)
    RED = glm.vec3(1.0, 0.0, 0.0)
    GREEN = glm.vec3(0.0, 1.0, 0.0)
    BLUE = glm.vec3(0.0, 0.0, 1.0)
    MAGENTA = glm.vec3(1.0, 0.0, 1.0)
    YELLOW = glm.vec3(1.0, 1.0, 0.0)
    CYAN = glm.vec3(0.0, 1.0, 1.0)
    WHITE = glm.vec3(1.0)


def BIT(x):
    return 1 << x


def POGLE_BIND_EVENT_FN(fn): return lambda *args, **kwargs: fn(*args, **kwargs)


cwd = os.getcwd()


def split_array(l: np.ndarray, n: int) -> np.ndarray:
    k, m = divmod(len(l), n)
    return np.array([l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)], l.dtype)
