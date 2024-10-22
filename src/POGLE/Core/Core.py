from POGLE.OGL.OpenGLContext import *
import os, sys, random, copy
from typing import Optional, Union, Any, Dict, List, Mapping, Generic, TypeVar, Iterable, Tuple, Type, Set
from enum import Enum, auto, unique
from collections import deque, namedtuple
import ctypes
from ctypes import sizeof as c_sizeof
import struct
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

_K = TypeVar('_K')
_V = TypeVar('_V')
class _ImmutableDict(Mapping[_K, _V], Generic[_K, _V]):
    def __init__(self, data: Optional[Union[Dict[_K, _V], Iterable[Tuple[_K, _V]]]] = None):
        if data is None:
            # Initialize with an empty dictionary
            self._data = {}
        elif isinstance(data, dict):
            # Initialize with a dictionary
            self._data = dict(data)
        elif isinstance(data, Iterable):
            # Initialize with an iterable of (key, value) tuples
            self._data = dict(data)
        else:
            raise TypeError("Invalid type for 'data'. Expected dict or iterable of tuples.")

    def __getitem__(self, key: _K) -> _V:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

ImDict = _ImmutableDict

class _RestrictedEnum(type(Enum)):
    def __call__(self, *args, **kwargs):
        raise RuntimeError("Calling Enum with metaclass RestrictedEnum is forbidden!")

class Renum(Enum, metaclass=_RestrictedEnum):
    pass

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
