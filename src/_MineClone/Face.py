from enum import Enum, auto
from typing import Optional, Union, Any, Dict, List, Mapping, Generic, TypeVar, Iterable, Tuple
import numpy as np

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

class FaceTex(Renum):
    Null = -1
    GrassTop = auto()
    Stone = auto()
    Dirt = auto()
    GrassSide = auto()

_face_tex_cache: ImDict[int, FaceTex] = ImDict(
    {member.value: member for member in FaceTex}
)
