from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from POGLE.Geometry.Texture import UniformTextureAtlas

from POGLE.Core.Core import Optional
from POGLE.Core.Core import Renum
from POGLE.Core.Core import glm

_faceTextureAtlas: Optional[UniformTextureAtlas] = None


def initFaceTextureAtlas():
    global _faceTextureAtlas
    if _faceTextureAtlas:
        return
    _faceTextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))


_ftType = np.short


class FaceTex(Renum):
    Null = _ftType(-1)
    GrassTop = _ftType(0)
    Stone = _ftType(1)
    Dirt = _ftType(2)
    GrassSide = _ftType(3)


_face_tex_cache: np.array = np.array([member for member in FaceTex])
