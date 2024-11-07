from __future__ import annotations

from typing import List, Union

import numpy as np
from dataclasses import dataclass

from POGLE.Geometry.Data import VertexAttribute
from POGLE.Geometry.Shape import TexCube, TexQuad
from POGLE.Geometry.Texture import UniformTextureAtlas, TexDims

from POGLE.Core.Core import Optional
from POGLE.Core.Core import Renum
from POGLE.Core.Core import glm

_faceTextureAtlas: Optional[UniformTextureAtlas] = None


def initFaceTextureAtlas():
    global _faceTextureAtlas
    if _faceTextureAtlas:
        return
    _faceTextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))


class FaceTexID(Renum):
    Null = -1
    GrassTop = 0
    Stone = 1
    Dirt = 2
    GrassSide = 3


class FaceSizeID(Renum):
    Full = 0


_face_tex_id_cache: np.array = np.array([member for member in FaceTexID])


