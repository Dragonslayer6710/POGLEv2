from __future__ import annotations
from POGLE.Geometry.Texture import UniformTextureAtlas

from POGLE.Core.Core import Optional
from POGLE.Core.Core import auto, ImDict, Renum
from POGLE.Core.Core import glm

_faceTextureAtlas: Optional[UniformTextureAtlas] = None


def initFaceTextureAtlas():
    global _faceTextureAtlas
    if _faceTextureAtlas:
        return
    _faceTextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))


class FaceTex(Renum):
    Null = -1
    GrassTop = auto()
    Stone = auto()
    Dirt = auto()
    GrassSide = auto()

_face_tex_cache: ImDict[int, FaceTex] = ImDict(
    {member.value: member for member in FaceTex}
)
