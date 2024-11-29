from __future__ import annotations
import numpy as np
from POGLE.Geometry.Texture import UniformTextureAtlas

from POGLE.Core.Core import Optional
from POGLE.Core.Core import Renum
from POGLE.Core.Core import glm

texture_atlas: Optional[UniformTextureAtlas] = None


def init_texture_atlas():
    global texture_atlas
    if texture_atlas:
        return
    texture_atlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))


class FaceTexID(Renum):
    Null = -1
    GrassTop = 0
    Stone = 1
    Dirt = 2
    GrassSide = 3


class FaceTexSizeID(Renum):
    Full = 0


face_tex_id_cache: np.array = np.array([member for member in FaceTexID])


