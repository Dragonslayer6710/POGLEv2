from __future__ import annotations
from POGLE.Geometry.Texture import *
from POGLE.Core.Core import *

_faceTextureAtlas: Optional[UniformTextureAtlas] = None


def initFaceTextureAtlas():
    global _faceTextureAtlas
    if _faceTextureAtlas:
        return
    _faceTextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))


class FaceTex(Enum, metaclass=REnum):
    Null = -1
    GrassTop = auto()
    Stone = auto()
    Dirt = auto()
    GrassSide = auto()

_face_tex_cache: Dict[int, FaceTex] = {
    face_tex.value: face_tex for face_tex in FaceTex
}


def test_faces():
    tests = 1_000_000
    import time
    start = time.time()
    face_textures = [-1 for i in range(6)]
    for i in range(tests):
        new_face_textures = [_face_tex_cache[ft] for ft in face_textures]
    end = time.time()
    print("Cache lookup time:", end - start)
    start = time.time()
    for i in range(tests):
        new_face_textures = [FaceTex.Null for ft in face_textures]
    end = time.time()
    print("Direct access time:", end - start)
    quit()


if __name__ == "__main__":
    test_faces()
