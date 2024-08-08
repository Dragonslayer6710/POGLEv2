from POGLE.Geometry.Shape2D import *
class WireframeCube(Shape):
    positions = [
        glm.vec3( -1.0,  1.0,  1.0), # Front Top Left
        glm.vec3(  1.0,  1.0,  1.0), # Front Top Right
        glm.vec3(  1.0, -1.0,  1.0), # Front Bottom Right
        glm.vec3( -1.0, -1.0,  1.0), # Front Bottom Left
        glm.vec3(  1.0,  1.0, -1.0), # Back Top Right
        glm.vec3( -1.0,  1.0, -1.0), # Back Top Left
        glm.vec3( -1.0, -1.0, -1.0), # Back Bottom Left
        glm.vec3(  1.0, -1.0, -1.0), # Back Bottom Right
    ]
    indices = [
        # Front Quad
        0, 1, 1, 2, 2, 3, 3, 0,
        # Back Quad
        4, 5, 5, 6, 6, 7, 7, 4,
        # Left Side Lines
        5, 0, 3, 6,
        # Right Side Lines
        1, 4, 7, 2,
    ]
    def __init__(self):
        super().__init__(Color.BLACK, 1.0)

class QuadCubes(Instances):
    class _Face:
        Left    = NewModelMatrix(glm.vec3(-1.0, 0.0, 0.0), glm.vec3(0, - 90, 0))
        Front   = NewModelMatrix(glm.vec3(0.0, 0.0, -1.0))
        Right   = NewModelMatrix(glm.vec3(1.0, 0.0, 0.0), glm.vec3(0, 90, 0))
        Back    = NewModelMatrix(glm.vec3(0.0, 0.0, 1.0), glm.vec3(0, 180, 0))
        Top     = NewModelMatrix(glm.vec3(0.0, 1.0, 0.0), glm.vec3(90, 0, 0))
        Bottom  = NewModelMatrix(glm.vec3(0.0, -1.0, 0.0), glm.vec3(- 90, 0, 0))

    _Faces = glm.array([
        _Face.Left,
        _Face.Front,
        _Face.Right,
        _Face.Back,
        _Face.Top,
        _Face.Bottom
    ])

    def __init__(self, worldModels: list[glm.mat4]):
        super().__init__([worldModel * face for face in self._Faces for worldModel in worldModels])


class Shapes:
    # 2D
    Quad = Quad()
    Pentagon = Pentagon()
    # 3D
    WireframeCube = WireframeCube()
