from POGLE.Geometry.Shape2D import *


class WireframeCube(Shape):
    positions = [
        glm.vec3(-1.0, 1.0, 1.0),  # Front Top Left
        glm.vec3(1.0, 1.0, 1.0),  # Front Top Right
        glm.vec3(1.0, -1.0, 1.0),  # Front Bottom Right
        glm.vec3(-1.0, -1.0, 1.0),  # Front Bottom Left
        glm.vec3(1.0, 1.0, -1.0),  # Back Top Right
        glm.vec3(-1.0, 1.0, -1.0),  # Back Top Left
        glm.vec3(-1.0, -1.0, -1.0),  # Back Bottom Left
        glm.vec3(1.0, -1.0, -1.0),  # Back Bottom Right
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


class QuadCube(Quad):
    _instanceLayout = VertexLayout([
        FloatVA.Vec3(divisor=1),   # Colour
        FloatVA.Single(divisor=1), # Alpha
        FloatVA.Mat4()    # World Model
    ])

    face_matrices = [
        NewModelMatrix(glm.vec3(-1.0, 0.0, 0.0), glm.vec3(0, - 90, 0)),
        NewModelMatrix(glm.vec3(0.0, 0.0, -1.0)),
        NewModelMatrix(glm.vec3(1.0, 0.0, 0.0), glm.vec3(0, 90, 0)),
        NewModelMatrix(glm.vec3(0.0, 0.0, 1.0), glm.vec3(0, 180, 0)),
        NewModelMatrix(glm.vec3(0.0, 1.0, 0.0), glm.vec3(90, 0, 0)),
        NewModelMatrix(glm.vec3(0.0, -1.0, 0.0), glm.vec3(- 90, 0, 0)),
    ]

    class Instance:
        def __init__(self, outerModelMatrix: glm.mat4, sideCols: list[glm.vec3] = 6 * Color.WHITE,
                     sideColAlphas: list[float] = 6 * [1.0]):
            if type(outerModelMatrix) != glm.mat4:
                raise TypeError("QuadCube outerModelMatrix must be a glm.mat4")

            if type(sideCols) == glm.vec3:
                sideCols = 6 * [sideCols]
            elif len(sideCols) != 6:
                raise TypeError("QuadCube sideCols must be of length 6")

            if type(sideColAlphas) == float:
                sideColAlphas = 6 * [sideColAlphas]
            elif len(sideColAlphas) != 6:
                raise TypeError("QuadCube sideColAlphas must be of length 6")

            face_matrices = [outerModelMatrix * face for face in QuadCube.face_matrices]
            self.data = interleave_arrays(sideCols, sideColAlphas, face_matrices)

    def __init__(self, outerModelMatrix: glm.mat4, sideCols: list[glm.vec3] = 6 * Color.WHITE,
                 sideColAlphas: list[float] = 6 * [1.0]):
        isList = type(outerModelMatrix) == list
        isDataLayout = type(outerModelMatrix) == QuadCube.Instance
        if isList or isDataLayout:
            if isDataLayout:
                instanceData = outerModelMatrix.data
            elif type(outerModelMatrix[0]) == QuadCube.Instance:
                instanceData = []
                for instance in outerModelMatrix:
                    instanceData += instance.data
        else:
            instanceData = QuadCube.Instance(outerModelMatrix, sideCols, sideColAlphas).data
        super().__init__(self._instanceLayout, instanceData)



class Shapes:
    # 2D
    Quad = Quad()
    Pentagon = Pentagon()
    # 3D
    WireframeCube = WireframeCube()
