from POGLE.Geometry.Shape2D import *


class WireframeCube(Shape):
    positions = [
        glm.vec3(-0.5,  0.5,  0.5),  # 0: Front Top Left
        glm.vec3( 0.5,  0.5,  0.5),  # 1: Front Top Right
        glm.vec3( 0.5, -0.5,  0.5),  # 2: Front Bottom Right
        glm.vec3(-0.5, -0.5,  0.5),  # 3: Front Bottom Left
        glm.vec3( 0.5,  0.5, -0.5),  # 4: Back Top Right
        glm.vec3(-0.5,  0.5, -0.5),  # 5: Back Top Left
        glm.vec3(-0.5, -0.5, -0.5),  # 6: Back Bottom Left
        glm.vec3( 0.5, -0.5, -0.5),  # 7: Back Bottom Right
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

    def __init__(self, position: glm.vec3, color: glm.vec3, alpha: float):

        super().__init__(instanceElements=[[color], [alpha], [NMM(position)]],
                         instanceAttributes=[FloatVA.Vec3(1), FloatVA.Single(1), FloatVA.Mat4()])


class QuadCube(Quad):
    face_matrices = [
        NewModelMatrix(glm.vec3(-0.5, 0.0, 0.0), glm.vec3(0, -90, 0)),  # Left
        NewModelMatrix(glm.vec3(0.0, 0.0, 0.5)),  # Front
        NewModelMatrix(glm.vec3(0.5, 0.0, 0.0), glm.vec3(0, 90, 0)),  # Right
        NewModelMatrix(glm.vec3(0.0, 0.0, -0.5), glm.vec3(0, 180, 0)),  # Back
        NewModelMatrix(glm.vec3(0.0, 0.5, 0.0), glm.vec3(-90, 0, 0)),  # Top
        NewModelMatrix(glm.vec3(0.0, -0.5, 0.0), glm.vec3( 90, 0, 0)),  # Bottom
    ]

    def __init__(self, outerModelMatrices: list[glm.mat4], vertexElements: list = [], vertexAttributes: list = [], instanceElements: list = [], instanceAttributes: list = []):
        if type(outerModelMatrices) != list:
            outerModelMatrices = [outerModelMatrices]
        #[instanceElements[i].append(outerModelMatrices[i]) for i in range(len(instanceElements))]
        outerModelMatrices = [[outerModelMatrix * face for face in self.face_matrices for outerModelMatrix in outerModelMatrices]]
        super().__init__(vertexElements, vertexAttributes, instanceElements + outerModelMatrices, instanceAttributes + [FloatVA.Mat4()])


class ColQuadCube(QuadCube):
    class Instance:
        def __init__(self, outerModelMatrix: glm.mat4, sideCols: list[glm.vec3] = 6 * Color.WHITE,
                     sideColAlphas: list[float] = 6 * [1.0]):
            if type(outerModelMatrix) != glm.mat4:
                raise TypeError("ColQuadCube outerModelMatrix must be a glm.mat4")

            if type(sideCols) == glm.vec3:
                sideCols = 6 * [sideCols]
            elif len(sideCols) != 6:
                raise TypeError("ColQuadCube sideCols must be of length 6")

            if type(sideColAlphas) == float:
                sideColAlphas = 6 * [sideColAlphas]
            elif len(sideColAlphas) != 6:
                raise TypeError("ColQuadCube sideColAlphas must be of length 6")

            self.data = [sideCols, sideColAlphas, outerModelMatrix]

    _instanceAttributes = [FloatVA.Vec3(1), FloatVA.Single(1)]
    def __init__(self, outerModelMatrices: glm.mat4, sideCols: list[glm.vec3] = 6 * Color.WHITE,
                 sideColAlphas: list[float] = 6 * [1.0]):
        isList = type(outerModelMatrices) == list
        isInstance = type(outerModelMatrices) == ColQuadCube.Instance
        instanceElements = []
        if isList or isInstance:
            if isInstance: #
                instanceElements = outerModelMatrices.data[0:2]
                outerModelMatrices = outerModelMatrices.data[2]
            elif type(outerModelMatrices[0]) == ColQuadCube.Instance:
                instanceElements = [[], []]
                temp = []
                for instance in outerModelMatrices:
                    instanceElements[0] += instance.data[0]
                    instanceElements[1] += instance.data[1]
                    temp.append(instance.data[2])
                outerModelMatrices = temp
        else:
            instance = self.Instance(outerModelMatrices, sideCols, sideColAlphas)
            instanceElements = instance.data[0:2]
            outerModelMatrices = instance.data[2]
        super().__init__(outerModelMatrices, instanceElements = instanceElements, instanceAttributes=self._instanceAttributes)


class TexQuadCube(QuadCube):
    texture_coords = [
        glm.vec2(0.0, 0.0),
        glm.vec2(0.0, 1.0),
        glm.vec2(1.0, 1.0),
        glm.vec2(1.0, 0.0)
    ]

    class Instance:
        def __init__(self, outerModelMatrix: glm.mat4, texPos: list[glm.vec2], texSize: list[glm.vec2]):
            if type(outerModelMatrix) != glm.mat4:
                raise TypeError("TexQuadCube outerModelMatrix must be a glm.mat4")

            if type(texPos) == glm.vec2:
                texPos = 6 * [texPos]
            elif len(texPos) != 6:
                raise TypeError("TexQuadCube texPos must be of length 6")

            if type(texSize) == glm.vec2:
                texSize = 6 * [texSize]
            elif len(texSize) != 6:
                raise TypeError("TexQuadCube texSize must be of length 6")

            self.data = [texPos, texSize, outerModelMatrix]

    _vertexAttributes = [FloatVA.Vec2()]
    _instanceAttributes = [FloatVA.Vec2(1), FloatVA.Vec2(1)]
    def __init__(self, outerModelMatrices: list[glm.mat4], texPos: list[glm.vec2], texSize: list[glm.vec2]):
        isList = type(outerModelMatrices) == list
        isInstance = type(outerModelMatrices) == TexQuadCube.Instance
        instanceElements = []
        if isList or isInstance:
            if isInstance:  #
                instanceElements = outerModelMatrices.data[0:2]
                outerModelMatrices = outerModelMatrices.data[2]
            elif type(outerModelMatrices[0]) == TexQuadCube.Instance:
                instanceElements = [[], []]
                temp = []
                for instance in outerModelMatrices:
                    instanceElements[0] += instance.data[0]
                    instanceElements[1] += instance.data[1]
                    temp.append(instance.data[2])
                outerModelMatrices = temp
        else:
            instance = self.Instance(outerModelMatrices, texPos, texSize)
            instanceElements = instance.data[0:2]
            outerModelMatrices = instance.data[2]
        super().__init__(outerModelMatrices, [self.texture_coords], self._vertexAttributes, instanceElements, self._instanceAttributes)

class Cube(Shape):
    positions = WireframeCube.positions
    indices = [
        # Front Face
        0, 1, 2,
        2, 3, 0,
        # Back Face
        4, 5, 6,
        6, 7, 4,
        # Left Face
        5, 0, 3,
        3, 6, 5,
        # Right Face
        1, 4, 7,
        7, 2, 1,
        # Top Face
        5, 4, 1,
        1, 0, 5,
        # Bottom Face
        3, 2, 7,
        7, 6, 3
    ]

    def __init__(self, color: glm.vec3, alpha: float, modelMat: glm.vec4):
        super().__init__(instanceElements=[[color], [alpha], [modelMat]],
                         instanceAttributes=[FloatVA.Vec3(1), FloatVA.Single(1), FloatVA.Mat4()])

class Shapes:
    # 2D
    Quad = Quad()
    Pentagon = Pentagon()
