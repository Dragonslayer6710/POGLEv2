from POGLE.Geometry.Shape2D import *
from POGLE.Physics.Collisions import Ray

class Shape3D(Shape):
    local_positions_attribute = aLocalPosXYZ

class _Line(Shape3D):
    positions = [glm.vec3(), glm.vec3()]
    indices = [0, 1]
    _use_base = False
    aPointPos = VA.Float().Vec3(attrName="aPointPos")
    aPointCol = VA.Float().Vec3(attrName="aPointCol")
    aPointAlpha = VA.Float().Vec3(attrName="aPointAlpha")
    def __init__(self):
        super().__init__(instanceAttributes=[self.aPointPos, aInstColorRGB, aInstAlpha])

def Line(ray: Ray, colors: List[glm.vec3], alphas: List[float]) -> _Line:
    if not isinstance(colors, List):
        colors = [colors, colors]
    if not isinstance(alphas, List):
        alphas = [alphas, alphas]
    return _Line._create_instance([[ray.start, ray.end], colors, alphas])

class _Cube(Shape3D):
    local_positions = [
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
        # Front face (z = 0.5)
        0, 1, 2, 2, 3, 0,

        # Back face (z = -0.5)
        4, 5, 6, 6, 7, 4,

        # Top face (y = 0.5)
        0, 1, 4, 4, 5, 0,

        # Bottom face (y = -0.5)
        2, 3, 6, 6, 7, 2,

        # Right face (x = 0.5)
        1, 2, 7, 7, 4, 1,

        # Left face (x = -0.5)
        0, 5, 6, 6, 3, 0
    ]

def Cube(modelMatrices: List[glm.mat4], color: List[glm.vec3], alpha: float = 1.0, sameColour = True, sameAlpha = True) -> _Cube:
    if isinstance(modelMatrices, glm.mat4):
        modelMatrices = [modelMatrices]
    if isinstance(color, glm.vec3):
        if sameColour:
            color = [color for i in modelMatrices]
        else:
            raise ValueError("Same Colour is set to false but only one colour was provided")
    if isinstance(alpha, float):
        if sameAlpha:
            alpha = [alpha for i in modelMatrices]
        else:
            raise ValueError("Same Alpha is set to false but only one alpha was provided")
    return _Cube._create_instance(modelMatrices, [aModel, aInstColorRGB, aInstAlpha])

class _WireframeCube(_Cube):
    indices = [
        # Front face edges
        0, 1,  # Top edge
        1, 2,  # Right edge
        2, 3,  # Bottom edge
        3, 0,  # Left edge

        # Back face edges
        4, 5,  # Top edge
        5, 6,  # Left edge
        6, 7,  # Bottom edge
        7, 4,  # Right edge

        # Connecting edges
        0, 5,  # Front top left to back top left
        1, 4,  # Front top right to back top right
        2, 7,  # Front bottom right to back bottom right
        3, 6  # Front bottom left to back bottom left
    ]

def WireframeCube(modelMatrices: List[glm.mat4], color: List[glm.vec3], alpha: float = 1.0, sameColour = True, sameAlpha = True) -> _WireframeCube:
    if isinstance(modelMatrices, glm.mat4):
        modelMatrices = [modelMatrices]
    if isinstance(color, glm.vec3):
        if sameColour:
            color = [color for i in modelMatrices]
        else:
            raise ValueError("Same Colour is set to false but only one colour was provided")
    if isinstance(alpha, float):
        if sameAlpha:
            alpha = [alpha for i in modelMatrices]
        else:
            raise ValueError("Same Alpha is set to false but only one alpha was provided")
    return _WireframeCube._create_instance(modelMatrices, [aModel, aInstColorRGB, aInstAlpha])

