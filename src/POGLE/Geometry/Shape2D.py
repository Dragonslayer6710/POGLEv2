import copy
import ctypes

import numpy as np

from POGLE.Geometry.Data import *


class Shape:
    local_positions = []
    local_positions_attribute: Optional[VA] = None

    indices = []

    vertices: Optional[DataSet] = None
    instances: Optional[DataSet] = None

    _base_instance = None
    def __init__(self, addVertAttrs: Optional[Union[VA, List[VA]]] = None, addVertData: Optional[List] = None,
                 instanceAttributes: Optional[List[VA]] = None):

        if self.__class__._base_instance:
            raise RuntimeError("A base instance already exists for this class. Cannot create another.")
        self.__class__._base_instance = self

        if self.local_positions_attribute is None:
            raise ValueError("Shape Defined Without Providing Attribute For Position")

        vertexAttributes = [self.local_positions_attribute]
        vertexDataParts = [self.local_positions]

        noAdditionalVertexAttributes = addVertAttrs is None
        noAdditionalVertexDataParts = addVertData is None
        if ((noAdditionalVertexAttributes and not noAdditionalVertexDataParts) or
                (not noAdditionalVertexAttributes and noAdditionalVertexDataParts)):
            raise ValueError("Additional Vertex Attributes or Data were provided without the other")
        if not noAdditionalVertexAttributes:
            if isinstance(addVertAttrs, VA):
                vertexAttributes += addVertAttrs
            if not isinstance(addVertData, List):
                vertexDataParts += addVertData


        if instanceAttributes:
            if isinstance(instanceAttributes, VA):
                instanceAttributes = [instanceAttributes]

        self.vertices: DataSet = self._gen_dataset(vertexAttributes, vertexDataParts)

        self._instanceAttributes = instanceAttributes
        self._instanceDataParts: Optional[List] = None

    def _create(self, instanceDataParts: Optional[List] = None,
                preInstAttribs: Optional[List[VA]] = None,
                postInstAttribs: Optional[List[VA]] = None):
        isPre = preInstAttribs is not None
        isPost = postInstAttribs is not None
        isCurrent = self._instanceAttributes is not None
        if not isCurrent:
            self._instanceAttributes = []
        # Handle pre-instance attributes (preInstAttribs)
        if isPre:
            if not isinstance(preInstAttribs, list):
                preInstAttribs = [preInstAttribs]
            self._instanceAttributes = preInstAttribs + self._instanceAttributes
        # Handle post-instance attributes (postInstAttribs)
        if isPost:
            if not isinstance(postInstAttribs, list):
                postInstAttribs = [postInstAttribs]
            self._instanceAttributes = self._instanceAttributes + postInstAttribs
        isCurrent = self._instanceAttributes is not None
        # Validate instance attributes and data parts
        if isCurrent:
            if (not isCurrent) != (instanceDataParts is None):
                raise ValueError("Instance Attributes or Data were provided without the other")
            if not isinstance(instanceDataParts, list):
                instanceDataParts = [instanceDataParts]
            # Generate dataset for instances with the updated attributes
            self.instances: DataSet = self._gen_dataset(self._instanceAttributes, instanceDataParts)

    def _gen_dataset(self, attributes, dataParts):
        layout: DVL = DVL(attributes)
        structs: List[ctypes.Structure] = [
            layout.struct([*vertex_data]) for vertex_data in zip(*dataParts)
        ]
        return DataSet(
            layout,
            structs
        )

    @classmethod
    def _create_instance(cls, instance_data_parts: Optional[List] = None,
                         preInstAttribs: Optional[List[VA]] = None,
                         postInstAttribs: Optional[List[VA]] = None) -> object:
        if cls._base_instance is None:
            cls()

        # Deep copy the base instance to create a new instance
        instance = copy.deepcopy(cls._base_instance)

        # Use the updated _create method to set up the instance
        instance._create(
            instanceDataParts=instance_data_parts,
            preInstAttribs=preInstAttribs,
            postInstAttribs=postInstAttribs
        )

        return instance

class Shape2D(Shape):
    local_positions_attribute = aLocalPosXY

class _Quad(Shape2D):
    local_positions = [
        glm.vec2(-1.0, -1.0) / 2,
        glm.vec2(-1.0, 1.0) / 2,
        glm.vec2(1.0, 1.0) / 2,
        glm.vec2(1.0, -1.0) / 2
    ]
    indices = [
        0, 1, 2,
        2, 3, 0
    ]

    def __init__(self, addVertAttrs: Optional[Union[VA, List[VA]]] = None, addVertData: Optional[List] = None):
        super().__init__(addVertAttrs, addVertData)

class _TexQuad(_Quad):
    tex_coords = [
        glm.vec2(0.0, 0.0),
        glm.vec2(0.0, 1.0),
        glm.vec2(1.0, 1.0),
        glm.vec2(1.0, 0.0)
    ]

    def __init__(self):
        super().__init__([aTexUV], [self.tex_coords])

class _Pent(Shape2D):
    local_positions = [
        glm.vec2(0.0, 1.0) / 2,
        glm.vec2(0.9511, 0.3090) / 2,
        glm.vec2(0.5878, -0.8090) / 2,
        glm.vec2(-0.5878, -0.8090) / 2,
        glm.vec2(-0.9511, 0.3090) / 2
    ]
    indices = [
        0, 1, 2,
        0, 2, 4,
        2, 3, 4
    ]

class _Crosshair(Shape2D):
    local_positions = [
        glm.vec2(0.0, 1.0),
        glm.vec2(0.0, -1.0),
        glm.vec2(-1.0, 0.0),
        glm.vec2(1.0, 0.0)
    ]
    indices = [
        0, 1,
        2, 3
    ]
    aScale = VA.Float().Vec2(1, attrName="aScale")
    def __init__(self, addVertAttrs: Optional[Union[VA, List[VA]]] = None, addVertData: Optional[List] = None,
                 instanceAttributes: Optional[List[VA]] = None):
        super().__init__(addVertAttrs, addVertData, [self.aScale, aInstColorRGB, aInstAlpha] + instanceAttributes)

def Pent(instanceDataParts: Optional[List] = None) -> _Pent:
    return _Pent._create_instance(instanceDataParts)

def Crosshair(scale: glm.vec2, color: glm.vec3, alpha: float) -> _Crosshair:
    return _Crosshair._create_instance([[scale], [color], [alpha]])


def Quad(sideIDs: List[int], colors: List[glm.vec3], alphas: List[float]) -> _Quad:
    idIsInt = isinstance(sideIDs, int)
    colIsVec = isinstance(colors, glm.vec2)
    alIsFloat = isinstance(alphas, float)
    if idIsInt == colIsVec == alIsFloat and (idIsInt or colIsVec or alIsFloat):
        sideIDs = [idIsInt]
        colors = [colors]
        alphas = [alphas]
    else:
        if not idIsInt:
            arr = sideIDs
        elif not colIsVec:
            arr = colors
        else:
            arr = alphas
        if idIsInt:
            sideIDs = [sideIDs for _ in arr]
        if colIsVec:
            colors = [colors for _ in arr]
        if alIsFloat:
            alphas = [alphas for _ in arr]

    return _Quad._create_instance(
        [
            sideIDs,
            colors,
            alphas
        ], [
            aSideID,
            aInstColorRGB,
            aInstAlpha
        ]
    )


def TexQuad(sideIDs: List[int], texturePositions: List[glm.vec2], textureSizes: List[glm.vec2]) -> _TexQuad:
    idIsInt = isinstance(sideIDs, int)
    posIsVec = isinstance(texturePositions, glm.vec2)
    sizeIsVec = isinstance(textureSizes, glm.vec2)
    if idIsInt == posIsVec == sizeIsVec and (idIsInt or posIsVec or sizeIsVec):
        sideIDs = [idIsInt]
        texturePositions = [texturePositions]
        textureSizes = [textureSizes]
    else:
        if not idIsInt:
            arr = sideIDs
        elif not posIsVec:
            arr = texturePositions
        else:
            arr = textureSizes
        if idIsInt:
            sideIDs = [sideIDs for _ in arr]
        if posIsVec:
            texturePositions = [texturePositions for _ in arr]
        if sizeIsVec:
            textureSizes = [textureSizes for _ in arr]

    return _TexQuad._create_instance(
        [
            sideIDs,
            texturePositions,
            textureSizes
        ], [
            aSideID,
            aTexPos,
            aTexSize
        ]
    )


class QuadCube:
    face_matrices = [
        NewModelMatrix(glm.vec3(-0.5, 0.0, 0.0), glm.vec3(0, -90, 0)),  # West
        NewModelMatrix(glm.vec3(0.0, 0.0, 0.5)),  # South
        NewModelMatrix(glm.vec3(0.5, 0.0, 0.0), glm.vec3(0, 90, 0)),  # East
        NewModelMatrix(glm.vec3(0.0, 0.0, -0.5), glm.vec3(0, 180, 0)),  # North
        NewModelMatrix(glm.vec3(0.0, 0.5, 0.0), glm.vec3(-90, 0, 0)),  # Top
        NewModelMatrix(glm.vec3(0.0, -0.5, 0.0), glm.vec3(90, 0, 0)),  # Bottom
    ]
    quads: Optional[List[_Quad]] = None

class ColQuadCube:
    def __init__(self, ):
        self.quads = Quad(list(range(6)), glm.vec3(), 1.0)

class TexQuadCube:
    def __init__(self):
        self.quads = TexQuad(list(range(6)), glm.vec2(), glm.vec2())

