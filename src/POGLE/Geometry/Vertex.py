import glm

from POGLE.Core.Core import *

import glm


def NMM(t: glm.vec3, r: glm.vec3 = glm.vec3(), s: glm.vec3 = glm.vec3(1)) -> glm.mat4:
    return NewModelMatrix(t, r, s)


def NewModelMatrix(translation: glm.vec3 = glm.vec3(),
                   rotation: glm.vec3 = glm.vec3(),
                   scale: glm.vec3 = glm.vec3(1.0)) -> glm.mat4:
    # Create the rotation matrix using Euler angles
    rotation_matrix = glm.mat4_cast(glm.quat(glm.vec3(glm.radians(rotation))))

    # Create the scale and translation matrices
    scale_matrix = glm.scale(glm.mat4(), scale)
    translation_matrix = glm.translate(glm.mat4(), translation)

    # Combine the matrices: translation * (rotation * scale)
    model_matrix = translation_matrix * rotation_matrix * scale_matrix

    return model_matrix


def interleave_arrays(*arrays):
    # Ensure all arrays have the same length
    if not all(len(arr) == len(arrays[0]) for arr in arrays):
        raise ValueError("All input arrays must have the same shape.")

    # Transpose the list of arrays to interleave them
    return [list(sub_lists) for sub_lists in zip(*arrays)]


_typeDict = {
    GL_BOOL: np.bool_,
    GL_UNSIGNED_BYTE: np.uint8,
    GL_BYTE: np.int8,
    GL_UNSIGNED_SHORT: np.uint16,
    GL_SHORT: np.int16,
    GL_UNSIGNED_INT: np.uint32,
    GL_INT: np.int32,
    GL_FLOAT: np.float32,
    GL_DOUBLE: np.double
}

_typeBytes = {k: np.dtype(v).itemsize for k, v in _typeDict.items()}


class _VertexAttribute:
    class MatData:
        def __init__(self, rows: int = 0, cols: int = 0):
            self.rows: int = rows
            self.cols: int = cols

    __create_key = object()

    _type = -1

    @classmethod
    def _new(cls, normalized: GLboolean, subAttribs, divisor: int = 0, matData: MatData = None):
        return _VertexAttribute(cls.__create_key, _typeBytes[cls._type] * subAttribs, cls._type, normalized, subAttribs,
                                divisor, matData)

    def __init__(self, create_key, bytes: GLint, type: GLenum, normalized: GLboolean, size: GLsizei, divisor: int,
                 matData: MatData = None):
        assert (create_key == _VertexAttribute.__create_key), \
            "_VertexAttribute objects must be created using _VertexAttribute._new"
        self.bytes: GLint = bytes
        self.dtype: GLenum = type
        self.normalized: GLboolean = normalized
        self.size: GLsizei = size
        self.divisor = divisor
        self.matData = matData

    def setPointer(self, id: GLuint, stride: GLsizei, offset: GLsizei):
        if self.matData:
            subSize = int(self.size / self.matData.rows)
            subBytes = int(self.bytes / self.matData.rows)
            for c in range(self.matData.cols):
                glEnableVertexAttribArray(id+c)
                glVertexAttribPointer(id + c, subSize, self.dtype, self.normalized, stride,
                                      ctypes.c_void_p(offset + subBytes * c))
                glVertexAttribDivisor(id + c, self.divisor)
                print(
                    f"\nPointer Set:\t{{id: {id + c} | size: {subSize} | bytes: {subBytes} | dtype: {self.dtype} | normalised: {self.normalized} | stride: {stride} | offset: {offset + subBytes * c} | divisor: {self.divisor}}}")
        else:
            glEnableVertexAttribArray(id)
            glVertexAttribPointer(id, self.size, self.dtype, self.normalized, stride, ctypes.c_void_p(offset))
            print(
                f"\nPointer Set:\t{{id: {id} | size: {self.size} | bytes: {self.bytes} | dtype: {self.dtype} | normalised: {self.normalized} | stride: {stride} | offset: {offset} | divisor: {self.divisor}}}")
            if self.divisor:
                glVertexAttribDivisor(id, self.divisor)

    @classmethod
    def Single(cls, divisor: int = 0, normalized: GLboolean = GL_FALSE):
        return cls._new(normalized, 1, divisor)

    @classmethod
    def _Vec(cls, normalized: GLboolean, vecSize: int, divisor: int = 0):
        return cls._new(normalized, vecSize, divisor)

    @classmethod
    def Vec2(cls, divisor: int = 0, normalized: GLboolean = GL_FALSE):
        return cls._Vec(normalized, 2, divisor)

    @classmethod
    def Vec3(cls, divisor: int = 0, normalized: GLboolean = GL_FALSE):
        return cls._Vec(normalized, 3, divisor)

    @classmethod
    def Vec4(cls, divisor: int = 0, normalized: GLboolean = GL_FALSE):
        return cls._Vec(normalized, 4, divisor)

    @classmethod
    def _Mat(cls, divisor: int = 0, matRows: int = 2, matCols: int = 2, normalized: GLboolean = GL_FALSE):
        return cls._new(normalized, matRows * matCols, divisor, _VertexAttribute.MatData(matRows, matCols))

    @classmethod
    def Mat4(cls, divisor=1, normalized: GLboolean = GL_FALSE ):
        return cls._Mat(divisor, 4, 4, normalized)


class BoolVA(_VertexAttribute):
    _type = GL_BOOL


class UByteVA(_VertexAttribute):
    _type = GL_UNSIGNED_BYTE


class ByteVA(_VertexAttribute):
    _type = GL_BYTE


class UShortVA(_VertexAttribute):
    _type = GL_UNSIGNED_SHORT


class ShortVA(_VertexAttribute):
    _type = GL_SHORT


class UintVA(_VertexAttribute):
    _type = GL_UNSIGNED_INT


class IntVA(_VertexAttribute):
    _type = GL_INT


class FloatVA(_VertexAttribute):
    _type = GL_FLOAT


class DoubleVA(_VertexAttribute):
    _type = GL_DOUBLE


class VertexLayout:
    def __init__(self, vertAttribs):
        if type(vertAttribs) != list:
            vertAttribs = [vertAttribs]
        self.vertAttribs: list[_VertexAttribute] = vertAttribs
        self.stride = 0
        self.nextID = 0
        for vertAttrib in self.vertAttribs:
            self.stride += vertAttrib.bytes
        self.count = len(vertAttribs)

    def setPointers(self, extraOffset: int = 0):
        offset = 0
        for vertAttrib in self.vertAttribs:
            vertAttrib.setPointer(self.nextID, self.stride, offset + extraOffset)
            offset += vertAttrib.bytes
            if not vertAttrib.matData:
                self.nextID += 1
            else:
                self.nextID += vertAttrib.matData.cols


defaultVertexLayout = VertexLayout([
    FloatVA.Vec3(),  # Position
    FloatVA.Vec3(),  # Colour
    FloatVA.Single()  # Alpha
])

defaultInstanceLayout = VertexLayout([
    FloatVA.Mat4(),  # Model Matrix
])


class Vertex:
    def __init__(self, vertexElements, layout: VertexLayout = defaultVertexLayout):
        self.layout: VertexLayout = layout
        self.data = []
        self.bytes = 0
        for i in range(len(vertexElements)):
            vertexElement = vertexElements[i]
            vertAttrib = layout.vertAttribs[i]

            self.bytes += vertAttrib.bytes
            dtype = _typeDict[vertAttrib.dtype]
            isMat = type(vertexElement) == glm.mat4
            vertexElement = np.array(vertexElement, dtype)
            if isMat:
                vertexElement = vertexElement.reshape(4,4).T
            self.data = np.concatenate((self.data, vertexElement.flatten()), dtype=np.float32)


class Vertices:
    def __init__(self, verticesData, layout: VertexLayout = defaultVertexLayout):
        self.layout = layout
        self.data = []
        self.bytes = 0
        if type(verticesData[0]) != list:
            verticesData = interleave_arrays(verticesData)
        for vertexData in verticesData:
            vertex = Vertex(vertexData, layout)
            self.bytes += vertex.bytes
            self.data = np.concatenate((self.data, vertex.data), dtype=np.float32)

    def setPointers(self, start: int = 0, extraOffset: int = 0):
        self.layout.nextID = start
        self.layout.setPointers(extraOffset)

    def nextID(self) -> int:
        return self.layout.nextID

class Instances(Vertices):

    def __init__(self, instancesData, layout: VertexLayout = defaultInstanceLayout):
        super().__init__(instancesData, layout)
        self.count = len(instancesData)

