import glm
import numpy as np

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


class _DataAttribute:
    class MatData:
        def __init__(self, rows: int = 0, cols: int = 0):
            self.rows: int = rows
            self.cols: int = cols

    __create_key = object()

    _type = -1

    @classmethod
    def _new(cls, normalized: GLboolean, subAttribs, divisor: int = 0, matData: MatData = None):
        return _DataAttribute(cls.__create_key, _typeBytes[cls._type] * subAttribs, cls._type, normalized, subAttribs,
                              divisor, matData)

    def __init__(self, create_key, bytes: GLint, type: GLenum, normalized: GLboolean, size: GLsizei, divisor: int,
                 matData: MatData = None):
        assert (create_key == _DataAttribute.__create_key), \
            "_DataAttribute objects must be created using _DataAttribute._new"
        self.bytes: GLint = bytes
        self.dtype: GLenum = type
        self.normalized: GLboolean = normalized
        self.size: GLsizei = size
        self.divisor = divisor
        self.matData = matData

    def set_vertex_attrib_pointer(self, id: GLuint, stride: GLsizei, offset: GLsizei):
        if self.matData:
            # For matrix data, set up each attribute separately
            subSize = int(self.size / self.matData.rows)
            subBytes = int(self.bytes / self.matData.rows)
            for c in range(self.matData.cols):
                glEnableVertexAttribArray(id + c)
                if self.dtype == GL_INT:
                    # Use integer pointer setup
                    glVertexAttribIPointer(id + c, subSize, self.dtype, stride, ctypes.c_void_p(offset + subBytes * c))
                else:
                    # Use floating-point pointer setup
                    glVertexAttribPointer(id + c, subSize, self.dtype, self.normalized, stride,
                                          ctypes.c_void_p(offset + subBytes * c))
                glVertexAttribDivisor(id + c, self.divisor)
                #print(
                #    f"\nPointer Set:\t{{id: {id + c} | size: {subSize} | bytes: {subBytes} | dtype: {self.dtype} | normalized: {self.normalized} | stride: {stride} | offset: {offset + subBytes * c} | divisor: {self.divisor}}}")
        else:
            glEnableVertexAttribArray(id)
            if self.dtype == GL_INT:
                # Use integer pointer setup
                glVertexAttribIPointer(id, self.size, self.dtype, stride, ctypes.c_void_p(offset))
            else:
                # Use floating-point pointer setup
                glVertexAttribPointer(id, self.size, self.dtype, self.normalized, stride, ctypes.c_void_p(offset))
            #print(
            #    f"\nPointer Set:\t{{id: {id} | size: {self.size} | bytes: {self.bytes} | dtype: {self.dtype} | normalized: {self.normalized} | stride: {stride} | offset: {offset} | divisor: {self.divisor}}}")
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
        return cls._new(normalized, matRows * matCols, divisor, _DataAttribute.MatData(matRows, matCols))

    @classmethod
    def Mat4(cls, divisor=1, normalized: GLboolean = GL_FALSE ):
        return cls._Mat(divisor, 4, 4, normalized)


class BoolDA(_DataAttribute):
    _type = GL_BOOL


class UByteDA(_DataAttribute):
    _type = GL_UNSIGNED_BYTE


class ByteDA(_DataAttribute):
    _type = GL_BYTE


class UShortDA(_DataAttribute):
    _type = GL_UNSIGNED_SHORT


class ShortDA(_DataAttribute):
    _type = GL_SHORT


class UIntDA(_DataAttribute):
    _type = GL_UNSIGNED_INT


class IntDA(_DataAttribute):
    _type = GL_INT


class FloatDA(_DataAttribute):
    _type = GL_FLOAT


class DoubleDA(_DataAttribute):
    _type = GL_DOUBLE


class DataLayout:
    def __init__(self, dataAttribs: list[_DataAttribute]):
        if type(dataAttribs) != list:
            dataAttribs: list[_DataAttribute] = [dataAttribs]
        self.dataAttribs: list[_DataAttribute] = dataAttribs
        self.stride = 0
        self.count = 0
        for vertAttrib in self.dataAttribs:
            self.stride += vertAttrib.bytes
            self.count += vertAttrib.size



class VertexLayout(DataLayout):
    def __init__(self, vertAttribs: list[_DataAttribute]):
        super().__init__(vertAttribs)
        self.nextID = 0

    def set_vertex_attrib_pointers(self, extraOffset: int = 0):
        offset = 0
        for vertAttrib in self.dataAttribs:
            vertAttrib.set_vertex_attrib_pointer(self.nextID, self.stride, offset + extraOffset)
            offset += vertAttrib.bytes
            if not vertAttrib.matData:
                self.nextID += 1
            else:
                self.nextID += vertAttrib.matData.cols

defaultVertexLayout = VertexLayout([
    FloatDA.Vec3(),  # Position
    FloatDA.Vec3(),  # Colour
    FloatDA.Single()  # Alpha
])

defaultInstanceLayout = VertexLayout([
    FloatDA.Mat4(),  # Model Matrix
])

class DataPoint:
    def __init__(self, dataElements, layout: DataLayout):
        self.layout: DataLayout = layout
        self.data: np.ndarray | None = None
        self.bytes: int | None = None
        self.setData(dataElements)

    def setData(self, dataElements):
        self.data = []
        self.bytes = 0
        for i in range(len(dataElements)):
            dataElement = dataElements[i]
            vertAttrib = self.layout.dataAttribs[i]

            self.bytes += vertAttrib.bytes
            dtype = _typeDict[vertAttrib.dtype]
            isMat = type(dataElement) == glm.mat4
            dataElement = np.array(dataElement, dtype)
            if isMat:
                dataElement = dataElement.reshape(4, 4).T
            self.data = np.concatenate((self.data, dataElement.flatten()), dtype=np.float32)


class DataPoints:
    layout: DataLayout
    def __init__(self, dataPointsData, layout: DataLayout, directData: bool = False):
        self.layout = layout
        self.data: np.ndarray | None = None
        self.bytes: int | None = None
        self.setData(dataPointsData, directData)

    def setData(self, dataPointsData, directData: bool = False):
        if directData:
            self.data = dataPointsData
            self.bytes = self.data.nbytes
        else:
            self.data = []
            self.bytes = 0
            if type(dataPointsData[0]) != list:
                dataPointsData = interleave_arrays(dataPointsData)
            for dataPointData in dataPointsData:
                dataPoint = DataPoint(dataPointData, self.layout)
                self.bytes += dataPoint.bytes
                self.data = np.concatenate((self.data, dataPoint.data), dtype=np.float32)


class Vertex(DataPoint):
    def __init__(self, vertexElements, layout: VertexLayout = defaultVertexLayout):
        super().__init__(vertexElements, layout)

class Vertices(DataPoints):
    layout: VertexLayout
    def __init__(self, verticesData, layout: VertexLayout = defaultVertexLayout, directData: bool = False):
        super().__init__(verticesData, layout, directData)

    def set_vertex_attrib_pointers(self, start: int = 0, extraOffset: int = 0):
        self.layout.nextID = start
        self.layout.set_vertex_attrib_pointers(extraOffset)

    def nextID(self) -> int:
        return self.layout.nextID

class Instances(Vertices):

    def __init__(self, instancesData, layout: VertexLayout = defaultInstanceLayout, directData: bool = False):
        super().__init__(instancesData, layout, directData)
        if directData:
            self.count = int(len(instancesData) / layout.count)
        else:
            self.count = len(instancesData)