import numpy as np

from Core import *


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

_typeBytes = {k: np.dtype(v).itemsize for k,v in _typeDict.items()}

class _VertexAttribute:
    __create_key = object()

    _type = -1

    @classmethod
    def _new(cls, normalized: GLboolean, subAttribs):
        return _VertexAttribute(cls.__create_key, _typeBytes[cls._type] * subAttribs, cls._type, normalized, subAttribs)

    def __init__(self, create_key, bytes: GLint, type: GLenum, normalized: GLboolean, subAttribs: GLsizei):
        assert (create_key == _VertexAttribute.__create_key), \
            "_VertexAttribute objects must be created using _VertexAttribute._new"
        self.bytes: GLint = bytes
        self.type: GLenum = type
        self.normalized: GLboolean = normalized
        self.elements: GLsizei = subAttribs

    def setPointer(self, id: GLuint, stride: GLsizei, offset: GLsizei):
        glVertexAttribPointer(id, 4, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        glEnableVertexAttribArray(id)

    @classmethod
    def Single(cls, normalized: GLboolean = False):
        return cls._new(normalized, 1)

    @classmethod
    def _Vec(cls, normalized: GLboolean, vecSize):
        return cls._new(normalized, vecSize)

    @classmethod
    def Vec2(cls, normalized: GLboolean = False):
        return cls._Vec(normalized, 2)

    @classmethod
    def Vec3(cls, normalized: GLboolean = False):
        return cls._Vec(normalized, 3)

    @classmethod
    def Vec4(cls, normalized: GLboolean = False):
        return cls._Vec(normalized, 4)


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
        self.vertAttribs: list[_VertexAttribute] = vertAttribs
        self.stride = 0
        for vertAttrib in self.vertAttribs: self.stride += vertAttrib.bytes

    def setPointers(self):
        offset = 0
        for id in range(len(self.vertAttribs)):
            vertAttrib = self.vertAttribs[id]
            vertAttrib.setPointer(id, self.stride, offset)
            offset += vertAttrib.bytes


class DefaultLayout(VertexLayout):
    def __init__(self):
        super().__init__(
            [
                FloatVA.Vec4(),  # Position
                FloatVA.Vec4()  # Colour
            ]
        )


class Vertex:
    def __init__(self, parts, layout: VertexLayout = DefaultLayout()):
        self.layout: VertexLayout = layout
        self.data = []
        self.bytes = 0
        for i in range(len(parts)):
            part = parts[i]
            if type(part) is not list:
                part = [part]
            self.bytes += layout.vertAttribs[i].bytes
            dtype = _typeDict[layout.vertAttribs[i].type]
            part = np.array(part, dtype)
            self.data = np.concatenate((self.data, part),dtype=np.float32)

class Vertices:
    def __init__(self, verticesData, layout: VertexLayout = DefaultLayout()):
        self.layout = layout
        self.data = []
        self.bytes = 0
        for vertexData in verticesData:
            vertex = Vertex(vertexData, layout)
            self.bytes += vertex.bytes
            self.data = np.concatenate((self.data, vertex.data), dtype=np.float32)

    def setPointers(self):
        self.layout.setPointers()