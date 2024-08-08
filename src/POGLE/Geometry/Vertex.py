import glm

from POGLE.Core.Core import *

import glm

def NMM(t: glm.vec3, r: glm.vec3 = glm.vec3(), s: glm.vec3 = glm.vec3(1)):
    return NewModelMatrix(t,r,s)
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
    __create_key = object()

    _type = -1

    @classmethod
    def _new(cls, normalized: GLboolean, subAttribs, divisor: int = 0):
        return _VertexAttribute(cls.__create_key, _typeBytes[cls._type] * subAttribs, cls._type, normalized, subAttribs, divisor)

    def __init__(self, create_key, bytes: GLint, type: GLenum, normalized: GLboolean, size: GLsizei, divisor: int):
        assert (create_key == _VertexAttribute.__create_key), \
            "_VertexAttribute objects must be created using _VertexAttribute._new"
        self.bytes: GLint = bytes
        self.type: GLenum = type
        self.normalized: GLboolean = normalized
        self.size: GLsizei = size
        self.divisor = divisor

    def setPointer(self, id: GLuint, stride: GLsizei, offset: GLsizei):
        glEnableVertexAttribArray(id)
        glVertexAttribPointer(id, self.size, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset))
        if self.divisor:
            glVertexAttribDivisor(id, self.divisor)

    @classmethod
    def Single(cls, normalized: GLboolean = False, divisor: int = 0):
        return cls._new(normalized, 1, divisor)

    @classmethod
    def _Vec(cls, normalized: GLboolean, vecSize: int, divisor: int = 0):
        return cls._new(normalized, vecSize, divisor)

    @classmethod
    def Vec2(cls, normalized: GLboolean = False, divisor: int = 0):
        return cls._Vec(normalized, 2, divisor)

    @classmethod
    def Vec3(cls, normalized: GLboolean = False, divisor: int = 0):
        return cls._Vec(normalized, 3, divisor)

    @classmethod
    def Vec4(cls, normalized: GLboolean = False, divisor: int = 0):
        return cls._Vec(normalized, 4, divisor)

    @classmethod
    def Mat4(cls, normalized: GLboolean = False, divisor=1):
        return [cls.Vec4(normalized, divisor)] * 4


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
        self.nextID = 0
        for vertAttrib in self.vertAttribs:
            if type(vertAttrib) == list:
                for subAttrib in vertAttrib:
                    self.stride += subAttrib.bytes
            else:
                self.stride += vertAttrib.bytes

    def setPointers(self, start=0):
        offset = 0
        for vertAttrib in self.vertAttribs:
            if type(vertAttrib) == list:
                for subAttrib in vertAttrib:
                    subAttrib.setPointer(self.nextID, self.stride, offset)
                    self.nextID += 1
                    offset += subAttrib.bytes
            else:
                vertAttrib.setPointer(self.nextID, self.stride, offset)
                offset += vertAttrib.bytes
                self.nextID += 1


defaultVertexLayout = VertexLayout([
    FloatVA.Vec3(),  # Position
    FloatVA.Vec3(),  # Colour
    FloatVA.Single()  # Alpha
])

defaultInstanceLayout = VertexLayout([
    FloatVA.Mat4(),  # Model Matrix
])


class Vertex:
    def __init__(self, parts, layout: VertexLayout = defaultVertexLayout):
        self.layout: VertexLayout = layout
        self.data = []
        self.bytes = 0
        for i in range(len(parts)):
            part = parts[i]
            try:
                list(part)
            except:
                part = [part]
            vertAttrib = layout.vertAttribs[i]
            if type(vertAttrib) == list:
                for i in range(len(vertAttrib)):
                    subAttrib = vertAttrib[i]
                    self.bytes += subAttrib.bytes
                    dtype = _typeDict[subAttrib.type]
                    subPart = np.array(part[i], dtype)
                    self.data = np.concatenate((self.data, subPart), dtype=np.float32)
            else:
                self.bytes += vertAttrib.bytes
                dtype = _typeDict[vertAttrib.type]
                part = np.array(part, dtype)
                self.data = np.concatenate((self.data, part), dtype=np.float32)


class Vertices:
    def __init__(self, verticesData, layout: VertexLayout = defaultVertexLayout):
        if type(verticesData[0]) != list:
            verticesData = interleave_arrays(verticesData)
        self.layout = layout
        self.data = []
        self.bytes = 0
        for vertexData in verticesData:
            vertex = Vertex(vertexData, layout)
            self.bytes += vertex.bytes
            self.data = np.concatenate((self.data, vertex.data), dtype=np.float32)

    def setPointers(self, start: int = 0):
        self.layout.nextID = start
        self.layout.setPointers(start)

    def nextID(self) -> int:
        return self.layout.nextID


class Instances(Vertices):

    def __init__(self, instancesData, layout: VertexLayout = defaultInstanceLayout):
        super().__init__(instancesData, layout)
        self.count = len(instancesData)
