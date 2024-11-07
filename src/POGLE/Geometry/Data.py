from OpenGL.GL import *
import glfw
import numpy as np
import ctypes
from ctypes import c_float, c_double, c_uint, c_short, c_ushort
import glm
from dataclasses import dataclass
from typing import List, Collection, Union, Any, Tuple, Dict


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


class Buffer:
    _bound_buffers: dict = {}

    def __init__(self, target, usage, dtype: np.dtype = np.float32):
        self.target = target
        self.usage = usage
        self.ID = glGenBuffers(1)
        if self.ID == 0:
            raise RuntimeError("Failed to generate buffer ID.")
        self.size = 0
        self._cleaned = False  # Flag to track if cleanup has been done
        if dtype is not None:
            self.dtype = dtype

    @staticmethod
    def getBoundBuffer(target):
        return Buffer._bound_buffers.get(target) or 0

    @staticmethod
    def isBufferBound(target, bufferID):
        return Buffer.getBoundBuffer(target) == bufferID

    @staticmethod
    def recBoundBuffer(target, bufferID):
        Buffer._bound_buffers[target] = bufferID

    @property
    def bound(self):
        return Buffer.isBufferBound(self.target, self.ID)

    def recBind(self, clear=False):
        bufferID = 0 if clear else self.ID
        Buffer.recBoundBuffer(self.target, bufferID)

    def bind(self):
        if self._cleaned:
            raise RuntimeError("Cannot bind a cleaned up buffer.")
        if not self.bound:
            glBindBuffer(self.target, self.ID)
            self.recBind()

    def unbind(self):
        if self._cleaned:
            raise RuntimeError("Cannot unbind a cleaned up buffer.")
        if self.bound:
            glBindBuffer(self.target, 0)
            self.recBind(True)

    def buffer_data(self, data: Union[np.ndarray, bytes]):
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif not isinstance(data, bytes):
            raise TypeError("Buffer can only receive an np array or bytes object when buffering data")
        if self._cleaned:
            raise RuntimeError("Cannot buffer data to a cleaned up buffer.")
        if not self.bound:
            raise RuntimeError("Cannot buffer data to an unbound buffer")
        self.size = len(data)
        glBufferData(self.target, self.size, data, self.usage)

    def buffer_sub_data(self, offset, data: bytes):
        if self._cleaned:
            raise RuntimeError("Cannot buffer sub-data to a cleaned up buffer.")
        if not self.bound:
            raise RuntimeError("Cannot buffer sub-data to an unbound buffer")
        glBufferSubData(self.target, offset, len(data), data)

    def allocate(self, size):
        if self._cleaned:
            raise RuntimeError("Cannot allocate a cleaned up buffer.")
        if not self.bound:
            raise RuntimeError("Cannot allocate an unbound buffer")
        self.size = size
        glBufferData(self.target, self.size, None, self.usage)

    def get_data(self, offset=0, size=None):
        current_buffer = self._bound_buffers[self.target]
        if current_buffer != self.ID:
            self.bind()
        if self._cleaned:
            raise RuntimeError("Cannot read a cleaned up buffer.")
        if not self.bound:
            raise RuntimeError("Cannot read an unbound buffer")

        # Get buffer size if not specified
        if size is None:
            buffer_size = glGetBufferParameteriv(self.target, GL_BUFFER_SIZE)
        else:
            buffer_size = size

        # Read the data from the buffer
        data = glGetBufferSubData(self.target, offset, buffer_size)
        if current_buffer != self.ID:
            self.unbind()
            glBindBuffer(self.target, current_buffer)

        # Convert to numpy array for easier inspection
        # Assuming the data is in float format for simplicity
        data_array = np.frombuffer(data, dtype=self.dtype)
        return data_array, buffer_size

    def print_data(self, offset=0, size=None):
        data_array, buffer_size = self.get_data()
        print(f"Buffer Data (offset: {offset}, size: {buffer_size}):")
        print(data_array)

    def __del__(self):
        self._cleanup()

    def _cleanup(self):
        if glIsBuffer(self.ID):
            if not self._cleaned and self.ID and self.ID != 0:
                try:
                    glDeleteBuffers(1, [self.ID])
                except TypeError as e:
                    print(f"TypeError during buffer deletion: {e}")
                except OpenGL.GL.error.GLError as e:
                    print(f"OpenGL error during buffer deletion: {e}")
                except Exception as e:
                    print(f"Exception during buffer deletion: {e}")
                finally:
                    self.ID = 0  # Mark as deleted
                    self._cleaned = True  # Set the flag to indicate cleanup is done
            else:
                if not self._cleaned:
                    print(f"Buffer ID is invalid or already deleted: {self.ID}")
        else:
            print(f"Buffer ID {self.ID} does not exist.")


class VertexBuffer(Buffer):
    def __init__(self, usage, dtype=np.float32):
        super().__init__(GL_ARRAY_BUFFER, usage, dtype)


class ElementBuffer(Buffer):
    dtype = GL_UNSIGNED_SHORT

    def __init__(self, usage):
        super().__init__(GL_ELEMENT_ARRAY_BUFFER, usage, np.ushort)


@dataclass
class VertexAttribute:
    name: str
    data: Union[Any, Collection[Any], List[Collection[Any]], np.ndarray]
    normalized: bool = False
    divisor: GLsizei = 0

    def __post_init__(self):
        if not isinstance(self.data, List):
            self.data = [self.data]

        self._num_components = 1
        self._elem_count = 1
        data_point: Union[Any, Collection[Any]] = self.data[0]
        if isinstance(data_point, Collection):  # Indicates Vector or Matrix
            self._elem_count = len(data_point)
            data_point = data_point[0]
            if isinstance(data_point, Collection):  # Indicates this is a Matrix
                self._num_components, self._elem_count = self._elem_count, len(data_point)
                data_point = data_point[0]
                if isinstance(data_point, Collection):
                    raise TypeError("Maximum Vertex Attribute Depth is 2 for Matrix (WxH)")
        self.num_elements = self._num_components * self._elem_count

        self.gl_type: GLenum = None
        self.dtype: np.dtype = None
        elem_size: int = 4  # int32 and float32 are 4 bytes in size
        if isinstance(data_point, int):
            self.gl_type = GL_INT
            self.dtype = np.int32
        elif isinstance(data_point, float):
            self.gl_type = GL_FLOAT
            self.dtype = np.float32
        else:
            raise TypeError("Only Base Data Types of int or float are allowed")

        self.data = np.array(self.data, dtype=self.dtype)
        self.base_size: int = elem_size * self.num_elements

    @property
    def bytes(self) -> bytes:
        return self.data.tobytes()

    @property
    def num_bytes(self) -> int:
        return self.data.nbytes

    @property
    def size(self):
        return len(self.data)

    def set_attribute_pointer(self, start_index: GLsizei, stride: GLsizei, pointer: GLsizei,
                              print_attribute=False) -> int:
        for i in range(self._num_components):
            attr_index = start_index + i
            glEnableVertexAttribArray(attr_index)

            component_bytes = (self.num_bytes // self.num_elements) * self._elem_count
            component_pointer = pointer + component_bytes * i

            if self.gl_type in (GL_INT, GL_UNSIGNED_INT, GL_SHORT, GL_UNSIGNED_SHORT):
                glVertexAttribIPointer(attr_index, self._elem_count, self.gl_type, stride,
                                       ctypes.c_void_p(component_pointer))
            else:
                glVertexAttribPointer(attr_index, self._elem_count, self.gl_type, self.normalized, stride,
                                      ctypes.c_void_p(component_pointer))
            glVertexAttribDivisor(attr_index, self.divisor)
            if print_attribute:
                print(
                    f"Pointer Set:\t{{index: {attr_index} | elements: {self._num_components} | bytes: {component_bytes} | dtype: {self.gl_type} | normalized: {self.normalized} | stride: {stride} | offset: {component_pointer} | divisor: {self.divisor}}}")
        return attr_index + 1


@dataclass
class DataLayout:
    attributes: List[VertexAttribute]

    def __post_init__(self):
        self._strides: Dict[int, int] = {}
        self._pointers: Dict[int, int] = {}
        self._attributes: Dict[int, List[VertexAttribute]] = {}
        self._bytes: Dict[int, Dict[GLenum, bytes]] = {}
        for attribute in self.attributes:
            div = attribute.divisor
            if div not in self._strides.keys():
                self._strides[div] = 0
                self._pointers[div] = 0
                self._attributes[div] = []
                self._bytes[div] = {}
            else:
                if self._strides[div] == 0:
                    self._strides[div] += self._attributes[div][-1].base_size
                self._strides[div] += attribute.base_size
            self._attributes[div].append(attribute)
            if attribute.gl_type not in self._bytes[div]:
                self._bytes[div][attribute.gl_type] = attribute.bytes
            else:
                self._bytes[div][attribute.gl_type] += attribute.bytes


    def set_pointers(self, print_attributes=False):
        index: int = 0
        for divisor in sorted(self._strides.keys()):
            for attribute in self._attributes[divisor]:
                index = attribute.set_attribute_pointer(
                    index, self._strides[divisor], self._pointers[divisor], print_attributes
                )

    def get_data(self) -> List[Tuple[GLenum, int, bytes]]:
        data: List[Tuple[GLenum, int, bytes]] = []
        offset: int = 0
        for div in sorted(self._bytes.keys()):
            for dtype in self._bytes[div].keys():
                data.append((dtype, offset, self._bytes[div][dtype]))
                offset += len(self._bytes[div][dtype])
        return data


class BufferManager:
    def __init__(self):
        self.buffers: dict[int, Buffer] = {}

    def _register_buffer(self, buffer):
        self.buffers[buffer.ID] = buffer

    def create_vbo(self, usage) -> VertexBuffer:
        buffer = VertexBuffer(usage)
        self._register_buffer(buffer)
        return buffer

    def create_ebo(self, usage) -> ElementBuffer:
        buffer = ElementBuffer(usage)
        self._register_buffer(buffer)
        return buffer

    def delete_buffer(self, buffer_id):
        self.buffers.pop(buffer_id, None)

    def _get_buffer(self, buffer_id, buffer_type=Buffer):
        buffer = self.buffers.get(buffer_id)
        if not isinstance(buffer, buffer_type):
            raise TypeError("Buffer Requested is not of Requested Type")
        return buffer

    def get_vbo(self, buffer_id) -> VertexBuffer:
        return self._get_buffer(buffer_id, VertexBuffer)

    def get_ebo(self, buffer_id) -> ElementBuffer:
        return self._get_buffer(buffer_id, ElementBuffer)


class VertexArray:
    def __init__(self):
        self.ID = glGenVertexArrays(1)  # Ensure you get the ID correctly
        self.ebo = None
        self.buffer_manager = BufferManager()
        self.attribute_index = 0

    def bind(self):
        glBindVertexArray(self.ID)
        if self.ebo:
            self.ebo.bind()

    def unbind(self):
        glBindVertexArray(0)
        if self.ebo:
            self.ebo.unbind()

    def add_vbo(self, layout: DataLayout, usage=GL_STATIC_DRAW, print_buffer=True) -> VertexBuffer:
        # Create and bind the vertex buffer object
        buffer: VertexBuffer = self.buffer_manager.create_vbo(usage)

        # Bind the VAO
        self.bind()

        # Bind the VBO
        buffer.bind()
        buffer.allocate(np.sum([attr.num_bytes for attr in layout.attributes]))
        [buffer.buffer_sub_data(*data[1:]) for data in layout.get_data()]
        try:
            # Set vertex attribute pointers
            layout.set_pointers(print_buffer)
        finally:
            # Unbind the VBO
            if print_buffer:
                buffer.print_data()
            buffer.unbind()

        # Unbind the VAO
        self.unbind()

        return buffer

    def set_ebo(self, usage=GL_STATIC_DRAW, data: np.ndarray = None, print_buffer=True) -> ElementBuffer:
        ebo: ElementBuffer = self.buffer_manager.create_ebo(usage)
        ebo.bind()
        ebo.buffer_data(data.tobytes())
        try:
            self.ebo = ebo
        finally:
            if print_buffer:
                ebo.print_data()
            ebo.unbind()
        return ebo


    def _cleanup(self):
        if self.ID and self.ID != 0:  # Check if ID is not None and valid
            try:
                glDeleteVertexArrays(1, [self.ID])
                self.ID = 0  # Mark as deleted
            except OpenGL.GL.error.GLError as e:
                print(f"OpenGL error during VAO deletion: {e}")
            except Exception as e:
                print(f"Exception during VAO deletion: {e}")
            finally:
                self.ID = 0

    def __del__(self):
        self._cleanup()
        # The buffer_manager's buffers should be deleted by their own __del__ methods
        self.buffer_manager = None
