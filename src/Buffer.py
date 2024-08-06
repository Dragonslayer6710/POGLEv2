import numpy as np

from Vertex import *
class Buffer:
    def __init__(self, buffers = GL_ARRAY_BUFFER, usage = GL_STATIC_DRAW):
        self.buffers: GLenum = buffers
        self.ID: GLuint = glGenBuffers(1)
        self.usage = usage

    def bind(self):
        glBindBuffer(self.buffers, self.ID)

    def unbind(self):
        glBindBuffer(self.buffers, 0)

    def buffer_data(self, size: GLsizeiptr, data: np.ndarray):
        glBufferData(self.buffers, size, data, GL_STATIC_DRAW)
        r_data = np.frombuffer(glGetBufferSubData(self.buffers, 0, size), dtype=data.dtype)
        print(r_data)


class VertexBuffer(Buffer):
    def __init__(self):
        super().__init__()

class ElementBuffer(Buffer):
    def __init__(self):
        super().__init__(GL_ELEMENT_ARRAY_BUFFER)

class VertexArray:
    def __init__(self, vertices: Vertices, indices: np.ndarray):
        self.EBO = None

        self.ID = glGenVertexArrays(1)
        self.bind()

        self.VBO = VertexBuffer()
        self.VBO.bind()
        self.VBO.buffer_data(vertices.bytes, vertices.data)
        vertices.setPointers()

        self.EBO = ElementBuffer()
        self.EBO.bind()
        self.EBO.buffer_data(indices.nbytes, indices)

        self.unbind()

    def bind(self):
        glBindVertexArray(self.ID)
        if self.EBO:
            self.EBO.bind()

    def unbind(self):
        glBindVertexArray(0)
        self.EBO.unbind()