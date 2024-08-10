from POGLE.Geometry.Vertex import *
class Buffer:
    def __init__(self, buffers = GL_ARRAY_BUFFER, usage = GL_STATIC_DRAW, dtype = GLfloat):
        self.target: GLenum = buffers
        self.ID: GLuint = glGenBuffers(1)
        self.usage = usage
        self.dtype = dtype

    def bind(self):
        glBindBuffer(self.target, self.ID)

    def unbind(self):
        glBindBuffer(self.target, 0)

    def buffer_data(self, size: GLsizeiptr, data: np.ndarray):
        glBufferData(self.target, size, data, GL_STATIC_DRAW)
        feedback = (self.dtype * len(data))()
        glGetBufferSubData(self.target, 0, size, feedback)
        print(list(feedback))

class VertexBuffer(Buffer):
    def __init__(self):
        super().__init__()

class ElementBuffer(Buffer):
    def __init__(self):
        super().__init__(GL_ELEMENT_ARRAY_BUFFER, dtype=GLushort)

class VertexArray:
    def __init__(self, vertices: Vertices, indices: list[int], instances: Vertices = None):
        self.EBO = None

        indices = np.array(indices, np.uint16)

        self.ID = glGenVertexArrays(1)
        self.bind()

        self.VBO = VertexBuffer()
        self.VBO.bind()
        self.VBO.buffer_data(vertices.bytes, vertices.data)
        vertices.setPointers()
        self.VBO.unbind()

        if instances:
            self.IBO = VertexBuffer()
            self.IBO.bind()
            self.IBO.buffer_data(instances.bytes, instances.data)
            instances.setPointers(vertices.nextID())
            self.IBO.unbind()

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