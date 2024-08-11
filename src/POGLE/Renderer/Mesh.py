from POGLE.Shader import *
from POGLE.Geometry.Texture import *
from POGLE.Geometry.Shape3D import *


class Mesh:
    def __init__(self, vertices: Vertices = None, indices: list[int] = None, textures: list[Texture] = None,
                 instances: Instances = None, primitive=GL_TRIANGLES):
        if None == indices:
            indices = vertices.indices
            vertices: Vertices = vertices.vertices
        self.vertices = vertices
        self.indices = indices
        self.count = len(indices)
        self.textures = textures
        self.instances: Instances = instances

        self.primitive = primitive

        self.VAO = VertexArray(self.vertices, self.indices, self.instances)
        self.VAO.unbind()

    def draw(self, shaderProgram: ShaderProgram = None):
        self.VAO.bind()
        if self.instances:
            glDrawElementsInstanced(self.primitive, self.count, GL_UNSIGNED_SHORT, None, self.instances.count)
        else:
            glDrawElements(self.primitive, self.count, GL_UNSIGNED_SHORT, None)
        self.VAO.unbind()


class QuadCubeMesh(Mesh):

    def __init__(self, qc: QuadCube):
        # Initialize Mesh with instances
        super().__init__(qc, instances=qc.instances)

class WireframeCubeMesh(Mesh):
    def __init__(self):
        super().__init__(Shapes.WireframeCube, instances=Instances([glm.mat4()]), primitive=GL_LINES)
