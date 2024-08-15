from POGLE.Shader import *
from POGLE.Geometry.Texture import *
from POGLE.Geometry.Shape3D import *


class Mesh:
    def __init__(self, vertices: Shape | Vertices = None, indices: list[int] = None, textures: list[Texture] = None,
                 instances: Instances = None, primitive=GL_TRIANGLES):
        if None == indices:
            indices = vertices.indices
            vertices: Vertices = vertices.vertices
        self.vertices = vertices
        self.indices = indices
        self.count = len(indices)
        self.texture: Texture = textures
        self.instances: Instances = instances

        self.primitive = primitive

        self.VAO = VertexArray(self.vertices, self.indices, self.instances)
        self.VAO.unbind()

    def bind(self):
        self.VAO.bind()
        if self.texture:
            self.texture.bind()

    def unbind(self):
        self.VAO.unbind()
        if self.texture:
            self.texture.unbind()

    def draw(self, shaderProgram: ShaderProgram, projection: glm.mat4 = None, view: glm.mat4 = None):
        shaderProgram.use()
        if projection:
            shaderProgram.setMat4("uProjection", projection)
        if view:
            shaderProgram.setMat4("uView", view)
        self.bind()
        if self.texture:
            shaderProgram.setInt("tex0", self.texture.get_texture_slot())
        if self.instances:
            glDrawElementsInstanced(self.primitive, self.count, GL_UNSIGNED_SHORT, None, self.instances.count)
        else:
            glDrawElements(self.primitive, self.count, GL_UNSIGNED_SHORT, None)
        self.unbind()


class QuadCubeMesh(Mesh):

    def __init__(self, qc: ColQuadCube):
        # Initialize Mesh with instances
        super().__init__(qc, instances=qc.instances)


class WireframeCubeMesh(Mesh):
    def __init__(self, position: glm.vec3, color: glm.vec3 = Color.BLACK, alpha: float = 1.0, thickness: float = 1.0):
        self.thickness = thickness
        self.shader = ShaderProgram()
        wcCube = WireframeCube(position, color, alpha)
        super().__init__(wcCube, instances=wcCube.instances, primitive=GL_LINES)

    def draw(self, projection: glm.mat4, view: glm.mat4):
        super().draw(self.shader, projection, view)


class CrosshairMesh(Mesh):

    def __init__(self, scale: glm.vec2, color: glm.vec3 = Color.WHITE, alpha: float = 1.0, thickness=1.0):
        self.thickness = thickness
        self.crosshairShader = ShaderProgram("crosshair", "crosshair")
        crosshair = Crosshair(scale, color, alpha)
        super().__init__(crosshair, instances=crosshair.instances, primitive=GL_LINES)

    def draw(self):
        super().draw(self.crosshairShader)
