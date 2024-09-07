import numpy as np

from POGLE.Shader import *
from POGLE.Geometry.Texture import *
from POGLE.Geometry.Shape3D import *


class Mesh:
    def __init__(self, vertices: DataSet, indices: list[int], textures: list[Texture] = None,
                 instances: Optional[Union[DataSet, List[DataSet]]] = None, primitive=GL_TRIANGLES):
        self.vertices = vertices
        self.indices = indices
        self.count = len(indices)
        self.texture: Texture = textures
        self.instances: DataSet = instances

        self.primitive = primitive

        self.VAO = VertexArray(self.vertices, self.indices, self.instances)
        self.VAO.unbind()

        self.UBO: UniformBuffer = UniformBuffer()

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

        self.bind()
        if self.texture:
            shaderProgram.setInt("tex0", self.texture.get_texture_slot())
        if self.instances:
            glDrawElementsInstanced(self.primitive, self.count, GL_UNSIGNED_SHORT, None, self.instances.count)
        else:
            glDrawElements(self.primitive, self.count, GL_UNSIGNED_SHORT, None)
        self.unbind()


# class QuadCubeMesh(Mesh):
#
#     def __init__(self, qc: QuadCube):
#         # Initialize Mesh with instances
#         super().__init__(qc, instances=qc.instances)
#
#
# class WireframeCubeMesh(Mesh):
#     def __init__(self, wfqCube: WireframeQuadCube, instances: DataSet, thickness: float = 2.0):
#         self.thickness = thickness
#         self.shader = ShaderProgram("wireframe_block")
#         self.shader.bind_uniform_block("Matrices")
#         self.shader.bind_uniform_block("BlockSides")
#         super().__init__(wfqCube.vertices, wfqCube.indices, instances=instances, primitive=GL_LINES)
#
#     def draw(self, projection: glm.mat4, view: glm.mat4):
#         glLineWidth(self.thickness)
#         glDisable(GL_DEPTH_TEST)
#         super().draw(self.shader, projection, view)
#         glEnable(GL_DEPTH_TEST)
#         glLineWidth(1.0)

class CrosshairMesh(Mesh):

    def __init__(self, scale: glm.vec2, color: glm.vec3 = Color.WHITE, alpha: float = 1.0, thickness=1.0):
        self.thickness = thickness
        self.crosshairShader = ShaderProgram("crosshair", "crosshair")
        self.crosshairShader.bind_uniform_block("Matrices")
        crosshair = Crosshair(scale, color, alpha)
        super().__init__(crosshair, instances=crosshair.instances, primitive=GL_LINES)

    def draw(self):
        super().draw(self.crosshairShader)

class CubeMesh(Mesh):
    def __init__(self, modelMatrix: glm.mat4, color: glm.vec3 = Color.WHITE, alpha: float = 1.0):
        self.shader = ShaderProgram()
        self.shader.bind_uniform_block("Matrices")
        cube = Cube(color, alpha, modelMatrix)
        super().__init__(cube, instances=cube.instances)

    def draw(self, projection: glm.mat4, view: glm.mat4):
        super().draw(self.shader, projection, view)

class LineSegmentMesh(Mesh):
    def __init__(self, ray: Ray, color: glm.vec3 = Color.BLACK, alpha: float = 1.0):
        self.shader = ShaderProgram("ray")
        self.shader.bind_uniform_block("Matrices")
        lineSegment = Line(ray, color, alpha)
        super().__init__(lineSegment, instances=lineSegment.instances, primitive=GL_LINES)

    def draw(self, projection: glm.mat4, view: glm.mat4):
        super().draw(self.shader, projection, view)