from POGLE.Geometry.Vertex import *


class Shape:
    from POGLE.Geometry.Vertex import _VertexAttribute as VertexAttribute
    positions = []
    indices = []

    instances: Instances = None
    initialVertexElements = [FloatVA.Vec3()]

    def __init__(self, vertexElements: list = [], vertexAttributes: list[VertexAttribute] = [],
                 instanceElements: list = [], instanceAttributes: list[VertexAttribute] = []):
        vertexElements = interleave_arrays(self.positions, *vertexElements)
        self.vertices: Vertices = Vertices(vertexElements, VertexLayout(self.initialVertexElements + vertexAttributes))
        if instanceElements:
            instanceElements = interleave_arrays(*instanceElements)
            self.instances: Instances = Instances(instanceElements, VertexLayout(instanceAttributes))


class Quad(Shape):
    positions = [
        glm.vec3(-1.0, -1.0, 0.0) / 2,
        glm.vec3(-1.0, 1.0, 0.0) / 2,
        glm.vec3(1.0, 1.0, 0.0) / 2,
        glm.vec3(1.0, -1.0, 0.0) / 2
    ]
    indices = [
        0, 1, 2,
        2, 3, 0
    ]


class Pentagon(Shape):
    positions = [
        glm.vec3(0.0, 1.0, 0.0) / 2,
        glm.vec3(0.9511, 0.3090, 0.0) / 2,
        glm.vec3(0.5878, -0.8090, 0.0) / 2,
        glm.vec3(-0.5878, -0.8090, 0.0) / 2,
        glm.vec3(-0.9511, 0.3090, 0.0) / 2
    ]
    indices = [
        0, 1, 2,
        0, 2, 4,
        2, 3, 4
    ]


class Crosshair(Shape):
    positions = [
        glm.vec2(0.0, 1.0),
        glm.vec2(0.0, -1.0),
        glm.vec2(-1.0, 0.0),
        glm.vec2(1.0, 0.0)
    ]
    indices = [
        0, 1,
        2, 3
    ]
    initialVertexElements = [FloatVA.Vec2()]

    def __init__(self, scale: glm.vec2, color: glm.vec3, alpha: float):
        super().__init__(instanceElements=[[scale], [color], [alpha]],
                         instanceAttributes=[FloatVA.Vec2(1), FloatVA.Vec3(1), FloatVA.Single(1)])
