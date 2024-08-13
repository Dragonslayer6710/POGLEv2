from POGLE.Geometry.Vertex import *


class Shape:
    from POGLE.Geometry.Vertex import _VertexAttribute as VertexAttribute
    positions = []
    indices = []

    instances: Instances = None

    def __init__(self, vertexElements: list = [], vertexAttributes: list[VertexAttribute] = [], instanceElements: list = [], instanceAttributes: list[VertexAttribute] = []):
        vertexElements = interleave_arrays(self.positions, *vertexElements)
        self.vertices: Vertices = Vertices(vertexElements, VertexLayout([FloatVA.Vec3()] + vertexAttributes))
        if instanceElements:
            instanceElements = interleave_arrays(*instanceElements)
            self.instances: Instances = Instances(instanceElements, VertexLayout(instanceAttributes))


class Quad(Shape):
    positions = [
        glm.vec3(-0.5, -0.5, 0.0),
        glm.vec3(-0.5,  0.5, 0.0),
        glm.vec3( 0.5,  0.5, 0.0),
        glm.vec3( 0.5, -0.5, 0.0)
    ]
    indices = [
        0, 1, 2,
        2, 3, 0
    ]


class Pentagon(Shape):
    positions = [
        glm.vec3(0.0, 1.0, 0.0),
        glm.vec3(0.9511, 0.3090, 0.0),
        glm.vec3(0.5878, -0.8090, 0.0),
        glm.vec3(-0.5878, -0.8090, 0.0),
        glm.vec3(-0.9511, 0.3090, 0.0)
    ]
    indices = [
        0, 1, 2,
        0, 2, 4,
        2, 3, 4
    ]
