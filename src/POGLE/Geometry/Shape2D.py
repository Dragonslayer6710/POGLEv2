from POGLE.Geometry.Vertex import *


class Shape:
    from POGLE.Geometry.Vertex import _VertexAttribute as VertexAttribute
    positions = []
    indices = []

    instances: Instances = None

    def __init__(self, instanceLayout: VertexLayout = None, instanceData: list = None):
        self.vertices: Vertices = Vertices(self.positions, VertexLayout([FloatVA.Vec3()]))
        if instanceLayout:
            self.instances: Instances = Instances(instanceData, instanceLayout)


class Quad(Shape):
    positions = [
        glm.vec3(-1.0, -1.0, 0.0),
        glm.vec3(1.0, -1.0, 0.0),
        glm.vec3(-1.0, 1.0, 0.0),
        glm.vec3(1.0, 1.0, 0.0)
    ]
    indices = [
        0, 1, 2,
        1, 2, 3
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
