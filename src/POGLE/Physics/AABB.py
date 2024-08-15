from POGLE.Core.Core import *
class AABB:
    def __init__(self, centre: glm.vec3, size: glm.vec3):
        self.centre = centre
        self.size = size
