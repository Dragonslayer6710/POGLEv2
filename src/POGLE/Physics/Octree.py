from POGLE.Physics.AABB import *

class OctreeObject:
    def __init__(self, boundingBox: AABB, object):
        self.boundingBox: AABB = boundingBox
        self.object = object

class Octree:
    def __init__(self, region: AABB, objList: list[OctreeObject]):
        region = AABB()
