from POGLE.Physics.Collisions import *


class SpatialTree:
    class Node(PhysicalBox):
        def __init__(self, bounds: AABB, activeDimensions: glm.vec3, divisions: int):
            super().__init__(bounds)
            self.objects: list[PhysicalBox] = []
            self.activeDimensions: glm.vec3 = activeDimensions
            self.divisions: int = divisions
            self.childNodes: list[SpatialTree.Node | None] = [None for i in range(self.divisions)]
            self.segment: list[AABB | None] = [None for node in self.childNodes]
            self.activeNodes: int = 0

            # Calculate midpoints for each active dimension

            for i in range(self.divisions):
                subMin: glm.vec3 = glm.vec3(self.min)
                subMax: glm.vec3 = glm.vec3(self.max)
                exp = 0
                for j in range(3):
                    if self.activeDimensions[j]:  # Check if the dimension is active
                        # Apply the bitmask to check whether to split on this dimension
                        if i & (1 << exp):
                            subMin[j] = self.pos[j]  # If the bit is set, use the midpoint as the lower bound
                        else:
                            subMax[j] = self.pos[j]  # If the bit is not set, use the midpoint as the upper bound
                        exp += 1  # Move to the next active dimension
                    else:
                        subMax[j] = self.max[j]  # For inactive dimensions, keep the max bounds unchanged

                self.segment[i] = AABB.from_min_max(subMin, subMax)

        def intersect_aabb(self, range: AABB) -> Hit:
            return self.bounds.intersectAABB(range)

        # def intersectRay(self, range: AABB) -> Hit: return self.bounds.intersectAABB(range)

        def set_active_node(self, index: int):
            self.activeNodes |= (1 << index)

        def clear_active_node(self, index: int):
            self.activeNodes &= ~(1 << index)

        def is_node_active(self, index: int) -> bool:
            return self.activeNodes & (1 << index) != 0

        def is_empty(self) -> bool:
            return len(self.objects) == 0

        def insert(self, obj: PhysicalBox):
            self.objects.append(obj)

        # Less than equal to Min Size
        def lte_min_size(self, minSize: glm.vec3) -> bool:
            return (self.size.x <= minSize.x and
                    self.size.y <= minSize.y and
                    self.size.z <= minSize.z)

        def clear_objects(self):
            self.objects.clear()

        def __str__(self):
            return f"Node(bounds: {self.bounds}, objects: {len(self.objects)},childNodes: {['Active' if self.is_node_active(i) else 'Inactive' for i in range(self.divisions)]}"

    def __init__(self, bounds: AABB, activeDimensions: glm.vec3, minSize: glm.vec3 = glm.vec3(1)):
        self.activeDimensions: glm.vec3 = activeDimensions
        self.divisions: int = 2 ** int(np.sum(activeDimensions))
        self.root: SpatialTree.Node = SpatialTree.Node(bounds, self.activeDimensions, self.divisions)
        self.minSize: glm.vec3 = minSize

    def insert(self, obj: PhysicalBox, node: Node = None) -> bool:
        if node is None:
            node = self.root

        if not node.intersect_aabb(obj.bounds):
            return False  # obj does not intersect node

        if not node.activeNodes:  # node is not subdivided
            # if node is empty or less than/equal to the minimum size it can get
            if node.is_empty() or node.lte_min_size(self.minSize):
                node.insert(obj)
                return True

            self.subdivide(node)
            return self.insert(obj, node)

        # node has subdivided
        return self.insert_into_child(node, obj)

    def insert_into_child(self, node: Node, obj: PhysicalBox) -> bool:
        objMoved: bool = False
        for i in range(self.divisions):
            if self.insert(obj, node.childNodes[i]):
                node.set_active_node(i)
                objMoved = True
        return objMoved

    def subdivide(self, node: Node):
        for i in range(self.divisions):
            node.childNodes[i] = SpatialTree.Node(node.segment[i], self.activeDimensions, self.divisions)

        objectsMoved = 0
        for obj in node.objects:
            objectsMoved += self.insert_into_child(node, obj)

        if objectsMoved != len(node.objects):
            assert "some objects were not moved when subdividing"
        node.clear_objects()

    def query_aabb(self, boxRange: AABB, result: set[PhysicalBox] = None, node: Node = None) -> set[PhysicalBox]:
        if node is None:
            if result is None:
                result = set()
            self.query_aabb(boxRange, result, self.root)
            return result
        else:
            if not node.intersect_aabb(boxRange):  # no intersection between range and node so exit now
                return

            if not node.activeNodes:  # node is not subdivided
                for obj in node.objects:
                    if obj.bounds.intersectAABB(boxRange):
                        result.add(obj)
                return

            # node is subdivided
            for i in range(node.divisions):
                if node.is_node_active(i):
                    self.query_aabb(boxRange, result, node.childNodes[i])
            return


class Octree(SpatialTree):
    def __init__(self, bounds: AABB, minSize: glm.vec3 = glm.vec3(1)):
        super().__init__(bounds, glm.vec3(1), minSize)


class QuadTree(SpatialTree):
    __create_key = object()

    @classmethod
    def _new(cls, bounds: AABB, activeDimensions: glm.vec3, minSize: glm.vec3 = glm.vec3(1)):
        return QuadTree(cls.__create_key, bounds, activeDimensions, minSize)

    def __init__(self, create_key, bounds: AABB, activeDimensions: glm.vec3, minSize: glm.vec3):
        assert (create_key == QuadTree.__create_key), \
            "QuadTree objects must be created using QuadTree._new"
        super().__init__(bounds, activeDimensions, minSize)

    @classmethod
    def XY(cls, bounds: AABB, minSize: glm.vec3 = glm.vec3(1)):
        return cls._new(bounds, glm.vec3(1, 1, 0), minSize)

    @classmethod
    def XZ(cls, bounds: AABB, minSize: glm.vec3 = glm.vec3(1)):
        return cls._new(bounds, glm.vec3(1, 0, 1), minSize)

    @classmethod
    def YZ(cls, bounds: AABB, minSize: glm.vec3 = glm.vec3(1)):
        return cls._new(bounds, glm.vec3(0, 1, 1), minSize)