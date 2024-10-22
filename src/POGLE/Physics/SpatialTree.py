from POGLE.Physics.Collisions import *


class SpatialTree:
    class Node(PhysicalBox):
        def __init__(self, bounds: AABB, activeDimensions: glm.vec3, divisions: int):
            super().__init__(bounds)
            self.objects: list[PhysicalBox] = []
            self.activeDimensions: glm.vec3 = activeDimensions
            self.divisions: int = divisions
            self.childNodes: list[SpatialTree.Node | None] = [None for i in range(self.divisions)]
            self.ray: list[AABB | None] = [None for node in self.childNodes]
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

                self.ray[i] = AABB.from_min_max(subMin, subMax)

        def intersect_aabb(self, boxRange: AABB) -> Hit:
            return self.bounds.intersectAABB(boxRange)

        def intersect_segment(self, ray: Ray) -> Hit:
            return self.bounds.intersectSegment(ray)

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
        self.to_insert: Set[PhysicalBox] = set()
        self.to_remove: Set[PhysicalBox] = set()

    def queue_insert(self, obj: PhysicalBox) -> bool:
        if obj in self.to_remove:
            self.to_remove.remove(obj)
        self.to_insert.add(obj)

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
            node.childNodes[i] = SpatialTree.Node(node.ray[i], self.activeDimensions, self.divisions)

        objectsMoved = 0
        for obj in node.objects:
            objectsMoved += self.insert_into_child(node, obj)

        if objectsMoved != len(node.objects):
            assert "some objects were not moved when subdividing"
        node.clear_objects()

    def queue_remove(self, obj: PhysicalBox) -> bool:
        if obj in self.to_remove:
            self.to_remove.remove(obj)
        self.to_insert.add(obj)

    def remove(self, obj: PhysicalBox, node: Node = None) -> bool:
        """
        Remove an object from the tree.

        :param obj: The PhysicalBox object to remove.
        :param node: The node to start the removal from, defaults to the root.
        :return: True if the object was removed, False if the object was not found.
        """
        if node is None:
            node = self.root

        if not node.intersect_aabb(obj.bounds):
            return False # Object does not intersect with this node

        # handle non-subdivided nodes
        if not node.activeNodes:
            if obj in node.objects:
                node.objects.remove(obj)
                return True
            return False # Object not found in this node

        # Node is subdivided, attempt to remove from children
        removed = False
        for i in range(self.divisions):
            child = node.childNodes[i]
            if child and node.is_node_active(i):
                removed |= self.remove(obj, child)

        # If an object was removed, check if we can collapse the node
        if removed and self.can_collapse(node):
            self.collapse(node)

        return removed

    def can_collapse(self, node: Node) -> bool:
        """
        Determine if a node can collapse, i.e., all its children are empty.

        :param node: The node to check.
        :return: True if the node can collapse, False otherwise.
        """
        for i in range(self.divisions):
            child = node.childNodes[i]
            if node.is_node_active(i) and child:
                # Recursively check if child can collapse and is empty
                if not child.is_empty() or not self.can_collapse(child):
                    return False
        return True

    def collapse(self, node: Node):
        """
        Collapse a node by moving all objects from its children back to the node and deactivating the children.

        :param node: The node to collapse.
        """
        for i in range(self.divisions):
            if node.is_node_active(i):
                child = node.childNodes[i]
                if child:
                    node.objects.extend(child.objects)
                    child.clear_objects()
                    node.childNodes[i] = None  # Clear the reference to the child node
                    node.clear_active_node(i) # Clear the active node bit


    def query_aabb(self, boxRange: AABB, result: set[PhysicalBox] = None, node: Node = None) -> set[PhysicalBox]:
        if node is None:
            if result is None:
                result = set()
            self.query_aabb(boxRange, result, self.root)
            return result
        else:
            if not node.intersect_aabb(boxRange):  # no intersection, stop recursion
                return result

            # add objects that intersect the aabb
            if not node.activeNodes:  # node is not subdivided
                for obj in node.objects:
                    if obj.bounds.intersectAABB(boxRange):
                        result.add(obj)
                return result

            # node is subdivided; recursively query child nodes
            for i in range(node.divisions):
                child = node.childNodes[i]
                if child and node.is_node_active(i):
                    self.query_aabb(boxRange, result, child)
            return result

    def query_segment(self, ray: Ray, result: set[PhysicalBox] = None, node: Node = None) -> set[PhysicalBox]:
        if node is None:
            if result is None:
                result = set()
            self.query_segment(ray, result, self.root)
            return result
        else:
            if not node.intersect_segment(ray):  # no intersection between range and node so exit now
                return result

            if not node.activeNodes:  # node is not subdivided
                for obj in node.objects:
                    if obj.bounds.intersectSegment(ray):
                        result.add(obj)
                return result

            # node is subdivided
            for i in range(node.divisions):
                child = node.childNodes[i]
                if child and node.is_node_active(i):
                    self.query_segment(ray, result, child)
            return result

    def update(self):
        for _ in range(len(self.to_insert)):
            self.insert(self.to_insert.pop())
        for _ in range(len(self.to_remove)):
            self.remove(self.to_remove.pop())

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
