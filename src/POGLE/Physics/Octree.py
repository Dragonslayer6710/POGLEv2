from __future__ import annotations

import copy

from POGLE.Physics.Collisions import *

class Tree:
    class Object:
        def __init__(self, boundingBox: AABB, obj):
            self.boundingBox: AABB = boundingBox
            self.obj = obj
            self.alive = True

        def update(self, deltaTime: float) -> bool:
            if self.obj.update:  # if object has update method (requires per frame updates) return result of this
                return self.obj.update(deltaTime)
            return False  # otherwise return false indicating no update since the object doesn't update on its own

    nextID: int = 0

    class Info:
        def __init__(self):
            # items awaiting insertion, we want to accrue as many objects here as possible before we inject them
            # this is slightly more cache friendly
            self.pendingInsertion: deque = deque()

            self.treeReady: bool = False
            self.treeBuilt: bool = False

    info: list[Info] | Info = []

    # min size for enclosing region is a 1x1x1 cube
    MIN_SIZE: int = 1
    MIN_SIZE_VEC: glm.vec3 = glm.vec3(MIN_SIZE)

    childNode: list[Tree] = None
    segmentAABB: list[AABB] = None
    segmentCount: int = None
    def __init__(self, dimensions: int, region: AABB = AABB(glm.vec3(), glm.vec3()), objects=None, rootID=-1):
        # State Bools
        self.hasChildren: bool = False
        self.isRoot: bool = False

        if rootID < 0:
            self.isRoot = True
            self.ID: int = Tree.nextID
            type(self).nextID += 1
            type(self).info.append(Tree.Info())
        else:
            self.ID: int = rootID

        self.info = Tree.info[self.ID]

        # Bounding Region and objects
        if objects is None:
            objects = []
        self.region: AABB = region
        self.objects: list[Tree.Object] = objects

        # bitmask indicating which child nodes are actively being used
        self.activeNodes: int = 0

        # how many frame to wait before deleting an empty tree branch (not a constant, max lifespan doubles
        # every time a node is reused until it hits a hard coded constant of 64
        self.maxLifespan: int = 8
        self.curLife: int = -1  # countdown time showing how much time left to live

        # reference to parent for updates
        self._parent: Tree = None
        if self.isRoot:
            if not type(self).segmentCount:
                self.segmentCount =  2**dimensions
                type(self).segmentCount = self.segmentCount

            # All child octants
            if not type(self).childNode:
                self.childNode = [None for ci in range(self.segmentCount)]
                type(self).childNode = copy.deepcopy(self.childNode)

            if not type(self).childNode:
                self.segmentAABB = [None for ci in range(self.segmentCount)]
                type(self).segmentAABB = copy.deepcopy(self.childNode)

            self.build_tree()

    def find_enclosing_cube(self):
        pass

    def create_node(self, region: AABB, objects: list[Object] | Object) -> Octree:
        if type(objects) == list:
            if len(objects):
                treeNode = Octree(region, objects, self.ID)
            else:
                return None
        else:
            treeNode = Octree(region, [objects], self.ID)

        treeNode._parent = self
        return treeNode

    def build_tree(self):
        if len(self.objects) <= 1:
            return

        dimensions: glm.vec3 = self.region.size
        if glm.vec3() == dimensions:
            self.find_enclosing_cube()
            dimensions = self.region.size

        # Check to see if dims of box are greater than minimum dimensions
        if glm.vec3(1) == dimensions <= Tree.MIN_SIZE_VEC:
            return

        center: glm.vec3 = self.region.pos
        rMin: glm.vec3 = self.region.min
        rMax: glm.vec3 = self.region.max

        # Create subdivided regions for each octant
        quarter: glm.vec3 = self.region.half / 2
        eighth: glm.vec3 = quarter / 2

        if 4 == self.segmentCount:
            self.segmentAABB = [
                AABB(rMin + eighth, quarter),
                AABB(glm.vec3(center.xy, rMin.z) + eighth, quarter),
                AABB(glm.vec3(rMin.xy, center.z) + eighth, quarter),
                AABB(center + eighth, quarter),
            ]
            print()

        elif 8 == self.segmentCount:
            self.segmentAABB = [
                AABB(rMin + eighth, quarter),
                AABB(glm.vec3(center.x, rMin.yz) + eighth, quarter),
                AABB(glm.vec3(center.x, rMin.y, center.z) + eighth, quarter),
                AABB(glm.vec3(rMin.xy, center.z) + eighth, quarter),
                AABB(glm.vec3(rMin.x, center.y, rMin.z) + eighth, quarter),
                AABB(glm.vec3(center.xy, rMin.z) + eighth, quarter),
                AABB(center + eighth, quarter),
                AABB(glm.vec3(rMin.x, center.yz) + eighth, quarter),
            ]

        # this will contain all objects which fit within each respective octant
        segList: list[list[Tree.Object]] = [[] for o in self.segmentAABB]

        # this list contains all of the objects which got moved down the tree and can be delisted from this node
        delist: list = []
        cnt = 0
        for obj in self.objects:
            if obj.boundingBox.min != obj.boundingBox.max:
                intersects = []
                for i in range(self.segmentCount):
                    if self.segmentAABB[i].contains(obj.boundingBox):
                        segList[i].append(obj)
                        delist.append(obj)
                        intersects = []
                        break
                    elif self.segmentAABB[i].intersectAABB(obj.boundingBox):
                        intersects.append(i)
                if len(intersects):
                    for i in intersects:
                        segList[i].append(obj)
                    delist.append(obj)

        # delist every moved object from this node
        for obj in delist:
            self.objects.remove(obj)

        # Create child nodes where there are items contained in the bounding region
        for i in range(self.segmentCount):
            if len(segList[i]):
                if not self.hasChildren:
                    self.hasChildren = True
                self.childNode[i] = self.create_node(self.segmentAABB[i], segList[i])
                self.activeNodes |= 1 << i
                self.childNode[i].build_tree()
            else:
                self.childNode[i] = None

        self.info.treeBuilt = True
        self.info.treeReady = True

        # a tree has already been created, so we're going to try to insert an item into the tree without rebuilding the whole thing
    def insert(self, obj: Object) -> bool:
        # if the current node is an empty leaf node, just insert and leave
        if not len(self.objects) and not self.activeNodes:
            self.objects.append(obj)
            return True

        # check to see if the dimensions of the box are greater than the minimum dimensions.
        # if we're at the smallest size, just insert. we can't fo lower
        dimensions: glm.vec3 = self.region.size
        if glm.vec3(1) == dimensions <= Tree.MIN_SIZE_VEC:
            self.objects.append(obj)
            return True

        # the object won't fit into the current region, so it won't fit into any child regions
        # therefore, try to push it up the tree. if we're at the root node, we need to resize the whole tree
        regionContainsObj: bool = self.region.contains(obj.boundingBox)
        if not regionContainsObj:
            if self._parent:
                return self._parent.insert(obj)
            else:
                return False

        # at this point we at least know this region can contain the object but there are child nodes let's see if the object wil fit
        # within a subregion of this region

        # First, is the item completely contained within the root bounding box
        if obj.boundingBox.max != obj.boundingBox.min:
            found: bool = False
            # we will try to place the object into a child node. if we can't, insert into current node obj list
            for i in range(self.segmentCount):
                # is the object fully contained within a quadrant
                if self.segmentAABB[i].contains(obj.boundingBox):
                    if self.childNode[i]:
                        return self.childNode[i].insert(
                            obj)  # add item into that tree and let the child tree deal with it
                    else:
                        self.childNode[i] = self.create_node(self.segmentAABB[i],
                                                             obj)  # creat new tree and give item to it
                        self.activeNodes |= 1 << i
                    found = True
                # we couldn't fir the item into a smaller box so we'll insert it into this region
                if not found:
                    self.objects.append(obj)

        # either the item lies outside of the enclosed bounding box or it is intersecting it. either way, we need to rebuild
        # the entire tree by enlarging the containing bounding box
        return False

    # Process all pending insertions by inserting them into the tree
    # Consider deprecating this?
    def update_tree(self):
        if not self.info.treeBuilt:
            while len(Tree.info.pendingInsertion) != 0:
                self.objects.append(self.info.pendingInsertion.popleft())
            self.build_tree()
        else:
            while len(Tree.info.pendingInsertion) != 0:
                self.insert(self.info.pendingInsertion.popleft())
        self.info.treeReady = True

    def update(self, deltaTime: float):
        if self.info.treeBuilt and self.info.treeReady:
            # start count down death timer for any leaf nodes which don't have objects or children
            # when timer reaches zero, we delete the leaf. if the node is reused before death, double its lifespan
            # this gives us a frequency usage score and lets us avoid allocating and deallocating memory unnecessarily
            listSize = len(self.objects)
            if not listSize:
                if not self.hasChildren:
                    if self.curLife == -1:
                        self.curLife = self.maxLifespan
                    else:
                        self.curLife -= 1
            else:
                if self.curLife > -1:
                    if self.maxLifespan <= 64:
                        self.maxLifespan *= 2
                    self.curLife = -1

            # go through and update every object in current tree node
            movedObjects: list[Tree.Object] = []
            [movedObjects.append(obj) if obj.update(deltaTime) else None for obj in self.objects]

            # prune any dead objects from the tree
            for obj in self.objects:
                if not obj.alive:
                    if obj in movedObjects:
                        movedObjects.remove(obj)
                    self.objects.remove(obj)
                    listSize -= 1

            # prune out any dead branches in the tree
            flags = self.activeNodes
            index = 0
            while flags > 0:
                if (flags & 1) == 1 and self.childNode[index].curLife == 0:
                    if len(self.childNode[index].objects):
                        # Uncomment the next line to raise an exception in Python
                        # raise Exception("Tried to delete a used branch!")
                        self.childNode[index].curLife = -1
                    else:
                        self.childNode[index] = None
                        # Remove the node from the active nodes flag list
                        self.activeNodes ^= (1 << index)

                flags >>= 1
                index += 1

            # recursively update any child nodes
            flags = self.activeNodes
            index = 0
            while flags > 0:
                if (flags & 1) == 1:
                    if self.childNode:
                        if self.childNode[index]:
                            self.childNode[index].update(deltaTime)

                flags >>= 1
                index += 1

            # if an object moved, we can insert it into the parent and that will insert it into the correct tree node
            # note that we have to do this last so we don't accidentally update the same object more than once per frame
            for movedObj in movedObjects:
                current: Octree = self

                # figure out how far up the tree we need to go to reinsert our moved object
                # we are using a bounding rect
                # try to move the object into an enclosing parent node until we've got full containment
                if movedObj.boundingBox.max != movedObj.boundingBox.min:
                    while not current.region.contains(movedObj.boundingBox):
                        if current._parent:
                            current = current._parent
                        else:
                            break  # prevent infinite loops when we go out of bounds of the root node region

                # now, remove the object from the current node and insert it into the current containing node
                self.objects.remove(movedObj)
                current.insert(movedObj)  # this will try to insert the object as deep into the tree as we can go
        else:
            if len(self.info.pendingInsertion):
                self.process_pending_items()
                self.update(deltaTime)  # try again

    def queryAABB(self, box: AABB) -> list[Collider|AABB]:
        if box.intersectAABB(self.region):
            colliders: list[Collider | AABB] = []
            if len(self.objects):
                [colliders.append(obj.boundingBox) for obj in self.objects]
            # recursively update any child nodes
            flags = self.activeNodes
            index = 0
            while flags > 0:
                if (flags & 1) == 1:
                    if self.childNode:
                        if self.childNode[index]:
                            colliders += self.childNode[index].queryAABB(box)

                flags >>= 1
                index += 1
            return colliders
        return []
class QuadTree(Tree):
    def __init__(self, region: AABB = AABB(glm.vec3(), glm.vec3()), objects=None, rootID=-1):
        super().__init__(2, region, objects, rootID)

class Octree(Tree):
    def __init__(self, region: AABB = AABB(glm.vec3(), glm.vec3()), objects=None, rootID=-1):
        super().__init__(3, region, objects, rootID)
