from POGLE.Core.Core import *


class Collider:
    pass


class Hit:
    def __init__(self, collider: Collider):
        self.collider: Collider = collider  # collider
        self.pos: glm.vec3 = glm.vec3()  # position of intersect
        self.normal: glm.vec3 = glm.vec3()  # surface normal at point of contact
        self.delta: glm.vec3 = glm.vec3()  # overlap between two objects, vector to correct
        self.time: float = 0.0  # fraction from 0 to 1 for segment / sweep indicating how far along line collision occured

class Sweep:
    def __init__(self):
        self.hit: Hit = None  # hit object or None if no collision
        self.pos: glm.vec3 = glm.vec3()  # furthest point object reached along swept path befor hit
        self.time: float = 1.0  # copy of self.hit.time, offset by epsilon or 1 if object didn't hit anything during sweep


class AABB:
    pos: glm.vec3
    half: glm.vec3
    size: glm.vec3
    min: glm.vec3
    max: glm.vec3


class Collider:
    hitRecall: dict[Collider, dict[Collider | glm.vec3, Hit]] = {}
    sweepRecall: dict[Collider, dict[Collider | glm.vec3, Sweep]] = {}
    def sweepAABB(self, box: AABB, delta: glm.vec3) -> Sweep:
        pass

    def __init__(self):
        Collider.hitRecall[self] = {}
        self.hitRecall: dict[Collider | glm.vec3, Hit] = Collider.hitRecall[self]
        Collider.sweepRecall[self] = {}
        self.sweepRecall: dict[Collider | glm.vec3, Sweep] = Collider.sweepRecall[self]

    def recallHit(self, collider: Collider | glm.vec3) -> Hit | None:
        if self.hitRecall.get(collider):
            return self.hitRecall.pop(collider)

    def recallSweep(self, collider: Collider | glm.vec3) -> Hit | None:
        if self.sweepRecall.get(collider):
            return self.sweepRecall.pop(collider)

class AABB(Collider):
    __create_key = object()

    @classmethod
    def _new(cls, pos: glm.vec3, size: glm.vec3):
        return AABB(cls.__create_key, pos, size)

    def __init__(self, create_key, pos: glm.vec3, size: glm.vec3):
        assert (create_key == AABB.__create_key), \
            "AABB objects must be created using AABB._new"
        super().__init__()
        self.pos: glm.vec3 = pos
        self.size: glm.vec3 = size


    @property
    def half(self) -> glm.vec3:
        return self.size / 2

    @property
    def min(self) -> glm.vec3:
        return self.pos - self.half

    @property
    def max(self) -> glm.vec3:
        return self.pos + self.half

    @classmethod
    def from_min_max(cls, min: glm.vec3, max: glm.vec3):
        size: glm.vec3 = max - min
        pos: glm.vec3 = min + size / 2
        return cls._new(pos, size)

    @classmethod
    def from_pos_size(cls, pos: glm.vec3, size: glm.vec3 = glm.vec3(1.0)):
        return cls._new(pos, size)

    def does_overlap(self, overlap: glm.vec3) -> bool:
        return overlap.x > 0 and overlap.y > 0 and overlap.z > 0

    def intersectPoint(self, point: glm.vec3) -> Hit:
        # Calculate the vector difference between the centers of the AABB and point
        delta = point - self.pos

        # Determine the sign of the difference for each axis (direction of the collision)
        sign = glm.sign(delta)

        # Calculate the overlap on each axis by subtracting the absolute distance
        # from the sum of the half-extents of the two boxes
        overlap = self.half - glm.abs(delta)

        # Check if there is an intersection by verifying if all components of `overlap` are positive
        if self.does_overlap(overlap):
            hit = Hit(self)
            if overlap.x < overlap.y and overlap.x < overlap.z:
                hit.delta.x = overlap.x * sign.x
                hit.normal.x = sign.x
                hit.pos.x = self.pos.x + (self.half.x * sign.x)
                hit.pos.y = point.y
                hit.pos.z = point.z
            elif overlap.y < overlap.z:
                hit.delta.y = overlap.y * sign.y
                hit.normal.y = sign.y
                hit.pos.x = point.x
                hit.pos.y = self.pos.y + (self.half.y * sign.y)
                hit.pos.z = point.z
            else:
                hit.delta.z = overlap.z * sign.z
                hit.normal.z = sign.z
                hit.pos.x = point.x
                hit.pos.y = point.y
                hit.pos.z = self.pos.z + (self.half.z * sign.z)
            self.hitRecall[point] = hit
            return hit

        # return None if no intersection
        return None

    def intersectSegment(self, pos: glm.vec3, delta: glm.vec3, padding: glm.vec3() = glm.vec3()) -> list[Hit]:
        scale = 1.0 / delta
        sign = glm.sign(scale)
        nearTime = (self.pos - sign * (self.half + padding) - pos) * scale
        farTime = (self.pos + sign * (self.half + padding) - pos) * scale

        if nearTime.x > farTime.x or nearTime.y > farTime.y or nearTime.z > farTime.z:
            return None

        nTime = max(nearTime.x, nearTime.y, nearTime.z)
        fTime = min(farTime.x, farTime.y, farTime.z)

        if nTime >= 1 or fTime <= 0:
            return None

        nearHit = Hit(self)
        nearHit.time = clamp(nTime, 0, 1)
        if nearTime.x > nearTime.y and nearTime.x > nearTime.z:
            nearHit.normal.x = -sign.x
        elif nearTime.y > nearTime.z:
            nearHit.normal.y = -sign.y
        else:
            nearHit.normal.z = -sign.z
        nearHit.delta = (1.0 - nearHit.time) * - delta

        farHit = Hit(self)
        farHit.time = clamp(fTime, 0, 1)
        if nearTime.x < nearTime.y and nearTime.x < nearTime.z:
            farHit.normal.x = sign.x
        elif nearTime.y > nearTime.z:
            farHit.normal.y = sign.y
        else:
            farHit.normal.z = sign.z
        farHit.delta = (1.0 - farHit.time) * - delta

        self.hitRecall[nearTime] = nearHit
        self.hitRecall[farTime] = farHit
        return [nearHit, farHit]

    def intersectAABB(self, box: AABB) -> Hit:
        # Calculate the vector difference between the centers of the two boxes
        delta = box.pos - self.pos

        # Determine the sign of the difference for each axis (direction of the collision)
        sign = glm.sign(delta)

        # Calculate the overlap on each axis by subtracting the absolute distance
        # from the sum of the half-extents of the two boxes
        overlap = (box.half + self.half) - glm.abs(delta)

        # Check if there is an intersection by verifying if all components of `overlap` are positive
        if self.does_overlap(overlap):
            hit = Hit(self)
            if overlap.x < overlap.y and overlap.x < overlap.z:
                hit.delta.x = overlap.x * sign.x
                hit.normal.x = sign.x
                hit.pos.x = self.pos.x + (self.half.x * sign.x)
                hit.pos.y = box.pos.y
                hit.pos.z = box.pos.z
            elif overlap.y < overlap.z:
                hit.delta.y = overlap.y * sign.y
                hit.normal.y = sign.y
                hit.pos.x = box.pos.x
                hit.pos.y = self.pos.y + (self.half.y * sign.y)
                hit.pos.z = box.pos.z
            else:
                hit.delta.z = overlap.z * sign.z
                hit.normal.z = sign.z
                hit.pos.x = box.pos.x
                hit.pos.y = box.pos.y
                hit.pos.z = self.pos.z + (self.half.z * sign.z)
            self.hitRecall[box] = hit
            return hit

        # return None if no intersection
        return None

    def sweepAABB(self, box: AABB, delta: glm.vec3) -> Sweep:
        sweep = Sweep()
        if glm.vec3() == 0:
            sweep.pos = box.pos
            sweep.hit = self.intersectAABB(box)
            if sweep.hit:
                sweep.hit.time = 0
                sweep.time = 0
            else:
                sweep.time = 1
            return sweep
        sweep.hit = self.intersectSegment(box.pos, delta, box.half)
        if sweep.hit:
            sweep.time = clamp(sweep.hit.time - EPSILON, 0, 1)
            sweep.pos = box.pos + delta * sweep.time
            direction = glm.normalize(glm.vec3(delta))
            sweep.hit.pos = glm.clamp(sweep.hit.pos + direction * box.half, self.pos - self.half,
                                      self.pos + self.half)
        else:
            sweep.pos = box.pos + delta
            sweep.time = 1
        return sweep

    def sweepInto(self, staticColliders: list[Collider], delta: glm.vec3) -> Sweep:
        nearest = Sweep()
        nearest.time = 1
        nearest.pos = self.pos + delta
        for staticCollider in staticColliders:
            sweep = staticCollider.sweepAABB(self, delta)
            if sweep.time < nearest.time:
                nearest = sweep
        return nearest

    def contains(self, box: AABB) -> bool:
        # Check if the entire 'box' is within the 'self' bounds
        return (self.min.x <= box.min.x and
                self.min.y <= box.min.y and
                self.min.z <= box.min.z and
                self.max.x >= box.max.x and
                self.max.y >= box.max.y and
                self.max.z >= box.max.z)

    def __str__(self):
        return f"AABB(pos: {self.pos}, size: {self.size})"


_collider = Collider


class Physical:
    _collider: Collider | None

    def __init__(self, collider: type(_collider) = None):
        self._collider = collider

    def _set_collider(self, collider: type(_collider)):
        self._collider: type(collider) = collider

    def recallHit(self, collider: Collider | glm.vec3) -> Hit:
        return self._collider.recallHit(collider)

    def recallSweep(self, collider: Collider | glm.vec3) -> Sweep:
        return self._collider.recallSweep(collider)


class PhysicalBox(Physical):
    _collider: AABB | None

    def __init__(self, bounds: AABB | None = None):
        super().__init__(bounds)

    @property
    def bounds(self):
        return self._collider

    @bounds.setter
    def bounds(self, newBounds: AABB):
        super()._set_collider(newBounds)

    @property
    def pos(self) -> glm.vec3:
        return self.bounds.pos

    @pos.setter
    def pos(self, newPos: glm.vec3):
        self.bounds.pos = newPos

    @property
    def size(self) -> glm.vec3:
        return self.bounds.size

    @size.setter
    def size(self, newSize: glm.vec3):
        self.bounds.size = newSize

    @property
    def half(self) -> glm.vec3:
        return self.bounds.half

    @property
    def min(self) -> glm.vec3:
        return self.bounds.min

    @property
    def max(self) -> glm.vec3:
        return self.bounds.max
