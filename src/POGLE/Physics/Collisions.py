import glm

from POGLE.Core.Core import *

class Collider:
    pass


class Hit:
    def __init__(self, collider: Collider):
        self.collider: Collider = collider  # collider
        self.pos: glm.vec3 = glm.vec3()  # position of intersect
        self.normal: glm.vec3 = glm.vec3()  # surface normal at point of contact
        self.delta: glm.vec3 = glm.vec3()  # overlap between two objects, vector to correct
        self.time: float = 0.0  # fraction from 0 to 1 for ray / sweep indicating how far along line collision occured


class AABB:
    pos: glm.vec3
    half: glm.vec3
    size: glm.vec3
    min: glm.vec3
    max: glm.vec3

class Frustum:
    class Intersection:
        INSIDE = 0
        OUTSIDE = 1
        INTERSECT = 2

    @staticmethod
    def get_frustum_planes(view_matrix: glm.mat4, proj_matrix: glm.mat4) -> List[glm.vec4]:
        planes = []
        m = glm.value_ptr(proj_matrix * view_matrix)

        # Extract frustum planes from the matrix
        planes.append(glm.vec4(m[3] + m[0], m[7] + m[4], m[11] + m[8], m[15] + m[12]))  # Left
        planes.append(glm.vec4(m[3] - m[0], m[7] - m[4], m[11] - m[8], m[15] - m[12]))  # Right
        planes.append(glm.vec4(m[3] - m[1], m[7] - m[5], m[11] - m[9], m[15] - m[13]))  # Top
        planes.append(glm.vec4(m[3] + m[1], m[7] + m[5], m[11] + m[9], m[15] + m[13]))  # Bottom
        planes.append(glm.vec4(m[3] + m[2], m[7] + m[6], m[11] + m[10], m[15] + m[14]))  # Near
        planes.append(glm.vec4(m[3] - m[2], m[7] - m[6], m[11] - m[10], m[15] - m[14]))  # Far

        return [glm.normalize(p) for p in planes]

    def __init__(self, view_matrix: glm.mat4, proj_matrix: glm.mat4):
        self.planes = Frustum.get_frustum_planes(view_matrix, proj_matrix)



class Collider:
    hitRecall: dict[Collider, dict[Collider | glm.vec3, Hit]] = {}

    def __init__(self):
        Collider.hitRecall[self] = {}
        self.hitRecall: dict[Collider | glm.vec3, Hit] = Collider.hitRecall[self]

    def recallHit(self, collider: Collider | glm.vec3) -> Hit | None:
        if self.hitRecall.get(collider):
            return self.hitRecall.pop(collider)

class Ray(Collider):
    __create_key = object()

    @classmethod
    def _new(cls, origin: glm.vec3, size: glm.vec3):
        return Ray(cls.__create_key, origin, size)

    def __init__(self, create_key, origin: glm.vec3, dir: glm.vec3):
        assert (create_key == Ray.__create_key), \
            "Ray objects must be created using Ray._new"
        super().__init__()
        self.start: glm.vec3 = origin
        self.dir: glm.vec3 = dir
        self.invDir: glm.vec3 = 1.0 / dir
        self.sign: glm.vec3 = glm.vec3([1 if i < 0 else 0 for i in self.invDir])
        self.normal: glm.vec3 = glm.normalize(dir)
        self.end: glm.vec3 = origin + dir

    @classmethod
    def from_start_end(cls, origin: glm.vec3, end: glm.vec3):
        return cls._new(origin, end - origin)

    @classmethod
    def from_start_dir(cls, origin: glm.vec3, dir: glm.vec3 = glm.vec3(1.0)):
        return cls._new(origin, dir)

    def __str__(self):
        return f"Ray(origin: {self.start}, dir: {self.dir}, normal: {self.normal}, end: {self.end})"

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
        self._bounds: list[glm.vec3] = [self.min, self.max]
        self._corners = [
            self.min,
            glm.vec3(self.max.x, self.min.y, self.min.z),
            glm.vec3(self.min.x, self.max.y, self.min.z),
            glm.vec3(self.max.x, self.max.y, self.min.z),
            glm.vec3(self.min.x, self.min.y, self.max.z),
            glm.vec3(self.max.x, self.min.y, self.max.z),
            glm.vec3(self.min.x, self.max.y, self.max.z),
            self.max
        ]


    @classmethod
    def from_min_max(cls, min: glm.vec3, max: glm.vec3):
        size: glm.vec3 = max - min
        pos: glm.vec3 = min + size / 2
        return cls._new(pos, size)

    @classmethod
    def from_pos_size(cls, pos: glm.vec3, size: glm.vec3 = glm.vec3(1.0)):
        return cls._new(pos, size)

    @property
    def half(self) -> glm.vec3:
        return self.size / 2

    @property
    def min(self) -> glm.vec3:
        return self.pos - self.half

    @property
    def max(self) -> glm.vec3:
        return self.pos + self.half

    def does_overlap(self, overlap: glm.vec3) -> bool:
        return overlap.x > 0 and overlap.y > 0 and overlap.z > 0

    # Example of the intersectPoint method with comments
    def intersectPoint(self, point: glm.vec3) -> Hit:
        # Calculate the vector difference between the centers of the AABB and the point
        delta = point - self.pos

        # Determine the sign of the difference for each axis (direction of the collision)
        sign = glm.sign(delta)

        # Calculate the overlap on each axis by subtracting the absolute distance
        # from the half-extents of the AABB
        overlap = self.half - glm.abs(delta)

        # Check if there is an intersection by verifying if all components of `overlap` are positive
        if self.does_overlap(overlap):
            hit = Hit(self)

            if abs(overlap.x - overlap.y) < EPSILON and abs(overlap.x - overlap.z) < EPSILON:
                # If all overlaps are nearly the same, default to the primary axis (e.g., x-axis)
                hit.delta.x = overlap.x * sign.x
                hit.normal.x = sign.x
                hit.pos = glm.vec3(self.pos.x + self.half.x * sign.x, point.y, point.z)
            elif abs(overlap.x - overlap.y) < EPSILON:
                # If x and y overlaps are similar, choose based on some rule (e.g., x-axis priority)
                if overlap.x < overlap.z:
                    hit.delta.x = overlap.x * sign.x
                    hit.normal.x = sign.x
                    hit.pos = glm.vec3(self.pos.x + self.half.x * sign.x, point.y, point.z)
                else:
                    hit.delta.z = overlap.z * sign.z
                    hit.normal.z = sign.z
                    hit.pos = glm.vec3(point.x, point.y, self.pos.z + self.half.z * sign.z)
            elif abs(overlap.y - overlap.z) < EPSILON:
                # If y and z overlaps are similar, choose based on some rule (e.g., y-axis priority)
                if overlap.y < overlap.x:
                    hit.delta.y = overlap.y * sign.y
                    hit.normal.y = sign.y
                    hit.pos = glm.vec3(point.x, self.pos.y + self.half.y * sign.y, point.z)
                else:
                    hit.delta.z = overlap.z * sign.z
                    hit.normal.z = sign.z
                    hit.pos = glm.vec3(point.x, point.y, self.pos.z + self.half.z * sign.z)
            else:
                # General case: choose the axis with the smallest overlap
                if overlap.x < overlap.y and overlap.x < overlap.z:
                    hit.delta.x = overlap.x * sign.x
                    hit.normal.x = sign.x
                    hit.pos = glm.vec3(self.pos.x + self.half.x * sign.x, point.y, point.z)
                elif overlap.y < overlap.z:
                    hit.delta.y = overlap.y * sign.y
                    hit.normal.y = sign.y
                    hit.pos = glm.vec3(point.x, self.pos.y + self.half.y * sign.y, point.z)
                else:
                    hit.delta.z = overlap.z * sign.z
                    hit.normal.z = sign.z
                    hit.pos = glm.vec3(point.x, point.y, self.pos.z + self.half.z * sign.z)

            # Store the result in the recall dictionary
            self.hitRecall[point] = hit
            return hit

        # Return None if no intersection
        return None

    def intersectSegment_old(self, ray: Ray) -> list[Hit] | None:
        # X Axis
        tMin = (self._bounds[int(ray.sign.x)].x - ray.start.x) * ray.invDir.x
        tMax = (self._bounds[int(1 - ray.sign.x)].x - ray.start.x) * ray.invDir.x

        # Y Axis
        min = (self._bounds[int(ray.sign.y)].y - ray.start.y) * ray.invDir.y
        max = (self._bounds[int(1 - ray.sign.y)].y - ray.start.y) * ray.invDir.y

        if max < tMin or min > tMax:
            return None
        if min > tMin:
            tMin = min
        if max < tMax:
            tMax = max

        # Z Axis
        min = (self._bounds[int(ray.sign.z)].z - ray.start.z) * ray.invDir.z
        max = (self._bounds[int(1 - ray.sign.z)].z - ray.start.z) * ray.invDir.z

        if max < tMin or min > tMax:
            return None
        if min > tMin:
            tMin = min
        if max < tMax:
            tMax = max

        # No intersection if the ray is entirely outside the AABB
        tMinMoreThanOne = tMin >= 1
        tMaxLessThanZero = tMax <= 0
        tMinGreaterThanMax = tMin > tMax
        if tMinMoreThanOne or tMaxLessThanZero or tMinGreaterThanMax:
            return None
        # Calculate the hit points
        nearHit = Hit(self)
        nearHit.time = clamp(tMin, 0, 1)

        nearHit.delta = (1.0 - nearHit.time) * -ray.dir
        nearHit.pos = ray.start + nearHit.time * ray.dir

        farHit = Hit(self)
        farHit.time = clamp(tMax, 0, 1)

        farHit.delta = (1.0 - farHit.time) * -ray.dir
        farHit.pos = ray.start + farHit.time * ray.dir

        # Store and return the hits
        self.hitRecall[ray] = [nearHit, farHit]
        return [nearHit, farHit]

    def intersectSegment(self, ray: Ray) -> list[Hit] | None:
        # Compute tMin and tMax for X, Y, and Z axes
        tMin_x = (self._bounds[int(ray.sign.x)].x - ray.start.x) * ray.invDir.x
        tMax_x = (self._bounds[1 - int(ray.sign.x)].x - ray.start.x) * ray.invDir.x

        tMin_y = (self._bounds[int(ray.sign.y)].y - ray.start.y) * ray.invDir.y
        tMax_y = (self._bounds[1 - int(ray.sign.y)].y - ray.start.y) * ray.invDir.y

        tMin_z = (self._bounds[int(ray.sign.z)].z - ray.start.z) * ray.invDir.z
        tMax_z = (self._bounds[1 - int(ray.sign.z)].z - ray.start.z) * ray.invDir.z

        # Compute overlap intervals
        tMin = max(min(tMin_x, tMax_x), min(tMin_y, tMax_y), min(tMin_z, tMax_z))
        tMax = min(max(tMin_x, tMax_x), max(tMin_y, tMax_y), max(tMin_z, tMax_z))

        # Early exit if there is no intersection
        if tMin > tMax or tMax <= 0 or tMin >= 1:
            return None

        # Calculate the hit points
        nearHit = Hit(self)
        nearHit.time = clamp(tMin, 0, 1)
        nearHit.delta = (1.0 - nearHit.time) * -ray.dir
        nearHit.pos = ray.start + nearHit.time * ray.dir

        farHit = Hit(self)
        farHit.time = clamp(tMax, 0, 1)
        farHit.delta = (1.0 - farHit.time) * -ray.dir
        farHit.pos = ray.start + farHit.time * ray.dir

        # Store and return the hits
        self.hitRecall[ray] = [nearHit, farHit]
        return [nearHit, farHit]

    def intersectAABB_old(self, box: AABB) -> Hit:
        delta = box.pos - self.pos
        sign = glm.sign(delta)
        overlap = (box.half + self.half) - glm.abs(delta)

        if self.does_overlap(overlap):
            hit = Hit(self)

            if abs(overlap.x - overlap.y) < EPSILON and abs(overlap.x - overlap.z) < EPSILON:
                # If all overlaps are nearly the same, default to a primary axis (e.g., x-axis)
                hit.delta.x = overlap.x * sign.x
                hit.normal.x = sign.x
                hit.pos = glm.vec3(self.pos.x + self.half.x * sign.x, box.pos.y, box.pos.z)
            elif abs(overlap.x - overlap.y) < EPSILON:
                # If x and y overlaps are similar, choose based on some rule (e.g., x-axis priority)
                if overlap.x < overlap.z:
                    hit.delta.x = overlap.x * sign.x
                    hit.normal.x = sign.x
                    hit.pos = glm.vec3(self.pos.x + self.half.x * sign.x, box.pos.y, box.pos.z)
                else:
                    hit.delta.z = overlap.z * sign.z
                    hit.normal.z = sign.z
                    hit.pos = glm.vec3(box.pos.x, box.pos.y, self.pos.z + self.half.z * sign.z)
            elif abs(overlap.y - overlap.z) < EPSILON:
                # If y and z overlaps are similar, choose based on some rule (e.g., y-axis priority)
                if overlap.y < overlap.x:
                    hit.delta.y = overlap.y * sign.y
                    hit.normal.y = sign.y
                    hit.pos = glm.vec3(box.pos.x, self.pos.y + self.half.y * sign.y, box.pos.z)
                else:
                    hit.delta.z = overlap.z * sign.z
                    hit.normal.z = sign.z
                    hit.pos = glm.vec3(box.pos.x, box.pos.y, self.pos.z + self.half.z * sign.z)
            else:
                # General case: choose the axis with the smallest overlap
                if overlap.x < overlap.y and overlap.x < overlap.z:
                    hit.delta.x = overlap.x * sign.x
                    hit.normal.x = sign.x
                    hit.pos = glm.vec3(self.pos.x + self.half.x * sign.x, box.pos.y, box.pos.z)
                elif overlap.y < overlap.z:
                    hit.delta.y = overlap.y * sign.y
                    hit.normal.y = sign.y
                    hit.pos = glm.vec3(box.pos.x, self.pos.y + self.half.y * sign.y, box.pos.z)
                else:
                    hit.delta.z = overlap.z * sign.z
                    hit.normal.z = sign.z
                    hit.pos = glm.vec3(box.pos.x, box.pos.y, self.pos.z + self.half.z * sign.z)

            # Store the result in the recall dictionary
            self.hitRecall[box] = hit
            return hit

        return None

    def intersectAABB(self, box: AABB) -> Hit:
        # Compute deltas
        delta = box.pos - self.pos

        # Compute overlaps
        overlap_x = (box.half.x + self.half.x) - abs(delta.x)
        overlap_y = (box.half.y + self.half.y) - abs(delta.y)
        overlap_z = (box.half.z + self.half.z) - abs(delta.z)

        # Early exit if there's no overlap
        if overlap_x <= 0 or overlap_y <= 0 or overlap_z <= 0:
            return None

        # Determine the axis of minimum overlap directly
        min_overlap = min(overlap_x, overlap_y, overlap_z)

        if min_overlap == overlap_x:
            sign_x = 1 if delta.x >= 0 else -1
            hit = Hit(self)
            hit.delta.x = overlap_x * sign_x
            hit.normal.x = sign_x
            hit.pos = glm.vec3(self.pos.x + self.half.x * sign_x, box.pos.y, box.pos.z)
        elif min_overlap == overlap_y:
            sign_y = 1 if delta.y >= 0 else -1
            hit = Hit(self)
            hit.delta.y = overlap_y * sign_y
            hit.normal.y = sign_y
            hit.pos = glm.vec3(box.pos.x, self.pos.y + self.half.y * sign_y, box.pos.z)
        else:
            sign_z = 1 if delta.z >= 0 else -1
            hit = Hit(self)
            hit.delta.z = overlap_z * sign_z
            hit.normal.z = sign_z
            hit.pos = glm.vec3(box.pos.x, box.pos.y, self.pos.z + self.half.z * sign_z)

        # Store the result in the recall dictionary
        self.hitRecall[box] = hit
        return hit

    def intersectFrustum(self, frustum: Frustum) -> bool:
        for plane in frustum.planes:
            # Calculate the distance from the center of the AABB to the plane
            d = glm.dot(glm.vec3(plane.x, plane.y, plane.z), self.pos) + plane.w
            # Compute the maximum distance from the AABB center to the plane normals
            r = self.half.x * abs(plane.x) + self.half.y * abs(plane.y) + self.half.z * abs(plane.z)
            if d < -r:
                return False  # Outside the frustum

        return True  # Inside or intersects the frustum

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


def test_aabb_frustum_intersection():
    # Define view and projection matrices for the frustum (identity matrices for simplicity)
    view_matrix = glm.mat4(1.0)
    proj_matrix = glm.mat4(1.0)

    # Create a frustum
    frustum = Frustum(view_matrix, proj_matrix)

    # Define some AABBs
    aabb1 = AABB.from_pos_size(glm.vec3(0.0, 0.0, 0.0), glm.vec3(1.0, 1.0, 1.0))
    aabb2 = AABB.from_pos_size(glm.vec3(5.0, 5.0, 5.0), glm.vec3(1.0, 1.0, 1.0))
    aabb3 = AABB.from_pos_size(glm.vec3(-2.0, -2.0, -2.0), glm.vec3(1.0, 1.0, 1.0))

    # Check if the AABBs intersect with the frustum
    print(f"AABB1 intersects with frustum: {aabb1.intersectFrustum(frustum)}")  # Expect True or False
    print(f"AABB2 intersects with frustum: {aabb2.intersectFrustum(frustum)}")  # Expect True or False
    print(f"AABB3 intersects with frustum: {aabb3.intersectFrustum(frustum)}")  # Expect True or False


# Run the test
test_aabb_frustum_intersection()

#quit()
