import math

import glm
import numpy as np

from MineClone.World import *
from POGLE.Renderer.Camera import *

_PLAYER_DIMENSIONS: glm.vec3 = glm.vec3(0.8, 1.8, 0.8)
_PLAYER_HALF_DIMENSIONS = _PLAYER_DIMENSIONS / 2
_PLAYER_CAMERA_HEIGHT_OFFSET_FROM_FEET = 1.62
_PLAYER_VERTICAL_OFFSET_FEET_TO_CAMERA = glm.vec3(0, _PLAYER_CAMERA_HEIGHT_OFFSET_FROM_FEET - _PLAYER_HALF_DIMENSIONS.y,
                                                  0)
_PLAYER_CAMERA_INITIAL_OFFSET: glm.vec3 = _PLAYER_VERTICAL_OFFSET_FEET_TO_CAMERA

_PLAYER_OFFSET_FEET_TO_CENTRE: glm.vec3 = glm.vec3(0, _PLAYER_HALF_DIMENSIONS.y, 0)
_Move = Control.ID.Move

_DAMPING_FACTOR: float = 0.99

_PLAYER_WALK_SPEED: float = 2.0
_PLAYER_SPRINT_SPEED: float = 1.5 * _PLAYER_WALK_SPEED
_PLAYER_CROUSH_SPEED: float = 0.7 * _PLAYER_WALK_SPEED

_GRAVITY: float = -9.8
_2X_FABS_GRAVITY: float = math.fabs(_GRAVITY)
_TERMINAL_VELOCITY: float = -math.sqrt(_2X_FABS_GRAVITY)

_JUMP_HEIGHT: float = 2
_JUMP_FORCE: float = math.sqrt(_2X_FABS_GRAVITY * _JUMP_HEIGHT)

_PLAYER_REACH_RADIUS: float = 4.5

class Player(PhysicalBox):

    def __init__(self, world: World, feetPos: glm.vec3):
        self.world: World = world

        self.bounds = AABB.from_pos_size(feetPos + _PLAYER_OFFSET_FEET_TO_CENTRE, _PLAYER_DIMENSIONS)
        camPos: glm.vec3 = self.pos + _PLAYER_CAMERA_INITIAL_OFFSET
        self.camera: Camera = Camera(camPos.x, camPos.y, camPos.z)

        # movement vector/values
        self.acceleration: glm.vec3 = glm.vec3()
        self.velocity: glm.vec3 = glm.vec3()
        self.moveSpeed = _PLAYER_WALK_SPEED

        # States
        self.isFlying = False
        self.isGrounded = False

        self.boundMoveControls: list[Control] = []
        for boundCtrl in GetBoundControls():
            if boundCtrl.GetType() == CTRL.Type.MOVEMENT:
                self.boundMoveControls.append(boundCtrl)

        self.firstPersonCamera: bool = True

        self.collidingBlockPositions = None

        self.targetBlock: Block = None
        self.checkTargetBlock = True
        self.reachRadius = _PLAYER_REACH_RADIUS
        self.blockReach = glm.vec2(4.0, 3.0)

    @property
    def playerModelMatrix(self) -> glm.mat4:
        return NMM(self.pos, s=_PLAYER_DIMENSIONS)

    @property
    def playerMesh(self) -> CubeMesh:
        return CubeMesh(self.playerModelMatrix, alpha=0.5)

    @property
    def collidingBlockWireCubesMesh(self) -> WireframeCubeMesh:
        return WireframeCubeMesh(self.collidingBlockPositions)

    @property
    def targetBlockWireframeCubeMesh(self) -> WireframeCubeMesh:
        return self.targetBlock.get_wireframe_cube_mesh()

    @property
    def feetPos(self) -> glm.vec3:
        return self.pos - _PLAYER_OFFSET_FEET_TO_CENTRE

    @ property
    def camFront(self) -> glm.vec3:
        return self.camera.Front
    @property
    def eyePos(self) -> glm.vec3:
        return self.pos + _PLAYER_VERTICAL_OFFSET_FEET_TO_CAMERA

    @property
    def thirdPersonCamPos(self) -> glm.vec3:
        return self.eyePos - 5 * glm.normalize(self.camFront)

    @property
    def camPos(self) -> glm.vec3:
        if self.firstPersonCamera:
            return self.eyePos
        else:
            return self.thirdPersonCamPos

    @property
    def reachLineDelta(self) -> glm.vec3:
        return self.camFront * self.reachRadius


    def apply_gravity(self):
        self.acceleration.y += _GRAVITY

    def jump(self):
        if not self.isGrounded:
            return
        self.velocity.y = _JUMP_FORCE
        self.isGrounded = False

    def crouch(self):
        pass

    def handle_movement_input(self) -> None:
        moveVector: glm.vec3 = glm.vec3()
        for ctrl in self.boundMoveControls:
            if ctrl.GetInputState().value:
                id = ctrl.GetID()
                if id.value < 4:
                    direction = -1 if id.value % 2 else 1
                    if id.value < 2:
                        if self.isFlying:
                            vectorMod: glm.vec3 = self.camFront
                        else:
                            vectorMod: glm.vec3 = glm.normalize(self.camFront)
                            vectorMod.y = 0
                    else:
                        vectorMod: glm.vec3 = self.camera.Right
                    moveVector += direction * vectorMod
                else:
                    direction = -1 if id.value % 2 else 1
                    if self.isFlying:
                        vectorMod: glm.vec3 = self.camera.WorldUp
                        moveVector += direction * vectorMod
                    else:
                        if direction == -1:
                            self.crouch()
                        else:
                            self.jump()

        if 0 == moveVector.x:
            self.velocity.x = 0
        if self.isFlying:
            if 0 == moveVector.y:
                self.velocity.y = 0
        if 0 == moveVector.z:
            self.velocity.z = 0
        if np.sum(moveVector):
            self.acceleration += moveVector * self.moveSpeed

    def handle_collision(self):
        # Get colliding blocks within the current bounds
        collidingBlocks: set[Block] = self.world.query_aabb_blocks(self.bounds)
        if len(collidingBlocks):
            self.collidingBlockPositions = [block.pos for block in collidingBlocks]
        elif self.collidingBlockPositions:
            self.collidingBlockPositions = None
        grounded: bool = False  # Assume not grounded initially

        # Initialize the correction vector and collision array
        correctionVector: glm.vec3 = glm.vec3(0.0, 0.0, 0.0)
        collisions = [0.0 for _ in range(6)]  # Stores corrections in [+, -, +, -, +, -] for [x, y, z]

        # Iterate through each block to check for collisions
        for block in collidingBlocks:
            if not block.is_block:
                continue

            # Retrieve the hit information from the block
            hit: Hit = block.recallHit(self.bounds)
            correction: glm.vec3 = hit.delta

            # Accumulate the correction vector
            correctionVector += correction

            # Track the collision magnitude for each axis
            if correction.x:
                if correction.x < 0.0:
                    collisions[1] = min(collisions[1], correction.x)
                else:
                    collisions[0] = max(collisions[0], correction.x)
            if correction.y:
                if correction.y < 0.0:
                    collisions[3] = min(collisions[3], correction.y)
                else:
                    collisions[2] = max(collisions[2], correction.y)
            if correction.z:
                if correction.z < 0.0:
                    collisions[5] = min(collisions[5], correction.z)
                else:
                    collisions[4] = max(collisions[4], correction.z)

            # Handle Y-axis collisions for grounded state and vertical motion stopping
            if correction.y:
                if correction.y > 0.0:
                    if grounded:
                        correctionVector.y -= correction.y  # Already grounded, remove this correction
                        continue
                    grounded = True  # Set grounded if moving upward and colliding
                elif correction.y < 0.0:
                    if self.velocity.y <= 0.0:
                        correctionVector.y -= correction.y  # Only correct downward movement
                self.velocity.y = 0.0  # Stop vertical velocity when colliding vertically
        if not grounded:
            # Apply horizontal (X and Z) corrections
            if collisions[0] or collisions[1]:
                self.velocity.x = 0.0
            if collisions[4] or collisions[5]:
                self.velocity.z = 0.0

        # Apply the accumulated correction vector to the object's movement
        self.applyMovement(correctionVector)

        # Update the grounded state
        self.isGrounded = grounded

        # Ensure downward velocity is reset if grounded
        if self.isGrounded and self.velocity.y < 0.0:
            self.velocity.y = 0.0

    def applyMovement(self, movement: glm.vec3):
        self.pos += movement
        if not self.firstPersonCamera:
            self.camera.Position = self.camPos
        else:
            self.camera.Position += movement

    def move(self, deltaTime: float):
        self.velocity += self.acceleration * deltaTime

        # cap horizontal speed
        horizSpeed: float = glm.length(glm.vec2(self.velocity.x,self.velocity.z))
        if horizSpeed != 0.0:
            horizScale: float = self.moveSpeed / horizSpeed
            if horizSpeed > self.moveSpeed:
                self.velocity *= glm.vec3(horizScale, 1, horizScale)

            if self.isFlying:
                if self.velocity.y > self.moveSpeed:
                    self.velocity.y *= horizScale

        movement: glm.vec3 = self.velocity * deltaTime
        self.applyMovement(movement)
        self.handle_collision()

        self.acceleration = glm.vec3()

    cnt = 2
    def update(self, deltaTime: float):
        if self.cnt:
            self.cnt -= 1
            return
        if not self.isFlying:
            if not self.isGrounded:
                self.apply_gravity()
        self.handle_movement_input()
        self.move(deltaTime)

        self.acquireTargetBlock()

    def acquireTargetBlock(self):
        if not self.checkTargetBlock:
            return

        ray: Ray = Ray.from_start_dir(self.eyePos, self.reachLineDelta)
        self.targetBlock = None
        nearBest = np.inf
        farBest = np.inf

        hitBlocks: set[Block] = self.world.query_segment_blocks(ray)
        for block in hitBlocks:
            hits: Tuple[Hit, Hit] = block.recallHit(ray)
            nearHit, farHit = hits
            if nearHit.time > self.reachRadius:
                continue

            if not block.is_block:
                continue

            if nearHit.time > nearBest:
                continue

            if nearHit.time == nearBest:
                if farHit.time > farBest:
                    continue

            #if abs(ray.dir.y) > self.blockReach[1]:
            #    continue

            #if glm.length(glm.vec2(ray.dir.xz)) > self.blockReach[0]:
            #    continue
            nearBest = min(nearBest, nearHit.time)
            farBest = min(farBest, farHit.time)

            self.targetBlock = block
        # TODO: target block face calculation



    def draw(self, projection: glm.mat4, view: glm.mat4):
        if not self.firstPersonCamera:
            self.playerMesh.draw(projection, view)
        #if self.collidingBlockPositions:
        #    self.collidingBlockWireCubesMesh.draw(projection, view)
        if self.targetBlock:
            self.targetBlockWireframeCubeMesh.draw(projection, view)
