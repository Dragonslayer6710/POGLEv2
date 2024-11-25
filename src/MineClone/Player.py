import math
import random

import glm
import numpy as np

from MineClone.World import *
from POGLE.Physics.Collisions import Hit, Ray
from POGLE.Renderer.Camera import *
from POGLE.Core.Application import GetApplication

from POGLE.Geometry.Data import NMM

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
_PLAYER_CROUCH_SPEED: float = 0.7 * _PLAYER_WALK_SPEED

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
        app = GetApplication()
        if app is not None:
            self.camera: Camera = Camera(camPos.x, camPos.y, camPos.z, aspectRatio=app.get_window().get_aspect_ratio())
        else:
            self.camera: Camera = Camera(camPos.x, camPos.y, camPos.z)

        # movement vector/values
        self.acceleration: glm.vec3 = glm.vec3()
        self.velocity: glm.vec3 = glm.vec3()
        self.moveSpeed = _PLAYER_WALK_SPEED

        # States
        self.isFlying = False
        self.isGrounded = False
        self.isSprinting = False
        self.isCrouching = False

        # cooldowns
        self.attackCooldown = 0.0
        self.interactCooldown = 0.0

        self.firstPersonCamera: bool = True

        self.collidingBlockPositions = None

        self.targetBlock: Block = None
        self.targetFaceBlockSpace: Block = None
        self.checkTargetBlock = True
        self.reachRadius = _PLAYER_REACH_RADIUS
        self.blockReach = glm.vec2(4.0, 3.0)

    @property
    def playerModelMatrix(self) -> glm.mat4:
        return NMM(self.pos, s=_PLAYER_DIMENSIONS)

    @property
    def playerMesh(self):
        return
        return CubeMesh(self.playerModelMatrix, alpha=0.5)

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

    def handle_input(self):
        self.moveVector: glm.vec3 = glm.vec3()
        for ctrl in GetBoundControls():
            ctrlType = ctrl.GetType()
            if ctrlType == CTRL.Type.MOVEMENT:
                self.handle_movement_input(ctrl)
            elif ctrlType == CTRL.Type.ACTION:
                self.handle_action_input(ctrl)
        if 0 == self.moveVector.x:
            self.velocity.x = 0
        if self.isFlying:
            if 0 == self.moveVector.y:
                self.velocity.y = 0
        if 0 == self.moveVector.z:
            self.velocity.z = 0
        if np.sum(self.moveVector):
            self.acceleration += self.moveVector * self.moveSpeed

    def handle_movement_input(self, ctrl: Control):
        if not ctrl.GetInputState().value:
            return
        id = ctrl.GetID()
        if id.value < 4:
            direction = -1 if id.value % 2 else 1
            if id.value < 2:
                if self.isFlying:
                    vectorMod: glm.vec3 = self.camFront
                else:
                    vectorMod: glm.vec3 = glm.vec3(self.camFront.x, 0, self.camFront.z)
                    vectorMod = glm.normalize(vectorMod)  # Normalize after zeroing out the y component

            else:
                vectorMod: glm.vec3 = self.camera.Right
            self.moveVector += direction * vectorMod
        else:
            direction = -1 if id.value % 2 else 1
            if self.isFlying:
                vectorMod: glm.vec3 = self.camera.WorldUp
                self.moveVector += direction * vectorMod
            else:
                if direction == -1:
                    self.crouch()
                else:
                    self.jump()

    def attack(self):
        if not self.attackCooldown:
            self.attackCooldown = 10.0
            if self.targetBlock:
                self.targetBlock.set(Block.ID.Air)

    def interact(self):
        if not self.interactCooldown:
            self.interactCooldown = 10.0
            if self.targetFaceBlockSpace:
                if not self.targetFaceBlockSpace.is_solid:
                    if not self.targetFaceBlockSpace.bounds.intersectAABB(self.bounds):
                        self.targetFaceBlockSpace.set(Block.ID(random.randrange(1, len(Block.ID))))

    def handle_action_input(self, ctrl: Control):
        id = ctrl.GetID()

        if id == CTRL.ID.Action.SPRINT:
            if ctrl.GetInputState() == Input.State.RELEASE:
                if self.isSprinting:
                    self.isSprinting = False
                    self.moveSpeed = _PLAYER_CROUCH_SPEED if self.isCrouching else _PLAYER_WALK_SPEED
            elif ctrl.GetInputState() == Input.State.PRESS:
                if not self.isSprinting and not self.isCrouching:
                    self.isSprinting = True
                    self.moveSpeed = _PLAYER_SPRINT_SPEED  # Set to sprint speed instead of True
        elif ctrl.GetInputState().value:
            if id == CTRL.ID.Action.ATTACK:
                self.attack()

            elif id == CTRL.ID.Action.INTERACT:
                self.interact()

    def handle_collision(self):
        return
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
            if not block.is_solid:
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

    def update(self, deltaTime: float):
        if not self.isFlying:
            if not self.isGrounded:
                self.apply_gravity()
        self.handle_input()
        self.move(deltaTime)

        self.acquireTargetBlock()
        if self.attackCooldown > 0:
            self.attackCooldown -= 1.0
        if self.interactCooldown > 0:
            self.interactCooldown -= 1.0

    def acquireTargetBlock(self):
        if not self.checkTargetBlock:
            return

        ray: Ray = Ray.from_start_dir(self.eyePos, self.reachLineDelta)
        self.targetBlock = None
        self.targetFaceBlockSpace = None
        nearBest = np.inf
        farBest = np.inf
        nearPos: glm.vec3 = None
        return
        hitBlocks: set[Block] = self.world.query_segment_blocks(ray)

        for block in hitBlocks:
            hits: Tuple[Hit, Hit] = block.recallHit(ray)
            nearHit, farHit = hits
            if nearHit.time > self.reachRadius:
                continue

            if not block.is_solid:
                continue

            if nearHit.time > nearBest:
                continue

            if nearHit.time == nearBest:
                if farHit.time > farBest:
                    continue

            nearBest = min(nearBest, nearHit.time)
            farBest = min(farBest, farHit.time)

            nearPos = nearHit.pos
            self.targetBlock = block

        if self.targetBlock:
            space = self.targetBlock.get_adjblock_at_segment_intersect(nearPos)
            if space:
                if not space.is_solid:
                    self.targetFaceBlockSpace = space

    def draw(self):
        if not self.firstPersonCamera:
            self.playerMesh.draw()
        if self.targetBlock:
            tbwfCubeMesh = self.targetBlockWireframeCubeMesh
            if tbwfCubeMesh:
                tbwfCubeMesh.draw()
