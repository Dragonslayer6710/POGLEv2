import math

import glm

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

        self.firstPersonCamera: bool = False

    @property
    def playerModelMatrix(self) -> glm.mat4:
        return NMM(self.pos, s=_PLAYER_DIMENSIONS)

    @property
    def playerMesh(self) -> CubeMesh:
        return CubeMesh(self.playerModelMatrix, alpha=0.5)
    @property
    def feetPos(self) -> glm.vec3:
        return self.pos - _PLAYER_OFFSET_FEET_TO_CENTRE

    @property
    def camOffset(self) -> glm.vec3:
        offset: glm.vec3 = self.camera.Front + _PLAYER_VERTICAL_OFFSET_FEET_TO_CAMERA
        if self.firstPersonCamera:
            return offset
        else:
            thirdPersonOffset: glm.vec3 =  -4 * glm.normalize(self.camera.Front)
            return offset + thirdPersonOffset


    @property
    def camPos(self) -> glm.vec3:
        return self.pos + self.camOffset

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
                            vectorMod: glm.vec3 = self.camera.Front
                        else:
                            vectorMod: glm.vec3 = glm.normalize(self.camera.Front)
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
        # get colliding blocks
        collidingBlocks: set[Block] = self.world.query_aabb_blocks(self.bounds)
        grounded: bool = False # assume not grounded initially

        correctionVector: glm.vec3 = glm.vec3()
        collisions = [0 for i in range(6)]
        for block in collidingBlocks:
            if not block.is_block:
                continue

            hit: Hit = block.recallHit(self.bounds)
            correction: glm.vec3 = hit.delta

            correctionVector += correction

            if correction.x:
                if correction.x < 0.0:
                    collisions[1] = correction.x
                else:
                    collisions[0] = correction.x
            if correction.y:
                if correction.y < 0.0:
                    collisions[3] = correction.y
                else:
                    collisions[2] = correction.y
            if correction.z:
                if correction.z < 0.0:
                    collisions[5] = correction.z
                else:
                    collisions[4] = correction.z

            if correction.y:
                if correction.y > 0:
                    if grounded:
                        correctionVector.y -= correction.y
                        continue
                    grounded = True
                elif correction.y < 0:
                    if self.velocity.y <= 0.0:
                        correctionVector.y -= correction.y
                self.velocity.y = 0.0
            if correctionVector.x:
                self.velocity.x = 0.0
            if correctionVector.z:
                self.velocity.z = 0.0

        self.applyMovement(correctionVector)

        self.isGrounded = grounded

        if self.isGrounded and self.velocity.y < 0.0:
            self.velocity = 0.0
    def applyMovement(self, movement: glm.vec3):
        self.pos += movement
        if not self.firstPersonCamera:
            self.camera.Position = self.camPos
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

    def draw(self, projection: glm.mat4, view: glm.mat4):
        if not self.firstPersonCamera:
            self.playerMesh.draw(projection, view)