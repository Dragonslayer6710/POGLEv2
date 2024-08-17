from MineClone.World import *
from POGLE.Renderer.Camera import *

_PLAYER_DIMENSIONS: glm.vec3 = glm.vec3(0.8, 1.8, 0.8)
_PLAYER_HALF_DIMENSIONS = _PLAYER_DIMENSIONS / 2
_PLAYER_CAMERA_HEIGHT_OFFSET_FROM_FEET = 1.62
_PLAYER_VERTICAL_OFFSET_FEET_TO_CAMERA = glm.vec3(0, _PLAYER_CAMERA_HEIGHT_OFFSET_FROM_FEET - _PLAYER_HALF_DIMENSIONS.y,
                                                  0)
_PLAYER_CAMERA_DISTANCE_FROM_HEAD = 0.3
# Assuming looking in negative z direction initially
_PLAYER_HORIZONTAL_OFFSET_HEAD_TO_CAMERA_INITAL = _PLAYER_CAMERA_DISTANCE_FROM_HEAD * CAMERA_INITIAL_FRONT
_PLAYER_CAMERA_INITIAL_OFFSET: glm.vec3 = _PLAYER_VERTICAL_OFFSET_FEET_TO_CAMERA + _PLAYER_HORIZONTAL_OFFSET_HEAD_TO_CAMERA_INITAL

_PLAYER_OFFSET_FEET_TO_CENTRE: glm.vec3 = glm.vec3(0, _PLAYER_HALF_DIMENSIONS.y, 0)
_Move = Control.ID.Move


class Player(PhysicalBox):
    def __init__(self, feetPos: glm.vec3):
        self.bounds = AABB.from_pos_size(feetPos + _PLAYER_OFFSET_FEET_TO_CENTRE, _PLAYER_DIMENSIONS)
        camPos: glm.vec3 = self.pos + _PLAYER_CAMERA_INITIAL_OFFSET
        self.camera: Camera = Camera(camPos.x, camPos.y, camPos.z)

    @property
    def feetPos(self) -> glm.vec3:
        return self.pos - _PLAYER_OFFSET_FEET_TO_CENTRE

    @property
    def camOffset(self) -> glm.vec3:
        return self.camera.Front * _PLAYER_CAMERA_DISTANCE_FROM_HEAD + _PLAYER_VERTICAL_OFFSET_FEET_TO_CAMERA

    @property
    def camPos(self) -> glm.vec3:
        return self.pos + self.camOffset

    @property
    def MovementSpeed(self) -> float:
        return self.camera.MovementSpeed

    def move(self, direction: Control.ID.Move, deltaTime: float) -> None:
        velocity = self.MovementSpeed * deltaTime
        movement: glm.vec3 = None
        if direction == _Move.FORWARD:
            movement = self.camera.Front * velocity
        elif direction == _Move.BACKWARD:
            movement = -self.camera.Front * velocity
        elif direction == _Move.LEFT:
            movement = -self.camera.Right * velocity
        elif direction == _Move.RIGHT:
            movement = self.camera.Right * velocity
        elif direction == _Move.UP:
            movement = self.camera.WorldUp * velocity
        elif direction == _Move.DOWN:
            movement = -self.camera.WorldUp * velocity
        self.pos += movement
        self.camera.Position += movement
