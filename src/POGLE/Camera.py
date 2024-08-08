from POGLE.Core.Core import *

# Simple abstraction for cam movements
class Camera_Movement(Enum):
    FORWARD = 1
    BACKWARD = 2
    LEFT = 3
    RIGHT = 4
    UP = 5
    DOWN = 6

# Default camera values
YAW         = -90.0
PITCH       =  0.0
SPEED       =  2.5
SENSITIVITY =  0.1
ZOOM        =  45.0

class Camera:
    def __init__(self, posX = 0.0, posY = 0.0, posZ = 0.0, upX = 0.0, upY = 1.0, upZ = 0.0, yaw = YAW, pitch = PITCH):
        self.Position = glm.vec3(posX, posY, posZ)
        self.WorldUp = glm.vec3(upX, upY, upZ)
        self.Yaw = yaw
        self.Pitch = pitch

        self.Front = glm.vec3(0.0, 0.0, -1.0)
        self.Up = glm.vec3()
        self.Right = glm.vec3()
        self.MovementSpeed = SPEED
        self.MouseSensitivity = SENSITIVITY
        self.Zoom = ZOOM

        self.UpdateCameraVectors()

    def GetViewMatrix(self) -> glm.mat4:
        return glm.lookAt(self.Position, self.Position + self.Front, self.Up)

    def ProcessKeyboard(self, direction: Camera_Movement, deltaTime: float) -> None:
        velocity = self.MovementSpeed * deltaTime
        if direction == Camera_Movement.FORWARD:
            self.Position += self.Front * velocity
        elif direction == Camera_Movement.BACKWARD:
            self.Position -= self.Front * velocity
        elif direction == Camera_Movement.LEFT:
            self.Position -= self.Right * velocity
        elif direction == Camera_Movement.RIGHT:
            self.Position += self.Right * velocity
        elif direction == Camera_Movement.UP:
            self.Position += self.WorldUp * velocity
        elif direction == Camera_Movement.DOWN:
            self.Position -= self.WorldUp * velocity
    def ProcessMouseMovement(self, xoffset: float, yoffset: float, constrainPitch: bool = True) -> None:
        xoffset *= self.MouseSensitivity
        yoffset *= self.MouseSensitivity

        self.Yaw += xoffset
        self.Pitch += yoffset

        # make sure that when pitch is out of bounds, screen doesn't get flipped
        if constrainPitch:
            if self.Pitch > 89.0:
                self.Pitch = 89.0
            elif self.Pitch < -89.0:
                self.Pitch = -89.0

        # update Front Right and Up Vectors using the updated euler angles
        self.UpdateCameraVectors()

    def ProcessMouseScroll(self, yoffset: float) -> None:
        self.Zoom -= yoffset
        if self.Zoom < 1.0:
            self.Zoom = 1.0
        elif self.Zoom > 45.0:
            self.Zoom = 45.0

    def UpdateCameraVectors(self):
        # Calculate new Front vector
        front = glm.vec3(
            glm.cos(glm.radians(self.Yaw)) * glm.cos(glm.radians(self.Pitch)),
            glm.sin(glm.radians(self.Pitch)),
            glm.sin(glm.radians(self.Yaw)) * glm.cos(glm.radians(self.Pitch))
        )
        self.Front = glm.normalize(front)
        # also re-calculate the Right and Up vector
        self.Right = glm.normalize(glm.cross(self.Front, self.WorldUp)) # normalize the vectors, because their length gets closer to 0 the more you look up which results in slower movement.
        self.Up    = glm.normalize((glm.cross(self.Right, self.Front)))