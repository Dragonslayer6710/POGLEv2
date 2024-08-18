from POGLE.Core.Core import *
from POGLE.Input.Control import *
# Simple abstraction for cam movements

# Default camera values
YAW         = -90.0
PITCH       =  0.0
SPEED       =  2.5
SENSITIVITY =  0.1
FOV        =  45.0

_Move = Control.ID.Move

CAMERA_INITIAL_FRONT = glm.vec3(0.0, 0.0, -1.0)
class Camera:
    def __init__(self, posX = 0.0, posY = 0.0, posZ = 0.0, upX = 0.0, upY = 1.0, upZ = 0.0, yaw = YAW, pitch = PITCH):
        self.Position = glm.vec3(posX, posY, posZ)
        self.WorldUp = glm.vec3(upX, upY, upZ)
        self.Yaw = yaw
        self.Pitch = pitch

        self.Front = CAMERA_INITIAL_FRONT
        self.Up = glm.vec3()
        self.Right = glm.vec3()
        self.MouseSensitivity = SENSITIVITY
        self.FOV = FOV

        self.UpdateCameraVectors()
        self.process_mouse = False

    def GetViewMatrix(self) -> glm.mat4:
        return glm.lookAt(self.Position, self.Position + self.Front, self.Up)

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

    def look_enabled(self, enabled: bool):
        self.process_mouse = enabled
        if enabled:
            inpStat.s_MousePosX = inpStat.s_NextMousePosX
            inpStat.s_MousePosY = inpStat.s_NextMousePosY
