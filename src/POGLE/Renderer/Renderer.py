import glm

from POGLE.Renderer.Camera import *
from POGLE.Renderer.Mesh import *

class Renderer:
    def __init__(self, width: int, height: int, fovDeg: GLfloat = FOV, nearPlane: GLfloat = NEAR, farPlane: GLfloat = FAR):
        self._EnabledBuffers = GL_COLOR_BUFFER_BIT
        self._Left: int = 0
        self._Bottom: int = 0
        self._Init(width, height, fovDeg, nearPlane, farPlane)

    @staticmethod
    def set_viewport(left: np.uint32, bottom: np.uint32, width: np.uint32, height: np.uint32):
        glViewport(left, bottom, width, height)

    @staticmethod
    def set_clear_color(color: glm.vec4):
        glClearColor(color.x, color.y, color.z, color.w)

    def clear(self):
        glClear(self._EnabledBuffers)

    def enable(self, e: GLenum):
        glEnable(e)
        if GL_DEPTH_TEST:
            self._EnabledBuffers |= GL_DEPTH_BUFFER_BIT

    def disable(self, e: GLenum):
        glDisable(e)
        if GL_DEPTH_TEST:
            self._EnabledBuffers &= ~GL_DEPTH_BUFFER_BIT

    def on_window_resize(self, width: GLsizei, height: GLsizei):
        self._Width: int = width
        self._Height: int = height
        self._AspectRatio: GLfloat = self._Width / self._Height
        self.set_viewport(self._Left, self._Bottom, self._Width, self._Height)

    def update_fov(self, fovDeg: GLfloat):
        self._FOV: GLfloat = glm.radians(fovDeg)

    def update_clip_planes(self, nearPlane: GLfloat = None, farPlane: GLfloat = None):
        if nearPlane:
            self._NearPlane: GLfloat = nearPlane
        if farPlane:
            self._FarPlane: GLfloat = farPlane

    def _Init(self, width: GLsizei, height: GLsizei, fovDeg: GLfloat, nearPlane: GLfloat, farPlane: GLfloat):
        self.on_window_resize(width, height)
        self.update_fov(fovDeg)
        self.update_clip_planes(nearPlane, farPlane)
        self.enable(GL_DEPTH_TEST)
        self.set_clear_color(glm.vec4(Color.BLACK,1.0))