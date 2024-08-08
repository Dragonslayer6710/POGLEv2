from POGLE.Core.Core import *

# Default Screen Dims
SCR_WIDTH = 800
SCR_HEIGHT = 600
ASPECT_RATIO = SCR_WIDTH / SCR_HEIGHT

class WindowInfo:
    Width: int = 800
    Height: int = 600
    Title: str = "Hello World!"
    _ar: float = None
    WinHint: dict = {"version_major":4, "version_minor":5, "opengl_profile":GLFW_OPENGL_CORE_PROFILE}

    def aspect_ratio(self):
        if self._ar == -1:
            self._ar = self.Width / self.Height
        return self._ar
class Window:
    def __init__(self, wi: WindowInfo = WindowInfo()):

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, wi.WinHint["version_major"])
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, wi.WinHint["version_minor"])
        glfwWindowHint(GLFW_OPENGL_PROFILE, wi.WinHint["opengl_profile"])

        self.glfwin: GLFWwindow = glfwCreateWindow(wi.Width, wi.Height, wi.Title, None, None)
        if not self.glfwin:
            print("Failed to create GLFW window")
            glfwTerminate()
            return

        self.currentFrame = 0.0
        self.deltaTime = 0.0
        self.lastFrame = 0.0

        self.lastX = SCR_WIDTH / 2.0
        self.lastY = SCR_HEIGHT / 2.0
        self.firstMouse = True

        self.nbFrames = 0
    def showFPS(self):
        self.nbFrames += 1
        if self.currentFrame % 1 > 0.9:
            fps = round(self.nbFrames / self.deltaTime)
            glfwSetWindowTitle(self.glfwin, f"Hello World!\t[{fps} FPS]")
            self.nbFrames = 0

    def Start(self, HandleInput, Update, Render):
        while not self.ShouldClose():
            # per-frame time logic
            self.currentFrame = glfwGetTime()
            self.deltaTime = self.currentFrame - self.lastFrame
            self.lastFrame = self.currentFrame
            self.showFPS()
            # input
            HandleInput(self)
            Update(self)
            Render(self)

            # Swap front and back buffers
            self.SwapBuffers()

            # Poll for and process events
            glfwPollEvents()

    def MakeContextCurrent(self):
        glfwMakeContextCurrent(self.glfwin)

    def SetFramebufferSizeCallback(self, framebuffer_size_callback):
        glfwSetFramebufferSizeCallback(self.glfwin, framebuffer_size_callback)

    def SetCursorPosCallBack(self, cursorpos_callback):
        glfwSetCursorPosCallback(self.glfwin, cursorpos_callback)

    def SetScrollCallback(self, scroll_callback):
        glfwSetScrollCallback(self.glfwin, scroll_callback)
    def SetInputMode(self, mode, value):
        glfwSetInputMode(self.glfwin, mode, value)

    def ShouldClose(self):
        return glfwWindowShouldClose(self.glfwin)

    def SetShouldClose(self, value: bool):
        glfwSetWindowShouldClose(self.glfwin, value)

    def GetKey(self, keycode: int):
        return glfwGetKey(self.glfwin, keycode)

    def KeyDown(self, keycode: int):
        return self.GetKey(keycode) == GLFW_PRESS

    def KeyUp(self, keycode: int):
        return self.GetKey(keycode) == GLFW_RELEASE

    def SwapBuffers(self):
        glfwSwapBuffers(self.glfwin)

