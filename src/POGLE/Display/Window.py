from POGLE.Event.ApplicationEvent import *
from POGLE.Event.MouseEvent import *
from POGLE.Event.KeyEvent import *

from POGLE.Renderer.Renderer import *

from typing import Callable

# Default Screen Dims
POGLE_DEF_WIDTH = 800
POGLE_DEF_HEIGHT = 600


class GLContextProps:
    def __init__(self, major: int = 4, minor: int = 5, profile: int = GLFW_OPENGL_CORE_PROFILE):
        self.Major = major
        self.Minor = minor
        self.Profile = profile


defaultGLContextProps = GLContextProps()


class WindowProps:
    def __init__(self, title: str = "POGLE", width: int = POGLE_DEF_WIDTH, height: int = POGLE_DEF_HEIGHT,
                 hints: GLContextProps = defaultGLContextProps):
        self.Title: str = title
        self.Width: int = width
        self.Height: int = height
        self.Hints: GLContextProps = hints


defaultWindowProps = WindowProps()

GLFWwindowCount = 0

class Window:
    _CentrePos: glm.vec2
    _CursorHidden: bool
    _ImGuiLayerBlock: bool
    class WindowData:
        def __init__(self, title: str, width: int, height: int):
            self.Title: str = title
            self.Width: int = width
            self.Height: int = height
            self.VSync: bool = False
            self.EventCallback: Callable[[Event], None] = None

    def __init__(self, props: WindowProps = defaultWindowProps):
        self._Init(props)
        self._CursorHidden = False
        self._ImGuiLayerBlock = False

        self.currentFrame = 0.0
        self.deltaTime = 0.0
        self.lastFrame = 0.0

        self.lastX = self._CentrePos.x
        self.lastY = self._CentrePos.y
        self.firstMouse = True

        self.nbFrames = 0

    def __del__(self):
        self._Shutdown()

    def on_update(self):
        glfwPollEvents()
        glfwSwapBuffers(self._Window)

    def get_width(self)->int:
        return self._Data.Width

    def get_height(self)->int:
        return self._Data.Height

    def set_event_callback(self, callback: Callable[[Event], None]):
        self._Data.EventCallback = callback

    def set_vsync(self, enabled: bool):
        if enabled:
            glfwSwapInterval(1)
        else:
            glfwSwapInterval(0)
        self._Data.VSync = enabled

    def is_vsync(self) -> bool:
        return self._Data.VSync

    def get_native_window(self) -> GLFWwindow:
        return self._Window

    def recalculate_centre_pos(self):
        self._CentrePos: glm.vec2 = glm.vec2(
            self._Data.Width / 2,
            self._Data.Height / 2
        )

    def centre_cursor(self):
        glfwSetCursorPos(self._Window, self._CentrePos.x, self._CentrePos.y)

    def hide_cursor(self):
        if not self._CursorHidden:
            if not self._ImGuiLayerBlock:
                glfwSetInputMode(self._Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED)
            self._CursorHidden = True

    def reveal_cursor(self):
        if self._CursorHidden:
            glfwSetInputMode(self._Window, GLFW_CURSOR, GLFW_CURSOR_NORMAL)
            if not self._ImGuiLayerBlock:
                self.centre_cursor()
            self._CursorHidden = False

    def is_cursor_hidden(self) -> bool:
        return self._CursorHidden

    def set_imgui_layer_block(self, status: bool):
        self._ImGuiLayerBlock = status

    def _Init(self, props: WindowProps):
        global GLFWwindowCount
        self._Data: Window.WindowData = Window.WindowData(props.Title, props.Width, props.Height)
        self.recalculate_centre_pos()

        if not GLFWwindowCount:
            if not glfwInit():
                raise Exception("Could not initialize GLFW!")

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, props.Hints.Major)
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, props.Hints.Minor)
        glfwWindowHint(GLFW_OPENGL_PROFILE, props.Hints.Profile)

        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)

        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE)

        self._Window: GLFWwindow = glfwCreateWindow(self._Data.Width, self._Data.Height, self._Data.Title, None, None)
        GLFWwindowCount += 1

        glfwMakeContextCurrent(self._Window)

        glfwSetWindowUserPointer(self._Window, self._Data)
        self.set_vsync(True)

        def window_size_callback(window: GLFWwindow, width: int, height: int):
            data = glfwGetWindowUserPointer(window)
            data.Width= width
            data.Height = height
            data.EventCallback(WindowResizeEvent(width, height))

        glfwSetWindowSizeCallback(self._Window, window_size_callback)

        def window_close_callback(window: GLFWwindow):
            data = glfwGetWindowUserPointer(window)
            data.EventCallback(WindowCloseEvent())

        glfwSetWindowCloseCallback(self._Window, window_close_callback)

        def key_callback(window: GLFWwindow, keycode: GLuint, scancode: GLint, action: GLint, mods: GLint):
            data = glfwGetWindowUserPointer(window)
            if GLFW_PRESS == action:
                data.EventCallback(KeyPressedEvent(keycode, False))
            elif GLFW_RELEASE == action:
                data.EventCallback(KeyReleasedEvent(keycode))
            elif GLFW_REPEAT == action:
                data.EventCallback(KeyPressedEvent(keycode, True))

        glfwSetKeyCallback(self._Window, key_callback)

        def char_callback(window: GLFWwindow, keycode: GLuint):
            data = glfwGetWindowUserPointer(window)
            data.EventCallback(KeyTypedEvent(keycode))

        glfwSetCharCallback(self._Window, char_callback)

        def mouse_button_callback(window: GLFWwindow, button: GLuint, action: GLint, mods: GLint):
            data = glfwGetWindowUserPointer(window)
            if GLFW_PRESS == action:
                data.EventCallback(MouseButtonPressedEvent(button))
            elif GLFW_RELEASE == action:
                data.EventCallback(MouseButtonReleasedEvent(button))

        glfwSetMouseButtonCallback(self._Window, mouse_button_callback)

        def scroll_callback(window: GLFWwindow, xOffset: float, yOffset: float):
            data = glfwGetWindowUserPointer(window)
            data.EventCallback(MouseScrolledEvent(xOffset, yOffset))

        glfwSetScrollCallback(self._Window, scroll_callback)

        def cursor_pos_callback(window: GLFWwindow, xPos: float, yPos: float):
            data = glfwGetWindowUserPointer(window)
            data.EventCallback(MouseMovedEvent(xPos, yPos))

        glfwSetCursorPosCallback(self._Window, cursor_pos_callback)

    def _Shutdown(self):
        global GLFWwindowCount
        glfwDestroyWindow(self._Window)
        GLFWwindowCount -= 1
        if not GLFWwindowCount:
            glfwTerminate()

    def get_time(self):
        return glfwGetTime()

    def show_fps(self, time: float, deltaTime: float):
        self.nbFrames += 1
        if time % 1 > 0.9:
            fps = round(self.nbFrames / deltaTime)
            glfwSetWindowTitle(self._Window, f"{self._Data.Title}\t[{fps} FPS]")
            self.nbFrames = 0