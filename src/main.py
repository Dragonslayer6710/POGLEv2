# from POGLE.Buffer import *
# from glfw.GLFW import *
# GLFWwindowCount = 0
# if not GLFWwindowCount:
#     if not glfwInit():
#         raise Exception("Could not initialize GLFW!")
#
# window =  glfwCreateWindow(640, 640, "", None, None)
# GLFWwindowCount += 1
#
# glfwMakeContextCurrent(window)
#
# #glfwSetWindowUserPointer(self._Window, self._Data)
# #self.set_vsync(True)
# layout = DL([
#     aLocalPosXY
# ])
# instanceLayout = DL([
#     aModel
# ], layout.nextID)
# VertexArray(
#     DataSet(
#         layout,
#         [DataPoint(layout, np.array(glm.vec2()))]
#     ),
#     [0],
#     DataSet(
#         instanceLayout,
#         [DataPoint(instanceLayout, np.array(glm.mat4()))]
#     )
# )
# quit()

from POGLE.Display.Layer.HelloLayer import *

class POGLEApp(Application):
    def __init__(self, spec: ApplicationSpecification):
        super().__init__(spec)
        self.push_layer(HelloLayer(self._Renderer))

def create_application(args: ApplicationCommandLineArgs) -> POGLEApp:
    spec = ApplicationSpecification(args)
    spec.Name = "POGLEApp"
    spec.CommandLineArgs = args
    return POGLEApp(spec)


def main(args: list[str] = []):
    myApp = create_application(args)
    myApp.open()
    del myApp


def process_input(window: Window) -> None:
    deltaTime = window.deltaTime
    if window.KeyDown(GLFW_KEY_ESCAPE):
        window.SetShouldClose(True)

    if window.KeyDown(GLFW_KEY_W):
        camera.ProcessKeyboard(Camera_Movement.FORWARD, deltaTime)
    if window.KeyDown(GLFW_KEY_S):
        camera.ProcessKeyboard(Camera_Movement.BACKWARD, deltaTime)
    if window.KeyDown(GLFW_KEY_A):
        camera.ProcessKeyboard(Camera_Movement.LEFT, deltaTime)
    if window.KeyDown(GLFW_KEY_D):
        camera.ProcessKeyboard(Camera_Movement.RIGHT, deltaTime)
    if window.KeyDown(GLFW_KEY_SPACE):
        camera.ProcessKeyboard(Camera_Movement.UP, deltaTime)
    if window.KeyDown(GLFW_KEY_LEFT_CONTROL):
        camera.ProcessKeyboard(Camera_Movement.DOWN, deltaTime)


# glfw: whenever the window size changed call this
def framebuffer_size_callback(window: GLFWwindow, width: int, height: int) -> None:
    # make sure viewport matches new window dimensions
    glViewport(0, 0, width, height)


def mouse_callback(window: GLFWwindow, xpos: float, ypos: float) -> None:
    global lastX, lastY, firstMouse

    if firstMouse:
        lastX = xpos
        lastY = ypos
        firstMouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos

    lastX = xpos
    lastY = ypos

    camera.ProcessMouseMovement(xoffset, yoffset)


def scroll_callback(window: GLFWwindow, xoffset: float, yoffset: float) -> None:
    camera.ProcessMouseScroll(yoffset)

do_profile=False
if __name__ == "__main__":
    if do_profile:
        import cProfile
        # Run the main profiling function
        profiler = cProfile.Profile()
        profiler.enable()
        main()  # your main function to profile
        profiler.disable()

        # Create a Stats object
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')  # or 'time', 'calls', etc.
        stats.dump_stats("POGLE.prof")  # Print the profiling results

    else:
        main()
