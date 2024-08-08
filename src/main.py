from POGLE.Geometry.Mesh import *

from POGLE.Camera import *
from POGLE.Core.Window import *

# Screen Dims
SCR_WIDTH = 800
SCR_HEIGHT = 600
ASPECT_RATIO = SCR_WIDTH / SCR_HEIGHT

# camera
camera = Camera()

if not glfwInit():
    raise Exception("glfw failed to initialise")

firstMouse = True


def main():
    global deltaTime, lastFrame, firstMouse

    window = Window()
    firstMouse = window.firstMouse

    window.MakeContextCurrent()

    window.SetFramebufferSizeCallback(framebuffer_size_callback)
    window.SetCursorPosCallBack(mouse_callback)
    window.SetScrollCallback(scroll_callback)

    # Capture mouse
    window.SetInputMode(GLFW_CURSOR, GLFW_CURSOR_DISABLED)

    # defShader = ShaderProgram()
    # defShader.use()

    # quadModel, pentaModel = glm.mat4(1.0), glm.mat4(1.0)

    # quadModel = glm.translate(glm.scale(quadModel, glm.vec3(0.5)), glm.vec3(-1.0, 0.0, -5.0))
    # pentaModel = glm.translate(glm.scale(pentaModel, glm.vec3(0.5)), glm.vec3(1.0, 0.0, -5.0))

    # quadMesh = Mesh(Shapes.Quad, instances=Instances(interleave_arrays([quadModel])))
    # pentaMesh = Mesh(Shapes.Pentagon)

    blockShader = ShaderProgram()
    blockShader.use()
    from random import randrange
    qcMesh = QuadCubeMesh(QuadCubes([
        NMM(glm.vec3(randrange(-100, 100) , randrange(-100, 100) , randrange(-100, 100) )) for i in range(randrange(2000, 5000))
    ]))

    #wcMesh = WireframeCubeMesh()

    def Update(window: Window):
        projection = glm.perspective(glm.radians(camera.Zoom), ASPECT_RATIO, 0.1, 100.0)

        blockShader.setMat4("uProjection", projection)
        blockShader.setMat4("uView", camera.GetViewMatrix())


    def Render(window: Window):
        # Render here
        glClearColor(0.5, 0.3, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # defShader.setMat4("uModel", pentaModel)
        # pentaMesh.draw()
        qcMesh.draw()
        #qcMeshB.draw()
        #wcMesh.draw()



    # configure global opengl state
    glEnable(GL_DEPTH_TEST)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_BLEND)
    window.Start(process_input, Update, Render)

    glfwTerminate()



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

if __name__ == "__main__":
    main()
