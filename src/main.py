import numpy as np

from Shader import *

vertices: Vertices = Vertices(
    [
        [
            [-0.5, -0.5, 0.0,  1.0],
            [ 1.0,  1.0, 1.0,  1.0],
        ],
        [
            [ 0.5, -0.5, 0.0,  1.0],
            [ 1.0,  0.0, 0.0,  1.0],
        ],
        [
            [-0.5,  0.5,  0.0,  1.0],
            [ 0.0,  1.0,  0.0,  1.0],
        ],
        [
            [ 0.5,  0.5,  0.0,  1.0],
            [ 0.0,  0.0,  1.0,  1.0],
        ]

    ]
)
indices = np.array([0,1,2, 1,2,3], np.uint16)
def main():
    # Initialise GLFW
    if not glfw.init():
        return
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(640, 480, "Hello World!", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    defShader = ShaderProgram()
    defShader.use()

    vao = VertexArray(vertices, indices)
    vao.bind()

    while not glfw.window_should_close(window):
        # Render here
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, None)
        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()

