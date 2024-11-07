import random

import glfw
import numpy as np

from Face import initFaceTextureAtlas
from POGLE.OGL.OpenGLContext import *
from POGLE.Shader import UniformBlockLayout

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW can't be initialized")

window = glfw.create_window(800, 600, "Face Test", None, None)
glfw.make_context_current(window)
initFaceTextureAtlas()

from Chunk import *
from Chunk import _Region
from Block import _face_model_mats, _faceTextureAtlas
import math

# dl = DataLayout(
#     [
#         VertexAttribute("a_Position", Quad._positions),
#         VertexAttribute("a_Normal", [glm.ivec3() for _ in range(4)]),
#         VertexAttribute("a_Colour", [glm.ivec3(_) for _ in range(4)]),
#
#         VertexAttribute("a_FaceInstanceID", [i for i in range(6)], divisor=1),
#         VertexAttribute("a_TexID", [i for i in range(6)], divisor=1),
#         VertexAttribute("a_SizeID", [i for i in range(6)], divisor=1),
#
#         VertexAttribute("a_Model", glm.imat4(), divisor=6)  # .set_attribute_pointer(0, 0, 0, True)
#     ]
# )#.set_pointers(True)
# dl.set_pointers(True)
# data = dl.get_data()
# vao = VertexArray()
# vao.add_vbo(dl)


# Circular motion parameters
radius = 5.0  # Distance from the origin
rotation_speed = 30.0  # Degrees per second
current_angle = 0.0  # Current rotation angle in degrees

# Y-axis motion parameters
y_rotation_speed = 15.0  # Degrees per second for y-axis motion
current_y_angle = 0.0  # Current rotation angle for y-axis motion


def _main():
    global current_angle, current_y_angle

    # world = World()#World.from_file()
    chunk = Chunk()
    r = _Region()
    chunk.initialize(r)
    # r.update()

    bfs = chunk.get_shape()

    shader = ShaderProgram("block", "block")

    # Define the perspective projection and view matrix
    from POGLE.Renderer.Camera import Camera
    camera = Camera(zFar=1000)
    camera.Position = glm.vec3(0, 64, 0)
    vao = VertexArray()
    vao.set_ebo(data=np.array(Quad._indices, dtype=np.ushort))
    vao.add_vbo(bfs)
    shader.use()
    _faceTextureAtlas.bind()
    texture_unit = _faceTextureAtlas.get_texture_slot()
    shader.setTexture("tex0", texture_unit)
    #mats = [glm.mat4()] * 65536
    #mats = chunk.block_instances

    #mesh = ShapeMesh(
    #    # mats,  # chunk.block_instances[:-1],
    #    bfs,
    #    shader, {"tex0": _faceTextureAtlas}
    #)
    # quit()
    glEnable(GL_DEPTH_TEST)

    projection = camera.get_projection()  # glm.perspective(glm.radians(45.0), 800 / 600, 0.1, 100.0)
    view = glm.translate(camera.GetViewMatrix(), glm.vec3(0, 0, -5))
    matrices_ubo = UniformBuffer()
    matrices_ub = UniformBlock.create(UniformBlockLayout(
        "ub_Matrices",
        [
            VertexAttribute("u_Projection", projection),
            VertexAttribute("u_View", view)
        ]
    ))
    matrices_ubo.bind_block(matrices_ub.binding)

    matrices_ubo.bind()
    data = matrices_ub.layout.get_data()
    matrices_ubo.buffer_data(data[0][2])
    matrices_ubo.unbind()

    shader.bind_uniform_block("ub_Matrices")

    face_data_ubo = UniformBuffer()
    face_data_ub = UniformBlock.create(
        UniformBlockLayout(
            "ub_FaceData",
            [
                VertexAttribute("u_FaceTransform", list(_face_model_mats.values())),
                VertexAttribute(
                    "u_TexPositions",
                    [
                        _faceTextureAtlas.get_sub_texture(i).pos
                        for i in range(4)
                    ]
                ),
                VertexAttribute(
                    "u_TexSizes",
                    [_faceTextureAtlas.get_sub_texture(0).size]
                )
            ]
        )
    )
    face_data_ubo.bind_block(face_data_ub.binding)

    face_data_ubo.bind()
    data = face_data_ub.layout.get_data()
    data_sum = data[0][2]# + data[0][2]
    for i in range(1, len(data)):
        data_sum += data[i][2]
    face_data_ubo.buffer_data(data_sum)
    face_data_ubo.unbind()

    shader.bind_uniform_block("ub_FaceData")

    # Main loop
    previous_time = glfw.get_time()
    old_x, old_y = 0, 0
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
    while not glfw.window_should_close(window):
        # Clear the color buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Update the current angle based on time elapsed
        current_time = glfw.get_time()
        delta_time = current_time - previous_time
        previous_time = current_time
        current_angle += rotation_speed * delta_time

        # Update angle for x-z circular motion
        current_angle += rotation_speed * delta_time

        # Reset angle to avoid overflow
        if current_angle >= 360.0:
            current_angle -= 360.0

        # Update angle for y motion (slower)
        current_y_angle += y_rotation_speed * delta_time
        if current_y_angle >= 360.0:
            current_y_angle -= 360.0

        # Calculate the camera's position based on the updated angles
        camera_x = radius * math.cos(math.radians(current_angle))
        camera_z = radius * math.sin(math.radians(current_angle))

        # Calculate y position based on the y rotation angle
        camera_y = 0.0  # math.sin(math.radians(current_y_angle)) * 2.0  # Adjust the multiplier for height

        camera_position = glm.vec3(camera_x, camera_y, camera_z)

        # Create the view matrix looking towards the origin
        new_x, new_y = glfw.get_cursor_pos(window)
        delta_x, delta_y = new_x - old_x, new_y - old_y
        old_x, old_y = new_x, new_y

        camera.ProcessMouseMovement(delta_x, -delta_y, True)

        keys = [
            glfw.KEY_W,
            glfw.KEY_A,
            glfw.KEY_S,
            glfw.KEY_D,
            glfw.KEY_SPACE,
            glfw.KEY_LEFT_CONTROL,
            glfw.KEY_ESCAPE
        ]
        speed = 0.01
        escape_pressed = False
        for key in keys:
            key_state = glfw.get_key(window, key)
            if key_state == 1:
                match key:
                    case glfw.KEY_W:
                        camera.Position += camera.Front * speed
                    case glfw.KEY_A:
                        camera.Position -= camera.Right * speed
                    case glfw.KEY_S:
                        camera.Position -= camera.Front * speed
                    case glfw.KEY_D:
                        camera.Position += camera.Right * speed
                    case glfw.KEY_SPACE:
                        camera.Position += camera.WorldUp * speed
                    case glfw.KEY_LEFT_CONTROL:
                        camera.Position -= camera.WorldUp * speed
                    case glfw.KEY_ESCAPE:
                        escape_pressed = True
        if escape_pressed:
            match glfw.get_input_mode(window, glfw.CURSOR):
                case glfw.CURSOR_DISABLED:
                    mode = glfw.CURSOR_NORMAL
                case _:
                    mode = glfw.CURSOR_DISABLED
            glfw.set_input_mode(window, glfw.CURSOR, mode)

        view = camera.GetViewMatrix()

        # Update UBO with new projection and view matrices
        matrices_ubo.bind()
        matrices_ub.setData([projection, view])  # Assuming you have modified setData to handle updates
        matrices_ubo.buffer_data(matrices_ub.data)
        matrices_ubo.unbind()

        shader.bind_uniform_block("Matrices")

        shader.use()
        vao.bind()
        glDrawElementsInstanced(GL_TRIANGLES, len(Quad._indices), GL_UNSIGNED_SHORT, None, bfs.attributes[-1].size)
        vao.unbind()
        #mesh.draw()

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    # Clean up
    glfw.terminate()


if __name__ == "__main__":
    _main()
