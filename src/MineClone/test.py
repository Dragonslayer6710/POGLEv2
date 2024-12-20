import random

import glfw
import glm
import numpy as np

import Face as face
from POGLE.OGL.OpenGLContext import *
from POGLE.Shader import UniformBlockLayout

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW can't be initialized")

WINDOW_TITLE = "Terrain Test"
window = glfw.create_window(800, 600, WINDOW_TITLE, None, None)
glfw.make_context_current(window)

# from World import *
#from Block import _face_model_mats, face_texture_atlas
from Game import *
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
# quit()

# Circular motion parameters
radius = 5.0  # Distance from the origin
rotation_speed = 30.0  # Degrees per second
current_angle = 0.0  # Current rotation angle in degrees

# Y-axis motion parameters
y_rotation_speed = 15.0  # Degrees per second for y-axis motion
current_y_angle = 0.0  # Current rotation angle for y-axis motion

draw_hitboxes = True
test_game = False
def _main():
    global current_angle, current_y_angle
    if not test_game:
        face.init_texture_atlas()

        world = World()#World.from_file()

        regions = []
        chunks = []
        blocks = []

        if draw_hitboxes:
            for region_axis in world.regions:
                for region in filter(lambda x: x is not None, region_axis):
                    regions.append(region)
                    for chunk_axis in region.chunks:
                        for chunk in filter(lambda x: x is not None, chunk_axis):
                            chunks.append(chunk)
                            for section in chunk.blocks:
                                for block_axis in section:
                                    for block in filter(lambda x: x is not None, block_axis):
                                        blocks.append(block)

            region_mats = VertexAttribute("a_Model", [
                NMM(region.pos, s=(region.size / 2)*1.0001, glr=True)
                for region in regions
            ], divisor=1)

            chunk_mats = VertexAttribute("a_Model", [
                NMM(chunk.pos, s=(chunk.size/2)*1.001, glr=True)
                for chunk in chunks
            ], divisor=1)

            block_mats = VertexAttribute("a_Model", [
                NMM(block.pos, s=(block.size / 2)*1.0001, glr=True)
                for block in blocks
            ], divisor=1)

            hitbox_data = [
                [region_mats, [Color.RED] * 8],
                [chunk_mats, [Color.BLUE] * 8],
                [block_mats, [Color.GREEN] * 8]
            ]

            hitbox_data_layouts = [
                DataLayout([
                    VertexAttribute("a_Position", Cube._positions),
                    VertexAttribute("a_Colour", colors),
                    VertexAttribute("a_Alpha", [0.15 for _ in range(8)]),
                    mats,
                ]) for (mats, colors) in hitbox_data
            ]

            hitbox_vaos = [None] * len(hitbox_data_layouts)
            for i, hitbox_dl in enumerate(hitbox_data_layouts):
                hitbox_vaos[i] = VertexArray()
                hitbox_vaos[i].set_ebo(data=np.array(Cube._indices, dtype=np.ushort))
                hitbox_vaos[i].add_vbo(hitbox_dl)
            cube_shader = ShaderProgram("cube", "cube")
        #chunk = Chunk()
        #r = _Region()
        #r.pos += glm.vec3(0, CHUNK.HEIGHT // 2, 0)
        #chunk.initialize(r)
        #chunk.update()

        # r.update()

        # Define the perspective projection and view matrix
        from POGLE.Renderer.Camera import Camera
        camera = Camera(zFar=1000)
        # mats = [glm.mat4()] * 65536
        # mats = chunk.block_instances

        # mesh = ShapeMesh(
        #    # mats,  # chunk.block_instances[:-1],
        #    blocks_data_layout,
        #    shader, {"tex0": _faceTextureAtlas}
        # )
        # quit()
        glEnable(GL_DEPTH_TEST)

        projection = camera.get_projection()  # glm.perspective(glm.radians(45.0), 800 / 600, 0.1, 100.0)
        camera.Position = glm.vec3(8, 60, 9)

        camera.Pitch = -15
        view = camera.get_view_matrix()
        ubo_mats = UniformBuffer()
        ub_mats = UniformBlock.create(UniformBlockLayout(
            "ub_Matrices",
            [
                VertexAttribute("u_Projection", projection),
                VertexAttribute("u_View", view)
            ]
        ))
        ubo_mats.bind_block(ub_mats.binding)
        ubo_mats.bind()
        ubo_mats.buffer_data(
            ub_mats.data
        )
        ubo_mats.unbind()

        ubo_face_transforms = UniformBuffer()
        ub_face_transforms = UniformBlock.create(
            UniformBlockLayout(
                "ub_FaceTransforms",
                [
                    VertexAttribute("u_FaceTransform", [np.array(mat.to_list()) for mat in face_model_mats.values()])
                ]
            )
        )
        ubo_face_transforms.bind_block(ub_face_transforms.binding)
        ubo_face_transforms.bind()
        ubo_face_transforms.buffer_data(
            ub_face_transforms.data
        )
        ubo_face_transforms.unbind()

        ubo_face_tex_positions = UniformBuffer()
        ub_face_tex_positions = UniformBlock.create(
            UniformBlockLayout(
                "ub_FaceTexPositions",
                [
                    VertexAttribute(
                        "u_TexPositions",
                        [
                            face.texture_atlas.get_sub_texture(i).pos
                            for i in range(4)
                        ]
                    )
                ]
            )
        )
        ubo_face_tex_positions.bind_block(ub_face_tex_positions.binding)
        ubo_face_tex_positions.bind()
        ubo_face_tex_positions.buffer_data(
            ub_face_tex_positions.data
        )
        ubo_face_tex_positions.unbind()

        ubo_face_tex_sizes = UniformBuffer()
        ub_face_tex_sizes = UniformBlock.create(
            UniformBlockLayout(
                "ub_FaceTexSizes",
                [
                    VertexAttribute(
                        "u_TexSizes",
                        [face.texture_atlas.get_sub_texture(0).size]
                    )
                ]
            )
        )
        ubo_face_tex_sizes.bind_block(ub_face_tex_sizes.binding)
        ubo_face_tex_sizes.bind()
        ubo_face_tex_sizes.buffer_data(
            ub_face_tex_sizes.data
        )

        ubo_face_tex_sizes.unbind()

        ubo_face_tex_positions.print_data()
        ubo_face_tex_sizes.print_data()

        mesh = world.spawn_region.get_mesh()
        mesh.add_texture("tex0", face.texture_atlas)

        mesh.bind_uniform_blocks(
            [
                "ub_Matrices",
                "ub_FaceTransforms",
                "ub_FaceTexPositions",
                "ub_FaceTexSizes"
            ]
        )

        if draw_hitboxes:
            cube_shader.use()
            cube_shader.bind_uniform_block("ub_Matrices")
    else:
        game = Game()
        camera = game.playerCam
    # Main loop
    previous_time = glfw.get_time()
    old_x, old_y = 0, 0
    track_cursor = False
    camera.ProcessMouseMovement(0,0, True)

    sprinting = False
    sprint_mod = 1.25

    if not test_game:
        mesh.shader.use()
        mesh.bind_textures()


    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
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

        keys = [
            glfw.KEY_W,
            glfw.KEY_A,
            glfw.KEY_S,
            glfw.KEY_D,
            glfw.KEY_SPACE,
            glfw.KEY_LEFT_CONTROL,
            glfw.KEY_ESCAPE
        ]
        speed = 0.05
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
                    case glfw.KEY_LEFT_SHIFT:
                        if not sprinting:
                            sprinting = True
                            speed *= sprint_mod
            elif key_state == 0:
                if key == glfw.KEY_LEFT_SHIFT:
                    if sprinting:
                        sprinting = False
                        speed /= sprint_mod
        if escape_pressed:
            track_cursor = not track_cursor
            match glfw.get_input_mode(window, glfw.CURSOR):
                case glfw.CURSOR_DISABLED:
                    mode = glfw.CURSOR_NORMAL
                case _:
                    mode = glfw.CURSOR_DISABLED
            glfw.set_input_mode(window, glfw.CURSOR, mode)

        if track_cursor:
            camera.ProcessMouseMovement(delta_x, -delta_y, True)

        projection = camera.get_projection()
        view = camera.get_view_matrix()
        if not test_game:
            # Update UBO with new projection and view matrices
            ubo_mats.bind()
            ub_mats.set_data([projection, view])  # Assuming you have modified setData to handle updates
            ubo_mats.buffer_data(ub_mats.data)
            ubo_mats.unbind()


        # shader.use()
        # vao.bind()
        # glDrawElementsInstanced(GL_TRIANGLES, len(Quad._indices), GL_UNSIGNED_SHORT, None, block_shape.data_layout.attributes[-1].size)
        # vao.unbind()

            mesh.draw()

            if draw_hitboxes:
                cube_shader.use()
                for hitbox_vao, hitbox_dl in zip(hitbox_vaos, hitbox_data_layouts):
                    hitbox_vao.bind()
                    glDrawElementsInstanced(GL_TRIANGLES, len(Cube._indices), GL_UNSIGNED_SHORT, None, hitbox_dl.attributes[-1].size)
                    hitbox_vao.unbind()
        else:
            game.update(delta_time, projection, view)
            game.draw()

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    # Clean up
    glfw.terminate()


if __name__ == "__main__":
    _main()
