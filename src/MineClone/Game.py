import os.path
import random

import numpy as np

from MineClone.Player import *
from POGLE.Event.Event import Event
from POGLE.Shader import UniformBuffer, UniformBlock

import MineClone.Face as face

class Game:
    def __init__(self):
        face.init_texture_atlas()
        self.controls = ControlSet()
        #import dill
        #if os.path.exists("worldFile.dill"):
        #    with open("worldFile.dill", "rb") as f:
        #        if Block._TextureAtlas is None:
        #            texQuadCube = TexQuadCube(NMM(glm.vec3()), glm.vec2(), glm.vec2())
        #            Block._TextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))
        #            Block.vertices = texQuadCube.vertices
        #        self.world: World = dill.load(f)

        ## elif os.path.exists("worldFile.bin"):
        ##     with open("worldfile.bin", "rb") as f:
        ##         self.world: World = World.deserialize(f.read())
        #else:
        #    self.world: World = World()
        #with open("worldFile.dill", "wb") as f:
        #        dill.dump(self.world, f)
        # with open("worldfile.bin","wb") as f:
        #     f.write(self.world.serialize())
        self.projection: glm.mat4 = glm.mat4()
        self.view: glm.mat4 = glm.mat4()

        self.ubo_mats = UniformBuffer()
        self.ub_mats = UniformBlock.create(UniformBlockLayout(
            "ub_Matrices",
            [
                VertexAttribute("u_Projection", self.projection),
                VertexAttribute("u_View", self.view)
            ]
        ))
        self.ubo_mats.bind_block(self.ub_mats.binding)
        self.ubo_mats.bind()
        self.ubo_mats.buffer_data(
            self.ub_mats.data
        )
        self.ubo_mats.unbind()

        self.ubo_face_transforms = UniformBuffer()
        self.ub_face_transforms = UniformBlock.create(
            UniformBlockLayout(
                "ub_FaceTransforms",
                [
                    VertexAttribute("u_FaceTransform", [np.array(mat.to_list()) for mat in face_model_mats.values()])
                ]
            )
        )
        self.ubo_face_transforms.bind_block(self.ub_face_transforms.binding)
        self.ubo_face_transforms.bind()
        self.ubo_face_transforms.buffer_data(
            self.ub_face_transforms.data
        )
        self.ubo_face_transforms.unbind()

        self.ubo_face_tex_positions = UniformBuffer()
        self.ub_face_tex_positions = UniformBlock.create(
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
        self.ubo_face_tex_positions.bind_block(self.ub_face_tex_positions.binding)
        self.ubo_face_tex_positions.bind()
        self.ubo_face_tex_positions.buffer_data(
            self.ub_face_tex_positions.data
        )
        self.ubo_face_tex_positions.unbind()

        self.ubo_face_tex_sizes = UniformBuffer()
        self.ub_face_tex_sizes = UniformBlock.create(
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
        self.ubo_face_tex_sizes.bind_block(self.ub_face_tex_sizes.binding)
        self.ubo_face_tex_sizes.bind()
        self.ubo_face_tex_sizes.buffer_data(
            self.ub_face_tex_sizes.data
        )

        self.ubo_face_tex_sizes.unbind()

        self.ubo_face_tex_positions.print_data()
        self.ubo_face_tex_sizes.print_data()

        # self.world = World()
        # quit()
        self.world = World(_from_file=True)

        self.mesh = self.world.spawn_region.get_mesh()
        self.mesh.add_texture("tex0", face.texture_atlas)

        self.mesh.bind_uniform_blocks(
            [
                "ub_Matrices",
                "ub_FaceTransforms",
                "ub_FaceTexPositions",
                "ub_FaceTexSizes"
            ]
        )

        self.mesh.shader.use()
        self.mesh.bind_textures()
        self.player: Player = Player(self.world, glm.vec3(1,CHUNK.HEIGHT+2,1), self.controls)
        # self.world.update()
        # self.crosshairMesh = CrosshairMesh(glm.vec2(0.05 / GetApplication().get_window().get_aspect_ratio(), 0.05))

        self.cursor_timer = 0
        self.tab_timer = 0

    @property
    def playerCam(self) -> Camera:
        return self.player.camera

    def toggle_cam_control(self) -> bool:
        window = GetApplication().get_window()
        if window.is_cursor_hidden():
            window.reveal_cursor()
            return False
        else:
            window.hide_cursor()
            return True

    def update(self, deltaTime: float, projection: glm.mat4, view: glm.mat4):
        for boundCtrl in self.controls.GetBoundControls():
            ctrlID = boundCtrl.GetID()
            if boundCtrl.GetInputState().value:
                if ctrlID == Control.ID.Config.CAM_CTRL_TGL:
                    if not self.cursor_timer:
                        self.playerCam.look_enabled(self.toggle_cam_control())
                        self.cursor_timer = 10
                elif ctrlID == Control.ID.Config.QUIT:
                    GetApplication().close()
                elif ctrlID == Control.ID.Config.CYCLE_RENDER_DISTANCE:
                    if not self.tab_timer:
                        self.renderDistance = (self.renderDistance + 1 - self.minRenderDistance) % self.renderDistanceRangeSize + self.minRenderDistance
                        self.worldRenderer.set_render_distance(self.renderDistance)
                        print(self.renderDistance + 1)
                        self.tab_timer = 20

        if self.cursor_timer > 0: self.cursor_timer -= 1
        if self.tab_timer > 0: self.tab_timer-=1

        self.player.update(deltaTime)
        # self.worldRenderer.update_origin(self.player.pos)
        self.ubo_mats.bind()
        self.ub_mats.set_data([projection, view])  # Assuming you have modified setData to handle updates
        self.ubo_mats.buffer_data(self.ub_mats.data)
        self.ubo_mats.unbind()
        # dataElements = []
        # newProjection = self.projection != projection
        # newView = self.view != view
        # if newProjection:
        #     print("New projection")
        #     self.projection = projection
        #     dataElements.append(self.projection)
        # if newView:
        #     print("New view")
        #     self.view = view
        #     dataElements.append(self.view)
        # numElements = len(dataElements)
        # if numElements:
        #     self.ubo_mats.bind()
        #     if numElements == 2:
        #         self.ub_face_transforms.set_data([projection, view])
        #         self.ubo_mats.buffer_data(self.ub_mats.data)
        #     else:
        #         data = np.array(dataElements).reshape(-1, 4, 4).transpose(0, 2, 1)
        #         offset = 0
        #         if newView:
        #             # self.world.update(self.playerCam.get_frustum(view, projection))
        #             offset = 64  # Offset to edit only the view matrix
        #         self.ubo_mats.buffer_sub_data(offset, data)
        #     self.ubo_mats.unbind()
        #     self.ubo_mats.bind_block()

    shader = None
    mesh = None
    def draw(self):
        # if self.shader == None:
        #     self.worldRenderer.worldBlockShader = ShaderProgram("block", "block")
        #     self.worldRenderer.worldBlockShader.bind_uniform_block("Matrices")
        #     self.worldRenderer.worldBlockShader.bind_uniform_block("BlockSides")
        #     self.shader = 1
        # if self.mesh is None:
        #     class Bk:
        #         cnt = 0
        #
        #         def __init__(self, pos: glm.vec3):
        #             self._pos = pos
        #             self.ID = Block.ID(random.randrange(1, len(Block.ID)))
        #             self.face_ids = list(range(6))
        #
        #             self.texDims = [
        #                 Block._TextureAtlas.get_sub_texture(Block.blockNets[self.ID][i].value) for i in range(6)
        #             ]
        #
        #             # Randomly remove face_ids and texDims
        #             keep_indices = [i for i in range(6) if random.randrange(0, 2)]  # 0 or 1
        #             if not len(keep_indices):
        #                 keep_indices.append(0)
        #             self.face_ids = [self.face_ids[i] for i in keep_indices]
        #             self.texDims = [self.texDims[i] for i in keep_indices]
        #             self.blockPositions = [self._pos for i in keep_indices]
        #
        #             self.face_instances = np.concatenate(
        #                 [np.concatenate([texDim.pos, texDim.size]) for texDim in self.texDims]
        #             )
        #             print(f"Block {Bk.cnt}:\n\t- ID: {self.ID}\n\t- pos: {self._pos}\n\t- faces: {self.face_ids}")
        #             Bk.cnt += 1
        #
        #     class Bks:
        #         def __init__(self, min: glm.vec3, max: glm.vec3, numBlocks: int = random.randrange(2, 5)):
        #             self.blocks: list[Bk] = []
        #             for _ in range(numBlocks):
        #                 self.blocks.append(Bk(
        #                     glm.vec3(
        #                         random.randrange(min.x, max.x + 1),
        #                         random.randrange(min.y, max.y + 1),
        #                         random.randrange(min.z, max.z + 1)
        #                     )
        #                 ))
        #
        #             self.faceIDs = [block.face_ids for block in self.blocks]
        #             self.faceIDs = np.concatenate(self.faceIDs, dtype=np.int32)
        #
        #             self.faceTexDims = [block.face_instances for block in self.blocks]
        #             self.faceTexDims = np.concatenate(self.faceTexDims, dtype=np.float32)
        #
        #             self.blockPositions = [
        #                 np.concatenate(block.blockPositions) for block in self.blocks
        #             ]
        #             self.blockPositions = np.concatenate(self.blockPositions, dtype=np.float32)
        #
        #     blocks = Bks(glm.vec3(-4, 2, -4), glm.vec3(4, 4, 4), 16)
        #     self.mesh = BlockMesh(blocks.faceIDs, blocks.faceTexDims, blocks.blockPositions)
        #
        # self.mesh.draw(self.shader)
        self.mesh.draw()
        # self.worldRenderer.draw()
        # self.player.draw()
        # self.crosshairMesh.draw()

    def OnEvent(self, e: Event):
        typ = e.getEventType()
        if e.isInCategory(Event.Category.Mouse):
            if typ == Event.Type.MouseButtonPressed:
                Input.SetState(e.getMouseButton(), Input.State.PRESS)
            elif typ == Event.Type.MouseButtonReleased:
                Input.SetState(e.getMouseButton(), Input.State.RELEASE)
            elif typ == Event.Type.MouseMoved:
                inpStat.s_NextMousePosX = e.getX()
                inpStat.s_NextMousePosY = e.getY()
            elif typ == Event.Type.MouseScrolled:
                inpStat.s_ScrollOffsetX = e.getXOffset()
                inpStat.s_ScrollOffsetY = e.getYOffset()
        elif e.isInCategory(Event.Category.Keyboard):
            if typ == Event.Type.KeyPressed:
                Input.SetState(e.getKeyCode(), Input.State.PRESS)
            elif typ == Event.Type.KeyReleased:
                Input.SetState(e.getKeyCode(), Input.State.RELEASE)