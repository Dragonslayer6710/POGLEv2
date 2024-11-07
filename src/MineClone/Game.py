import os.path
import random

import numpy as np

from MineClone.Player import *
from POGLE.Geometry.Shape import QuadCube
from POGLE.Shader import UniformBuffer, UniformBlock


class Game:
    def __init__(self):
        initFaceTextureAtlas()
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

        self.MatricesUBO: UniformBuffer = UniformBuffer()
        self.matUB: UniformBlock = UniformBlock.create([self.projection, self.view])

        self.BlockSidesUBO: UniformBuffer = UniformBuffer()
        self.blockSidesUB: UniformBlock = UniformBlock.create(
            QuadCube.face_matrices,
            UniformBlockLayout(
                "BlockSides",
                [
                    VA.Float().Mat4() for i in range(6)
                ]
            )
        )
        self.BlockSidesUBO.bind()
        self.BlockSidesUBO.buffer_data(
            self.blockSidesUB.bytes,
            self.blockSidesUB.data
        )
        self.BlockSidesUBO.unbind()
        self.BlockSidesUBO.bind_block(1)

        self.world: World = World()
        self.player: Player = Player(self.world, glm.vec3(0,self.world.max.y+2,0))
        self.world.update(self.playerCam.get_frustum())
        self.worldRenderer: WorldRenderer = WorldRenderer(self.world)
        self.crosshairMesh = CrosshairMesh(glm.vec2(0.05 / GetApplication().get_window().get_aspect_ratio(), 0.05))

    @property
    def playerCam(self) -> Camera:
        return self.player.camera

    def update(self, deltaTime: float, projection: glm.mat4, view: glm.mat4):
        self.player.update(deltaTime)
        self.worldRenderer.update_origin(self.player.pos)

        dataElements = []
        newProjection = self.projection != projection
        newView = self.view != view
        if newProjection:
            self.projection = projection
            dataElements.append(self.projection)
        if newView:
            self.view = view
            dataElements.append(self.view)
        numElements = len(dataElements)
        if numElements:
            self.MatricesUBO.bind()
            if numElements == 2:
                self.matUB.set_data([projection, view])
                self.MatricesUBO.buffer_data(self.matUB.bytes, self.matUB.data)
            else:
                data = DataPoint.prepare_data(dataElements, DataLayout([FloatDA.Mat4()]))
                offset = 0
                if newView:
                    self.world.update(self.playerCam.get_frustum(view, projection))
                    offset = 64  # Offset to edit only the view matrix
                self.MatricesUBO.buffer_sub_data(offset, 64, data)
            self.MatricesUBO.unbind()
            self.MatricesUBO.bind_block()

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

        self.worldRenderer.draw()
        self.player.draw()
        self.crosshairMesh.draw()