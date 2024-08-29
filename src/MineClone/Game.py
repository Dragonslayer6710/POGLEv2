import os.path
import random

import numpy as np

from MineClone.Player import *

class Game:
    def __init__(self):
        Block._TextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))

        import dill
        if os.path.exists("worldFile.dill"):
            with open("worldFile.dill", "rb") as f:
                if Block._TextureAtlas is None:
                    texQuadCube = TexQuadCube(NMM(glm.vec3()), glm.vec2(), glm.vec2())
                    Block._TextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))
                    Block.vertices = texQuadCube.vertices
                self.world: World = dill.load(f)

        # elif os.path.exists("worldFile.bin"):
        #     with open("worldfile.bin", "rb") as f:
        #         self.world: World = World.deserialize(f.read())
        else:
            self.world: World = World()
        with open("worldFile.dill", "wb") as f:
                dill.dump(self.world, f)
        # with open("worldfile.bin","wb") as f:
        #     f.write(self.world.serialize())

        self.worldRenderer: WorldRenderer = WorldRenderer(self.world)
        self.player: Player = Player(self.world, glm.vec3(0,self.world.max.y+2,0))

        self.crosshairMesh = CrosshairMesh(glm.vec2(0.5))

        self.MatricesUBO: UniformBuffer = UniformBuffer()
        self.matUB: UniformBlock = UniformBlock.create([glm.mat4(), glm.mat4()])

        self.BlockSidesUBO: UniformBuffer = UniformBuffer()
        self.blockSidesUB: UniformBlock = UniformBlock.create(
            QuadCube.face_matrices,
            UniformBlockLayout(
                "BlockSides",
                [
                    FloatDA.Mat4() for i in range(6)
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

    @property
    def playerCam(self) -> Camera:
        return self.player.camera

    def update(self, deltaTime: float):
        self.player.update(deltaTime)
        self.worldRenderer.update_origin(self.player.pos)

    shader = None
    mesh = None
    def draw(self, projection: glm.mat4, view: glm.mat4):
        self.MatricesUBO.bind()
        self.matUB.setData([projection, view])
        self.MatricesUBO.buffer_data(self.matUB.bytes, self.matUB.data)
        self.MatricesUBO.unbind()
        self.MatricesUBO.bind_block()

        if self.shader == None:
            self.shader = ShaderProgram("block", "block")
            self.shader.bind_uniform_block("Matrices")
            self.shader.bind_uniform_block("BlockSides")
        if self.mesh is None:
            class Bk:
                cnt = 0
                def __init__(self, pos: glm.vec3):
                    self.pos = pos
                    self.ID = Block.ID(random.randrange(1,len(Block.ID)))
                    self.face_ids = [i for i in range(6)]
                    print(f"Block {Bk.cnt}:\n\t- ID: {self.ID}\n\t- pos: {self.pos}")
                    Bk.cnt += 1
                    self.texDims = [
                        Block._TextureAtlas.get_sub_texture(Block.blockNets[self.ID][i].value) for i in range(6)
                    ]
                    for i in range(6):
                        if random.randrange(0,1):
                            del self.face_ids[i]
                            del self.texDims[i]
                    self.face_instances = np.concatenate(list([*self.texDims[i].pos, *self.texDims[i].size] for i in range(6)))
            class Bks:
                def __init__(self, min: glm.vec3, max: glm.vec3, numBlocks: int = random.randrange(2,5)):
                    self.blocks: list[Bk] = []
                    for i in range(numBlocks):
                        self.blocks.append(Bk(
                            glm.vec3(
                                random.randrange(min.x, max.x + 1),
                                random.randrange(min.y, max.y + 1),
                                random.randrange(min.z, max.z + 1)
                            )
                        ))
                    self.faceIDs = [block.face_ids for block in self.blocks]
                    self.faceIDs = np.concatenate(self.faceIDs, dtype=np.short)
                    self.faceTexDims = [block.face_instances for block in self.blocks]
                    self.faceTexDims = np.concatenate(self.faceTexDims, dtype=np.float32)
                    self.blockPositions = [block.pos for block in self.blocks]
                    self.blockPositions = np.concatenate(self.blockPositions, dtype=np.float32)

            blocks = Bks(glm.vec3(0,2,0), glm.vec3(0,4,0))
            self.mesh = BlockMesh(blocks.faceIDs, blocks.faceTexDims, blocks.blockPositions)
        self.mesh.draw(self.shader)

        #self.worldRenderer.draw(projection, view)
        self.player.draw(projection, view)
        self.crosshairMesh.draw()