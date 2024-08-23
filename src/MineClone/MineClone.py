import os.path

from MineClone.Player import *

class Game:
    def __init__(self):
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

    @property
    def playerCam(self) -> Camera:
        return self.player.camera

    def update(self, deltaTime: float):
        self.player.update(deltaTime)
        self.worldRenderer.update_origin(self.player.pos)


    def draw(self, projection: glm.mat4, view: glm.mat4):
        self.worldRenderer.draw(projection, view)
        self.player.draw(projection, view)
        self.crosshairMesh.draw()