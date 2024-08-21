import os.path

from MineClone.WorldRenderer import *
from MineClone.Player import *

class Game:
    def __init__(self):
        if os.path.exists("worldFile.bin"):
            with open("worldfile.bin", "rb") as f:
                self.world: World = World.deserialize(f.read())
        else:
            self.world: World = World()
        with open("worldfile.bin","wb") as f:
            f.write(self.world.serialize())
        self.world.update()

        self.worldRenderer: WorldRenderer = WorldRenderer(self.world)
        self.player: Player = Player(self.world, glm.vec3(0,5,0))

        self.crosshairMesh = CrosshairMesh(glm.vec2(0.5))

    @property
    def playerCam(self) -> Camera:
        return self.player.camera

    def update(self, deltaTime: float):
        self.player.update(deltaTime)
        if self.world.run_partial_update():
            self.worldRenderer.set_instance_data()
        self.worldRenderer.update_origin(self.player.pos)


    def draw(self, projection: glm.mat4, view: glm.mat4):
        self.worldRenderer.draw(projection, view)
        self.player.draw(projection, view)
        self.crosshairMesh.draw()