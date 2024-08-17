from MineClone.WorldRenderer import *
from MineClone.Player import *

class Game:
    def __init__(self):
        self.world: World = World()
        self.world.update()
        self.worldRenderer: WorldRenderer = WorldRenderer(self.world)
        self.player: Player = Player(glm.vec3(0,2,0))

    @property
    def playerCam(self) -> Camera:
        return self.player.camera

