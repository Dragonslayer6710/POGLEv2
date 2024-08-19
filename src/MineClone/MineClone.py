from MineClone.WorldRenderer import *
from MineClone.Player import *

class Game:
    def __init__(self):
        self.world: World = World()
        self.world.update()
        self.worldRenderer: WorldRenderer = WorldRenderer(self.world)
        self.player: Player = Player(self.world, glm.vec3(0,5,0))

        self.crosshairMesh = CrosshairMesh(glm.vec2(0.5))
        self.wcCubeMesh = WireframeCubeMesh(glm.vec3())

    @property
    def playerCam(self) -> Camera:
        return self.player.camera

    def update(self, deltaTime: float):
        self.player.update(deltaTime)
        self.worldRenderer.update_origin(self.player.pos)


    def draw(self, projection: glm.mat4, view: glm.mat4):
        self.worldRenderer.draw(projection, view)
        self.wcCubeMesh.draw(projection, view)
        self.player.draw(projection, view)
        self.crosshairMesh.draw()