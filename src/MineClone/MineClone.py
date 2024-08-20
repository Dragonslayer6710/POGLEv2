from MineClone.WorldRenderer import *
from MineClone.Player import *

class Game:
    def __init__(self):
        self.world: World = World()
        self.world.update()
        start_pos = glm.vec3(0,5.12,0)
        end_pos = glm.vec3(-0.837441,0.620986,-0.0432235)
        ray = Ray.from_start_end(start_pos, end_pos)
        print(ray)
        hitBlocks: set[Block] = self.world.query_segment_blocks(ray)
        for block in hitBlocks:
            nearHit, farHit = block.bounds.intersectSegment(ray)
            print(f"Near: {nearHit.time}, Far: {farHit.time}")
        print("[\n\t"+",\n\t".join([str(block) for block in hitBlocks])+"\n]")
        #quit()
        self.worldRenderer: WorldRenderer = WorldRenderer(self.world)
        self.player: Player = Player(self.world, glm.vec3(0,5,0))

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