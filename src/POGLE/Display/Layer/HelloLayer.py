import random

from POGLE.Core.Application import *
from MineClone.Game import *
game: Game = None
class HelloLayer(Layer):
    def __init__(self, renderer: Renderer):
        super().__init__("Hello Layer!")
        self._Renderer = renderer

    def OnAttach(self):
        global game
        game = Game()

        self.initUpdate = True


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

    def OnDetach(self):
        pass

    def OnImGuiRender(self):
        playerFeetPos = game.player.feetPos
        imgui.input_float3("Player Pos:", playerFeetPos - glm.vec3(0, 0.5, 0))
        imgui.label_text("Target Block:", str(game.player.targetBlock))

    import random
    def OnUpdate(self, deltaTime: float):
        if self.initUpdate:
            colors = [
                Color.BLACK,
                Color.RED,
                Color.GREEN,
                Color.BLUE,
                Color.MAGENTA,
                Color.YELLOW,
                Color.CYAN,
                Color.WHITE
            ]

            #testBlock = Block(NMM(glm.vec3(0,0,-5)))
            #testBlock.visibleSides[Block.Side.Top] = False
            #instance_data = testBlock.get_instance_data()
            #instance_data = testWorldRenderer.get_instance_data()

            #self.testBlockMesh = QuadCubeMesh(testQCs)
            #self.testBlockMesh = Mesh(Block.vertices, Block.indices, Block._TextureAtlas, Instances(instance_data, Block.instanceLayout, True))
            #self.testBlockShader = ShaderProgram("block", "block")
            #self.testBlockShader.use()

            # self.renderDistance = game.worldRenderer._render_distance
            self.maxRenderDistance = 1#_WORLD_CHUNK_AXIS_LENGTH
            self.minRenderDistance = 1
            self.renderDistanceRangeSize = self.maxRenderDistance - self.minRenderDistance + 1
            #self.testWorldRenderer._set_render_distance(self.renderDistance)


            glClearColor(0.5, 0.3, 0.1, 1.0)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_BLEND)
            # glEnable(GL_CULL_FACE)
            # glCullFace(GL_BACK)
            # glFrontFace(GL_CCW)
            # wcMesh = WireframeCubeMesh()
            self.initUpdate = False
        self._Renderer.clear()

        if game.playerCam.process_mouse:
            inpStat.s_MouseDeltaX = inpStat.s_NextMousePosX - inpStat.s_MousePosX
            inpStat.s_MouseDeltaY = inpStat.s_MousePosY - inpStat.s_NextMousePosY
            if bool(inpStat.s_MouseDeltaX) | bool(inpStat.s_MouseDeltaY) != 0:
                game.playerCam.ProcessMouseMovement(inpStat.s_MouseDeltaX, inpStat.s_MouseDeltaY)
                inpStat.s_MousePosX = inpStat.s_NextMousePosX
                inpStat.s_MousePosY = inpStat.s_NextMousePosY
        # self.blockShader.setMat4("uProjection", projection)
        # self.blockShader.setMat4("uView", game.playerCam.GetViewMatrix())
        projection = game.playerCam.get_projection()
        view = game.playerCam.get_view_matrix()
        game.update(deltaTime, projection, view)
        game.draw()
