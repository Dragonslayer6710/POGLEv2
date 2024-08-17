import random

from POGLE.Core.Application import *
from MineClone.MineClone import *
from MineClone.World import _WORLD_CHUNK_AXIS_LENGTH
game = None
class HelloLayer(Layer):
    def __init__(self, renderer: Renderer):
        super().__init__("Hello Layer!")
        self._Renderer = renderer

    def OnAttach(self):

        InitControls()
        self.initUpdate = True
        self.cursor_timer = 0
        self.tab_timer = 0
        self.chunk_shift_timer = 0

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
        pass

    import random
    def OnUpdate(self, deltaTime: float):
        global game
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
            testQCs = ColQuadCube(
                [
                    ColQuadCube.Instance(
                        NMM(
                            glm.vec3(
                                random.randrange(-50, 50),
                                random.randrange(-50, 50),
                                random.randrange(-50, 50)
                            )
                        ),
                        [colors[random.randrange(0, len(colors) - 1)] for i in range(6)],
                        [random.randrange(50, 100) / 100 for i in range(6)]
                    ) for i in range(0, random.randrange(0, 1000)+1)
                ]
            )
            testQC = ColQuadCube(
                NMM(glm.vec3(0.0, 0.0, -5.0)),
                [
                    Color.RED,
                    Color.GREEN,
                    Color.BLUE,
                    Color.YELLOW,
                    Color.CYAN,
                    Color.MAGENTA
                ],
                1.0
            )

            #testBlock = Block(NMM(glm.vec3(0,0,-5)))
            #testBlock.visibleSides[Block.Side.Top] = False
            #instance_data = testBlock.get_instance_data()
            game = Game()
            game.worldRenderer = game.worldRenderer
            #instance_data = testWorldRenderer.get_instance_data()

            #self.testBlockMesh = QuadCubeMesh(testQCs)
            #self.testBlockMesh = Mesh(Block.vertices, Block.indices, Block._TextureAtlas, Instances(instance_data, Block.instanceLayout, True))
            #self.testBlockShader = ShaderProgram("block", "block")
            #self.testBlockShader.use()

            game.worldRenderer.worldBlockShader.use()

            self.renderDistance = game.worldRenderer.renderDistance
            self.maxRenderDistance = _WORLD_CHUNK_AXIS_LENGTH
            self.minRenderDistance = 1
            self.renderDistanceRangeSize = self.maxRenderDistance - self.minRenderDistance + 1
            #self.testWorldRenderer._set_render_distance(self.renderDistance)

            self.crosshairMesh = CrosshairMesh(glm.vec2(0.5))
            self.wcCubeMesh = WireframeCubeMesh(glm.vec3())

            glClearColor(0.5, 0.3, 0.1, 1.0)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_BLEND)
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
            glFrontFace(GL_CW)

            # wcMesh = WireframeCubeMesh()
            self.initUpdate = False
        self._Renderer.clear()

        if self.cursor_timer > 0: self.cursor_timer -= 1
        if self.tab_timer > 0: self.tab_timer-=1
        if self.chunk_shift_timer > 0: self.chunk_shift_timer-=1
        for boundCtrl in GetBoundControls():
            ctrlID = boundCtrl.GetID()
            if boundCtrl.GetInputState().value:
                if ctrlID in Control.ID.MoveCtrls:
                    game.player.move(ctrlID, deltaTime)
                elif ctrlID == Control.ID.Config.CAM_CTRL_TGL:
                    if not self.cursor_timer:
                        game.playerCam.look_enabled(self.toggle_cam_control())
                        self.cursor_timer = 10
                elif ctrlID == Control.ID.Config.QUIT:
                    GetApplication().close()
                elif ctrlID == Control.ID.Config.CYCLE_RENDER_DISTANCE:
                    if not self.tab_timer:
                        self.renderDistance = (self.renderDistance + 1 - self.minRenderDistance) % self.renderDistanceRangeSize + self.minRenderDistance
                        game.worldRenderer._set_render_distance(self.renderDistance)
                        print(game.worldRenderer)
                        print(self.renderDistance + 1)
                        self.tab_timer = 20
                elif not self.chunk_shift_timer:
                    if ctrlID == Control.ID.Config.CHUNK_WEST:
                        game.worldRenderer._shift_rendered_chunks(glm.vec2(-1, 0))
                    elif ctrlID == Control.ID.Config.CHUNK_SOUTH:
                        game.worldRenderer._shift_rendered_chunks(glm.vec2(0, 1))
                    elif ctrlID == Control.ID.Config.CHUNK_EAST:
                        game.worldRenderer._shift_rendered_chunks(glm.vec2(1, 0))
                    elif ctrlID == Control.ID.Config.CHUNK_NORTH:
                        game.worldRenderer._shift_rendered_chunks(glm.vec2(0, -1))
                    print(game.worldRenderer)
                    self.chunk_shift_timer = 20

        if game.playerCam.process_mouse:
            inpStat.s_MouseDeltaX = inpStat.s_NextMousePosX - inpStat.s_MousePosX
            inpStat.s_MouseDeltaY = inpStat.s_MousePosY - inpStat.s_NextMousePosY
            if bool(inpStat.s_MouseDeltaX) | bool(inpStat.s_MouseDeltaY) != 0:
                game.playerCam.ProcessMouseMovement(inpStat.s_MouseDeltaX, inpStat.s_MouseDeltaY)
                inpStat.s_MousePosX = inpStat.s_NextMousePosX
                inpStat.s_MousePosY = inpStat.s_NextMousePosY
        # self.blockShader.setMat4("uProjection", projection)
        # self.blockShader.setMat4("uView", game.playerCam.GetViewMatrix())
        projection = self._Renderer.get_projection()
        view = game.playerCam.GetViewMatrix()
        # defShader.setMat4("uModel", pentaModel)
        # pentaMesh.draw()
        # self.qcMesh.draw()
        # qcMeshB.draw()
        # wcMesh.draw()
        game.worldRenderer.draw(projection, view)
        self.wcCubeMesh.draw(projection, view)
        self.crosshairMesh.draw()

    def toggle_cam_control(self) -> bool:
        window = GetApplication().get_window()
        if window.is_cursor_hidden():
            window.reveal_cursor()
            return False
        else:
            window.hide_cursor()
            return True
