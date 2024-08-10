from MineClone.Block import *
from POGLE.Core.Application import *

class HelloLayer(Layer):

    def __init__(self, renderer: Renderer):
        super().__init__("Hello Layer!")
        # camera
        self.camera = Camera()

        self._Renderer = renderer

    def OnAttach(self):
        InitControls()
        self.initUpdate = True
        self.cursor_timer = 0
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

    def OnUpdate(self, deltaTime: float):
        if self.initUpdate:
            # defShader = ShaderProgram()
            # defShader.use()

            # quadModel, pentaModel = glm.mat4(1.0), glm.mat4(1.0)

            # quadModel = glm.translate(glm.scale(quadModel, glm.vec3(0.5)), glm.vec3(-1.0, 0.0, -5.0))
            # pentaModel = glm.translate(glm.scale(pentaModel, glm.vec3(0.5)), glm.vec3(1.0, 0.0, -5.0))

            # quadMesh = Mesh(Shapes.Quad, instances=Instances(interleave_arrays([quadModel])))
            # pentaMesh = Mesh(Shapes.Pentagon)

            # self.blockShader = ShaderProgram()
            # self.blockShader.use()
            testBlockShader.use()
            from random import randrange
            # self.qcMesh = QuadCubeMesh(QuadCubes([
            #     NMM(glm.vec3(randrange(-20, 20), randrange(-50, 0), randrange(-20, 20))) for i in
            #     range(randrange(1000, 3000))
            # ]))

            glClearColor(0.5, 0.3, 0.1, 1.0)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_BLEND)
            # wcMesh = WireframeCubeMesh()
            self.initUpdate = False
        self._Renderer.clear()
        projection = self._Renderer.get_projection()

        if self.cursor_timer > 0: self.cursor_timer -=1
        for boundCtrl in GetBoundControls():
            ctrlID = boundCtrl.GetID()
            if boundCtrl.GetInputState().value:
                if ctrlID in Control.ID.MoveCtrls:
                    self.camera.ProcessKeyboard(ctrlID, deltaTime)
                elif ctrlID == Control.ID.Config.CAM_CTRL_TGL:
                    if not self.cursor_timer:
                        self.camera.look_enabled(self.toggle_cam_control())
                        self.cursor_timer = 10
                elif ctrlID == Control.ID.Config.QUIT:
                    GetApplication().close()
        if self.camera.process_mouse:
            inpStat.s_MouseDeltaX = inpStat.s_NextMousePosX - inpStat.s_MousePosX
            inpStat.s_MouseDeltaY = inpStat.s_MousePosY - inpStat.s_NextMousePosY
            if bool(inpStat.s_MouseDeltaX) | bool(inpStat.s_MouseDeltaY) != 0:
                self.camera.ProcessMouseMovement(inpStat.s_MouseDeltaX, inpStat.s_MouseDeltaY)
                inpStat.s_MousePosX = inpStat.s_NextMousePosX
                inpStat.s_MousePosY = inpStat.s_NextMousePosY
        # self.blockShader.setMat4("uProjection", projection)
        # self.blockShader.setMat4("uView", self.camera.GetViewMatrix())
        testBlockShader.setMat4("uProjection", projection)
        testBlockShader.setMat4("uView", self.camera.GetViewMatrix())
        # defShader.setMat4("uModel", pentaModel)
        # pentaMesh.draw()
        # self.qcMesh.draw()
        # qcMeshB.draw()
        # wcMesh.draw()
        testBlockMesh.draw()

    def toggle_cam_control(self) -> bool:
        window = GetApplication().get_window()
        if window.is_cursor_hidden():
            window.reveal_cursor()
            return False
        else:
            window.hide_cursor()
            return True