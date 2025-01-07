from __future__ import annotations
from POGLE.Input.Input import *


class _CtrlIDMove(Enum):
    FORWARD = 0
    BACKWARD = auto()
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    DOWN = auto()


lastMove = list(_CtrlIDMove.__dict__.items())[-2][1].value


class _CtrlIDAction(Enum):
    SPRINT = lastMove + 1
    ATTACK = auto()
    INTERACT = auto()


lastAction = list(_CtrlIDAction.__dict__.items())[-2][1].value


class _CtrlIDConfig(Enum):
    CAM_CTRL_TGL = lastAction + 1
    QUIT = auto()
    CYCLE_RENDER_DISTANCE = auto()


class Control:
    class ID:
        Move = _CtrlIDMove
        _MoveCtrls = [ctrl for ctrl in Move]
        Action = _CtrlIDAction
        _ActionCtrls = [ctrl for ctrl in Action]
        Config = _CtrlIDConfig
        _CfgCtrls = [ctrl for ctrl in Config]


    class Type(Enum):
        MOVEMENT = 0
        CAMERA = auto()
        ACTION = auto()
        CONFIG = auto()

    _idTypeMap: dict[ID, Type] = {
        ID.Move.FORWARD: Type.MOVEMENT,
        ID.Move.BACKWARD: Type.MOVEMENT,
        ID.Move.LEFT: Type.MOVEMENT,
        ID.Move.RIGHT: Type.MOVEMENT,
        ID.Move.UP: Type.MOVEMENT,
        ID.Move.DOWN: Type.MOVEMENT,

        ID.Action.SPRINT: Type.ACTION,
        ID.Action.ATTACK: Type.ACTION,
        ID.Action.INTERACT: Type.ACTION,

        ID.Config.CAM_CTRL_TGL: Type.CONFIG,
        ID.Config.QUIT: Type.CONFIG,

        ID.Config.CYCLE_RENDER_DISTANCE: Type.CONFIG,
    }

    _InitialBinds: dict[ID, int] = {
        ID.Move.FORWARD: KeyCode.W,
        ID.Move.BACKWARD: KeyCode.S,
        ID.Move.LEFT: KeyCode.A,
        ID.Move.RIGHT: KeyCode.D,
        ID.Move.UP: KeyCode.Space,
        ID.Move.DOWN: KeyCode.LeftControl,

        ID.Action.SPRINT: KeyCode.LeftShift,
        ID.Action.ATTACK: MouseCode.ButtonLeft,
        ID.Action.INTERACT: MouseCode.ButtonRight,

        ID.Config.CAM_CTRL_TGL: KeyCode.Escape,
        ID.Config.QUIT: KeyCode.Q,

        ID.Config.CYCLE_RENDER_DISTANCE: KeyCode.Tab,
    }

    class State(Enum):
        UNBOUND = 0
        BOUND = 1

    @staticmethod
    def New(controls: ControlSet, ctrlID: ID):
        controls._NewControl(Control(controls, ctrlID))

    @staticmethod
    def Get(controls: ControlSet, ctrlID: ID):
        if ctrlID not in controls._Controls.keys():
            Control.New(controls, ctrlID)
        return controls._Controls[ctrlID]

    def BindInput(self, controls: ControlSet, input: Input):
        self._BoundInput = input
        self._UpdateControlBind(controls)

    def UnbindInput(self, controls: ControlSet):
        self.BindInput(controls, None)

    def GetID(self) -> ID:
        return self._ControlID

    def GetType(self) -> Type:
        return Control._idTypeMap[self._ControlID]

    def GetInputState(self) -> Input.State:
        return self._BoundInput.InstGetState()

    def __init__(self, controls: ControlSet, ctrlID: ID, inp: Input = None):
        self._ControlID = ctrlID
        self._InitControl(controls)
        if inp:
            controls.BindInput()

    def _InitControl(self, controls: ControlSet):
        controls._NewControl(self)

    def _UpdateControlBind(self, controls: ControlSet):
        if self._BoundInput:
            controls._BoundControls.append(self)
        else:
            if self in controls._BoundControls:
                controls._BoundControls.remove(self)


CTRL = Control


class ControlSet:
    _Controls: dict[Control.ID, Control] = None
    _BoundControls: list[Control] = None

    def __init__(self, initialControls: dict[Control.ID, int] = Control._InitialBinds):
        self._Controls = {}
        self._BoundControls = []
        self.ResetControls(initialControls)

    def ResetControls(self, initialControls: dict[Control.ID, int] = None):
        ResetInputs()
        if initialControls:
            for ctrlID, inputID in initialControls.items():
                self.BindInput(ctrlID=ctrlID, inputID=inputID)

    def _NewControl(self, ctrl: Control):
        self._Controls[ctrl.GetID()] = ctrl

    def GetBoundControls(self) -> list[Control]:
        return self._BoundControls

    def BindInput(self, ctrl: Control = None, inp: Input = None, ctrlID: Control.ID = None, inputID: int = None, ):
        if inp == ctrl == None:
            inp = Input.Get(inputID)
            ctrl = Control.Get(self, ctrlID)
            self.BindInput(ctrl, inp)
        else:
            ctrl.BindInput(self, inp)