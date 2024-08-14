from POGLE.Input.Input import *

class _CtrlIDMove(Enum):
    FORWARD = auto()
    BACKWARD = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()

class _CtrlIDConfig(Enum):
    CAM_CTRL_TGL = len(_CtrlIDMove)
    QUIT = auto()
    CYCLE_RENDER_DISTANCE = auto()
class Control:
    class ID:
        Move = _CtrlIDMove
        MoveCtrls = [ctrl for ctrl in Move]
        Config = _CtrlIDConfig
        CfgCtrls = [ctrl for ctrl in Config]
    class Type(Enum):
        MOVEMENT = auto()
        CAMERA = auto()
        CONFIG = auto()

    class State(Enum):
        UNBOUND = 0
        BOUND = 1

    @staticmethod
    def New(ctrlID: ID):
        _NewControl(Control(ctrlID))

    @staticmethod
    def Get(ctrlID: ID):
        if ctrlID not in _Controls.keys():
            Control.New(ctrlID)
        return _Controls[ctrlID]

    def BindInput(self, input: Input):
        self._BoundInput = input
        self._UpdateControlBind()

    def UnbindInput(self):
        self.BindInput(None)

    def GetID(self) -> ID:
        return self._ControlID

    def GetType(self) -> Type:
        return _ControlTypes[self._ControlID]

    def GetInputState(self) -> Input.State:
        return self._BoundInput.InstGetState()

    def __init__(self, ctrlID: ID, inp: Input = None):
        self._ControlID = ctrlID
        self._InitControl()
        if inp:
            BindInput()

    def _InitControl(self):
        _NewControl(self)

    def _UpdateControlBind(self):
        if self._BoundInput:
            _BoundControls.append(self)
        else:
            if self in _BoundControls:
                _BoundControls.remove(self)


CTRL = Control

_Controls: dict[Control.ID, Control] = None
_BoundControls: list[Control] = None
_ControlTypes: dict[Control.ID, Control.Type] = {
    CTRL.ID.Move.FORWARD: CTRL.Type.MOVEMENT,
    CTRL.ID.Move.BACKWARD: CTRL.Type.MOVEMENT,
    CTRL.ID.Move.LEFT: CTRL.Type.MOVEMENT,
    CTRL.ID.Move.RIGHT: CTRL.Type.MOVEMENT,
    CTRL.ID.Move.UP: CTRL.Type.MOVEMENT,
    CTRL.ID.Move.DOWN: CTRL.Type.MOVEMENT,

    CTRL.ID.Config.CAM_CTRL_TGL: CTRL.Type.CONFIG,
    CTRL.ID.Config.QUIT: CTRL.Type.CONFIG,
    CTRL.ID.Config.CYCLE_RENDER_DISTANCE: CTRL.Type.CONFIG
}


def ResetControls(initialControls: dict[Control.ID, int] = None):
    ResetInputs()
    if initialControls:
        for ctrlID, inputID in initialControls.items():
            BindInput(ctrlID=ctrlID, inputID=inputID)


def _InitControls(initialControls: dict[Control.ID, int] = None):
    global _Controls, _BoundControls
    _Controls = {}
    _BoundControls = []
    ResetControls(initialControls)


def _NewControl(ctrl: Control):
    _Controls[ctrl.GetID()] = ctrl


_InitialControls: dict[Control.ID, int] = {
    CTRL.ID.Move.FORWARD: KeyCode.W,
    CTRL.ID.Move.BACKWARD: KeyCode.S,
    CTRL.ID.Move.LEFT: KeyCode.A,
    CTRL.ID.Move.RIGHT: KeyCode.D,
    CTRL.ID.Move.UP: KeyCode.Space,
    CTRL.ID.Move.DOWN: KeyCode.LeftControl,

    CTRL.ID.Config.CAM_CTRL_TGL: MouseCode.ButtonLeft,
    CTRL.ID.Config.QUIT: KeyCode.Escape,
    CTRL.ID.Config.CYCLE_RENDER_DISTANCE: KeyCode.Tab
}


def InitControls(initialControls: dict[Control.ID, int] = _InitialControls):
    _InitControls(initialControls)


def GetBoundControls() -> list[Control]:
    return _BoundControls


def BindInput(ctrl: Control = None, inp: Input = None, ctrlID: Control.ID = None, inputID: int = None, ):
    if inp == ctrl == None:
        inp = Input.Get(inputID)
        ctrl = Control.Get(ctrlID)
        BindInput(ctrl, inp)
    else:
        ctrl.BindInput(inp)
