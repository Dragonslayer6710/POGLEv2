from POGLE.Input.Input import *

class _CtrlIDMove(Enum):
    FORWARD = 0
    BACKWARD = auto()
    RIGHT = auto()
    LEFT = auto()
    UP = auto()
    DOWN = auto()

class _CtrlIDConfig(Enum):
    CAM_CTRL_TGL = len(_CtrlIDMove)
    QUIT = auto()
    CYCLE_RENDER_DISTANCE = auto()
    CHUNK_WEST = auto()
    CHUNK_SOUTH = auto()
    CHUNK_EAST = auto()
    CHUNK_NORTH = auto()
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

    _idTypeMap: dict[ID, Type] = {
        ID.Move.FORWARD: Type.MOVEMENT,
        ID.Move.BACKWARD: Type.MOVEMENT,
        ID.Move.LEFT: Type.MOVEMENT,
        ID.Move.RIGHT: Type.MOVEMENT,
        ID.Move.UP: Type.MOVEMENT,
        ID.Move.DOWN: Type.MOVEMENT,

        ID.Config.CAM_CTRL_TGL: Type.CONFIG,
        ID.Config.QUIT: Type.CONFIG,

        ID.Config.CYCLE_RENDER_DISTANCE: Type.CONFIG,
        ID.Config.CHUNK_WEST: Type.CONFIG,
        ID.Config.CHUNK_SOUTH: Type.CONFIG,
        ID.Config.CHUNK_EAST: Type.CONFIG,
        ID.Config.CHUNK_NORTH: Type.CONFIG
    }

    _InitialBinds: dict[ID, int] = {
        ID.Move.FORWARD: KeyCode.W,
        ID.Move.BACKWARD: KeyCode.S,
        ID.Move.LEFT: KeyCode.A,
        ID.Move.RIGHT: KeyCode.D,
        ID.Move.UP: KeyCode.Space,
        ID.Move.DOWN: KeyCode.LeftControl,

        ID.Config.CAM_CTRL_TGL: MouseCode.ButtonLeft,
        ID.Config.QUIT: KeyCode.Escape,

        ID.Config.CYCLE_RENDER_DISTANCE: KeyCode.Tab,
        ID.Config.CHUNK_WEST: KeyCode.KP4,
        ID.Config.CHUNK_SOUTH: KeyCode.KP2,
        ID.Config.CHUNK_EAST: KeyCode.KP6,
        ID.Config.CHUNK_NORTH: KeyCode.KP8,
    }

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
        return Control._idTypeMap[self._ControlID]

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

def InitControls(initialControls: dict[Control.ID, int] = Control._InitialBinds):
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
