from POGLE.Input.KeyCodes import *
from POGLE.Input.MouseCodes import *
from enum import auto

class Input:
    class State(Enum):
        RELEASE = 0
        PRESS = 1

    class Type(Enum):
        KEY = 0
        MOUSE = 1

    _InputID: Enum
    _InputType: Type
    _KeyCode: KeyCode = None
    _MouseButton: MouseCode = None

    @staticmethod
    def New(inputID: Enum):
        if inputID.value < 8:
            _NewInput(Input(inputID, Input.Type.MOUSE))
        else:
            _NewInput(Input(inputID, Input.Type.KEY))

    @staticmethod
    def Get(inputID: Enum):
        if inputID not in _Inputs.keys():
            Input.New(inputID)
        return _Inputs[inputID]

    @staticmethod
    def SetState(inputID: Enum, state: State):
        Input.Get(inputID).InstSetState(state)

    @staticmethod
    def GetState(inputID: Enum, state: State) -> State:
        return Input.Get(inputID).InstGetState()

    def GetID(self) -> Enum:
        return self._InputID

    def InstSetState(self, state: State):
        _InputStates[self._InputID] = state

    def InstGetState(self) -> State:
        return _InputStates[self._InputID]

    def GetType(self) -> Type:
        return self._InputType

    def __init__(self, inputID: Enum, inputType: Type):
        self._InputID = inputID
        self._InputType = inputType

        if self._InputType.value:
            self._MouseButton = MouseCode(self._InputID)
        else:
            self._KeyCode = KeyCode(self._InputID)

        self.InstSetState(Input.State.RELEASE)


_Inputs: dict[Enum, Input] = None
_InputStates: dict[Enum, Input.State] = None


def _InitInputs():
    global _Inputs, _InputStates
    _Inputs = {}
    _InputStates = {}


def ResetInputs():
    _InitInputs()


def _NewInput(inp: Input):
    _Inputs[inp.GetID()] = inp

class InputStatics:
    s_MousePosX: float = 0
    s_MousePosY: float = 0
    s_NextMousePosX: float = 0
    s_NextMousePosY: float = 0
    s_MouseDeltaX: float = 0
    s_MouseDeltaY: float = 0
    s_ScrollOffsetX: float = 0
    s_ScrollOffsetY: float = 0

inpStat = InputStatics()
