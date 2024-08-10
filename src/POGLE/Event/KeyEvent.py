from POGLE.Event.Event import *
from POGLE.Input.KeyCodes import *

class KeyEvent(Event):
    category = Event.Category.Keyboard | Event.Category.Input
    def __init__(self, keycode: int):
        super().__init__()
        self._KeyCode: KeyCode = KeyCode(keycode)

    def getKeyCode(self) -> KeyCode:
        return self._KeyCode

class KeyPressedEvent(KeyEvent):
    type = Event.Type.KeyPressed

    def __init__(self, keycode: int, isRepeat:bool=False):
        super().__init__(keycode)
        self._IsRepeat: bool = isRepeat

    def isRepeat(self) -> bool:
        return self._IsRepeat

    def toString(self) -> str:
        return f"KeyPressedEvent: {self._KeyCode} (repeat = {self._IsRepeat})"

class KeyReleasedEvent(KeyEvent):
    type = Event.Type.KeyReleased

    def __init__(self, keycode: int):
        super().__init__(keycode)

    def toString(self) -> str:
        return f"KeyReleasedEvent: {self._KeyCode}"

class KeyTypedEvent(KeyEvent):
    type = Event.Type.KeyTyped

    def __init__(self, keycode: int):
        if 96 < keycode < 123:
            self._IsLower: bool = True
            keycode -= 32
        else:
            self._IsLower: bool = False

        super().__init__(keycode)

    def toString(self) -> str:
        if self._IsLower:
            return f"KeyTypedEvent: {self._KeyCode + 12}"
        else:
            return f"KeyTypedEvent: {self._KeyCode}"