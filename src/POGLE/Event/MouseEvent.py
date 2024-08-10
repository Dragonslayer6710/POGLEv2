from POGLE.Event.Event import *
from POGLE.Input.MouseCodes import *


class MouseEvent(Event):
    category = Event.Category.Mouse | Event.Category.Input

    def __init__(self):
        super().__init__()


class MouseMovedEvent(MouseEvent):
    type = Event.Type.MouseMoved

    def __init__(self, x: float, y: float):
        super().__init__()
        self._MouseX: float = x
        self._MouseY: float = y

    def getX(self) -> float:
        return self._MouseX

    def getY(self) -> float:
        return self._MouseY

    def toString(self) -> str:
        return f"MouseMovedEvent: {self._MouseX}, {self._MouseY}"


class MouseScrolledEvent(MouseEvent):
    type = Event.Type.MouseScrolled

    def __init__(self, x: float, y: float):
        super().__init__()
        self._XOffset: float = x
        self._YOffset: float = y

    def getXOffset(self) -> float:
        return self._XOffset

    def getYOffset(self) -> float:
        return self._YOffset

    def toString(self) -> str:
        return f"MouseScrolledEvent: {self._XOffset}, {self._YOffset}"


class MouseButtonEvent(MouseEvent):
    def __init__(self, button: int):
        super().__init__()
        self._Button: MouseCode = MouseCode(button)
        self.category |= Event.Category.MouseButton

    def getMouseButton(self) -> MouseCode:
        return self._Button


class MouseButtonPressedEvent(MouseButtonEvent):
    type = Event.Type.MouseButtonPressed

    def __init__(self, button: int, isRepeat: bool = False):
        super().__init__(button)
        self._IsRepeat: bool = isRepeat

    def isRepeat(self) -> bool:
        return self._IsRepeat

    def toString(self) -> str:
        return f"MouseButtonPressedEvent: {self._Button} (repeat = {self._IsRepeat})"


class MouseButtonReleasedEvent(MouseButtonEvent):
    type = Event.Type.MouseButtonReleased

    def __init__(self, button: int):
        super().__init__(button)

    def toString(self) -> str:
        return f"MouseButtonReleasedEvent: {self._Button}"
