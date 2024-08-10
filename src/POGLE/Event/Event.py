from POGLE.Core.Core import *
class Event:
    class Type(Enum):
        Null = 0
        WindowClose = 1
        WindowResize = 2
        WindowFocus = 3
        WindowLostFocus = 4
        WindowMoved = 5
        AppTick = 6
        AppUpdate = 7
        AppRender = 8
        KeyPressed = 9
        KeyReleased = 10
        KeyTyped = 11
        MouseButtonPressed = 12
        MouseButtonReleased = 13
        MouseMoved = 14
        MouseScrolled = 15

    class Category(Enum):
        Null = 0,
        Application = BIT(0)
        Input = BIT(1)
        Keyboard = BIT(2)
        Mouse = BIT(3)
        MouseButton = BIT(4)

        def __or__(self, other):
            return self.value | other.value

        def __ror__(self, other):
            # if type(other) == int:
            #     return self.value | other
            return self.value | other

        def __and__(self, other):
            return self.value & other.value

        def __rand__(self, other):
            # if type(other) == int:
            #     return self.value | other
            return self.value & other

    type: Type
    category: Category
    def __init__(self):
        self.Handled = False

    def getEventType(self) -> Type:
        return self.type

    def getName(self) -> str:
        pass

    def getCategoryFlags(self) -> int:
        return self.category

    def toString(self) -> str:
        return self.getName()

    def isInCategory(self, category: Category) -> bool:
        return self.getCategoryFlags() & category

class EventDispatcher:
    def __init__(self, event: Event):
        self._Event: Event = event

    def Dispatch(self, typ, func):
        if self._Event.getEventType() == typ().type:
            self._Event.Handled |= func(self._Event)
            return True
        return False
