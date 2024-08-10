from POGLE.Event.Event import *

class WindowResizeEvent(Event):
    type = Event.Type.WindowResize
    category = Event.Category.Application

    def __init__(self, width: np.uint32=0, height: np.uint32=0):
        super().__init__()
        self._Width = width
        self._Height = height

    def getWidth(self) -> np.uint32:
        return self._Width

    def getHeight(self) -> np.uint32:
        return self._Height

    def toString(self) -> str:
        return f"WindowResizeEvent: {self._Width}, {self._Height}"

class WindowCloseEvent(Event):
    type = Event.Type.WindowClose
    category = Event.Category.Application

    def __init__(self):
        super().__init__()

class AppTickEvent(Event):
    type = Event.Type.AppTick
    category = Event.Category.Application

    def __init__(self):
        super().__init__()


class AppUpdateEvent(Event):
    type = Event.Type.AppUpdate
    category = Event.Category.Application

    def __init__(self):
        super().__init__()

class AppRenderEvent(Event):
    type = Event.Type.AppRender
    category = Event.Category.Application

    def __init__(self):
        super().__init__()