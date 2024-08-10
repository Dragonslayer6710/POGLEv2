from POGLE.Event.Event import *
class Layer:

    def __init__(self, name: str = "Layer"):
        self._DebugName = ""

    def __del__(self):
        pass

    def OnAttach(self):
        pass

    def OnDetach(self):
        pass

    def OnUpdate(self):
        pass

    def OnImGuiRender(self):
        pass

    def OnEvent(self, event: Event):
        pass

    def getName(self) -> str:
        return self._DebugName