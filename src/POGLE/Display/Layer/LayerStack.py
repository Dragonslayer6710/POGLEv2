from POGLE.Display.Layer.Layer import *

class LayerStack:
    def __init__(self):
        self._Layers: list[Layer] = []
        self._IterPointer = -1
        self._LayerInsertIndex = 0

    def __del__(self):
        for layer in self:
            layer.OnDetatch()
            del layer

    def pushLayer(self, layer: Layer):
        self._Layers.insert(self._LayerInsertIndex, layer)
        self._LayerInsertIndex += 1

    def pushOverlay(self, overlay: Layer):
        self._Layers.append(overlay)

    def popLayer(self, layer: Layer):
        index = self._Layers.index(layer)
        if index != self._LayerInsertIndex:
            layer.OnDetatch()
            self._Layers.pop(index)
            self._LayerInsertIndex -= 1

    def PopLayer(self, overlay: Layer):
        index = self._Layers.index(overlay)
        if index != len(self._Layers):
            overlay.OnDetach()
            self._Layers.pop(-1)

    def __iter__(self):
        return self

    def __next__(self):
        self._IterPointer += 1
        if self._IterPointer < len(self._Layers):
            return self._Layers[self._IterPointer]
        else:
            self._IterPointer = -1
            raise StopIteration

    def _RetLayer(self):
        return self._Layers[self._IterPointer]