from POGLE.Display.Window import *
from POGLE.Display.Layer.LayerStack import *
from POGLE.Input.Control import *

class _Application:
    pass

def _GetApp() -> _Application:
    return _Instance

def GetApplication() -> _Application:
    return _GetApp()

from POGLE.Display.Layer.ImGui.ImGuiLayer import *


import os
import threading

class ApplicationCommandLineArgs:
    Count: int
    Args: list[str]

    def __init__(self, args: list[str]):
        self.Count = len(args)
        self.Args = args

    def __getitem__(self, index):
        return self.Args[index]

class ApplicationSpecification:
    Name: str
    WorkingDirectory: str
    CommandLineArgs: ApplicationCommandLineArgs

    def __init__(self, commandLineArgs: ApplicationCommandLineArgs, name="", workingDirectory=""):
        self.Name = name
        self.WorkingDirectory = workingDirectory
        self.CommandLineArgs = commandLineArgs

def ExampleFunction():
    pass

class Application:
    def __init__(self, spec: ApplicationSpecification):
        global _Instance
        self._ImGuiLayer = None
        self._Running = False
        self._Minimized = None
        self._LastFrameTime = 0.0

        #TODO: ensure this is correct
        self._MainThreadQueue: list[type(ExampleFunction)] = []
        self._MainThreadQueueMutex: threading.Lock = threading.Lock()

        self._Specification: ApplicationSpecification = spec

        _Instance = self
        if not self._Specification.WorkingDirectory == "":
            os.chdir(self._Specification.WorkingDirectory)

        self._Window: Window = Window(WindowProps(self._Specification.Name))
        self._Window.set_event_callback(POGLE_BIND_EVENT_FN(self.on_event))

        self._Renderer: Renderer = Renderer(self._Window.get_width(), self._Window.get_height())

        self._LayerStack = LayerStack()

        self._ImGuiLayer: ImGuiLayer = ImGuiLayer()
        self.push_overlay(self._ImGuiLayer)

    def on_event(self, e: Event):
        dispatcher = EventDispatcher(e)
        dispatcher.Dispatch(WindowCloseEvent, POGLE_BIND_EVENT_FN(self.on_window_close))
        dispatcher.Dispatch(WindowResizeEvent, POGLE_BIND_EVENT_FN(self.on_window_resize))

        for layer in self._LayerStack:
            if e.Handled:
                break
            layer.OnEvent(e)

    def push_layer(self, layer: Layer):
        self._LayerStack.pushLayer(layer)
        layer.OnAttach()

    def push_overlay(self, overlay: Layer):
        self._LayerStack.pushOverlay(overlay)
        overlay.OnAttach()

    def get_window(self) -> Window:
        return self._Window

    def open(self):
        self._Running: bool = True
        self._Run()

    def close(self):
        self._Running: bool = False

    def get_imgui_layer(self):
        return self._ImGuiLayer

    def get_specification(self) -> ApplicationSpecification:
        return self._Specification

    def submit_to_main_thread(self, func: type(ExampleFunction)):
        with self._MainThreadQueueMutex:
            self._MainThreadQueue.append(func)

    def get_renderer(self) -> Renderer:
        return self._Renderer

    def _Run(self):
        while self._Running:
            time = self._Window.get_time()
            deltaTime = time - self._LastFrameTime
            refreshRate = glfwGetVideoMode(self._Window.get_monitor()).refresh_rate
            deltaTime = clamp(deltaTime, 0, 1/refreshRate*2)
            self._Window.show_fps(time, deltaTime)

            self._LastFrameTime = time

            self.execute_main_thread_queue()

            if not self._Minimized:
                for layer in self._LayerStack:
                    layer.OnUpdate(deltaTime)

                self._ImGuiLayer.Begin()
                for layer in self._LayerStack:
                    layer.OnImGuiRender()
                self._ImGuiLayer.End()

            self._Window.on_update()

    def on_window_close(self, e: WindowCloseEvent) -> bool:
        self._Running = False
        return True

    def on_window_resize(self, e: WindowResizeEvent) -> bool:
        width, height = e.getWidth(), e.getHeight()
        if width | height == 0:
            self._Minimized: bool = True
            return False

        self._Minimized = False
        self._Renderer.on_window_resize(width, height)

        return False


    def execute_main_thread_queue(self):
        with self._MainThreadQueueMutex:
            for func in self._MainThreadQueue:
                func()
            self._MainThreadQueue.clear()

_Instance: Application = None

_Application = Application