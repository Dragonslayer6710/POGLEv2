import ctypes

import glfw

from POGLE.Core.Application import *

from imgui_bundle import imgui, imguizmo

from POGLE.Display.Layer.Layer import *

ImGuizmo = imguizmo.im_guizmo

class ImGuiLayer(Layer):
    def __init__(self):
        super().__init__("ImGuiLayer")
        self._BlockEvents: bool = True

    def __del__(self):
        pass

    def OnAttach(self):
        imgui.create_context()
        io = imgui.get_io()
        io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard
        io.config_flags |= imgui.ConfigFlags_.docking_enable
        io.config_flags |= imgui.ConfigFlags_.viewports_enable

        fontSize = 18.0
        io.fonts.add_font_from_file_ttf("../assets/fonts/opensans/OpenSans-Bold.ttf", fontSize)
        io.font_default = io.fonts.add_font_from_file_ttf("../assets/fonts/opensans/OpenSans-Regular.ttf", fontSize)

        imgui.style_colors_dark()

        style = imgui.get_style()
        if io.config_flags and imgui.ConfigFlags_.viewports_enable:
            style.window_rounding = 0.0
            style.color_(imgui.Col_.window_bg).w = 1.0

        self.SetDarkThemeColors()
        imgui.backends.glfw_init_for_opengl(ctypes.addressof(GetApplication().get_window().get_native_window().contents), True)
        imgui.backends.opengl3_init("#version 450")

    def OnDetach(self):
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()

    def OnEvent(self, event: Event):
        if self._BlockEvents:

            io = imgui.get_io()
            window: Window = GetApplication().get_window()

            if io.want_capture_mouse:
                window.set_imgui_layer_block(True)
                if event.getEventType() == Event.Type.MouseButtonReleased:
                    window.reveal_cursor()
                inpStat.s_NextMousePosX = inpStat.s_MousePosX
                inpStat.s_NextMousePosY = inpStat.s_MousePosY

            event.Handled |= event.isInCategory(Event.Category.Mouse) and io.want_capture_mouse
            event.Handled |= event.isInCategory(Event.Category.Keyboard) and io.want_capture_keyboard

            if io.want_capture_mouse:
                if event.getEventType() == Event.Type.MouseButtonPressed:
                    window.hide_cursor()
                window.set_imgui_layer_block(False)

    def Begin(self):
        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()
        ImGuizmo.begin_frame()

    def End(self):
        io = imgui.get_io()
        window: Window = GetApplication().get_window()
        io.display_size = imgui.ImVec2(window.get_width(), window.get_height())

        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())

        if io.config_flags and imgui.ConfigFlags_.viewports_enable:
            backup_current_context = glfw.get_current_context()
            imgui.update_platform_windows()
            imgui.render_platform_windows_default()
            glfw.make_context_current(backup_current_context)

    def BlockEvents(self, block: bool):
        self._BlockEvents = block

    def SetDarkThemeColors(self):
        style = imgui.get_style()
        style.set_color_(imgui.Col_.window_bg, imgui.ImVec4(0.1, 0.105, 0.11, 1.0))

        # Headers
        style.set_color_(imgui.Col_.header, imgui.ImVec4(0.2, 0.205, 0.21, 1.0))
        style.set_color_(imgui.Col_.header_hovered, imgui.ImVec4(0.3, 0.305, 0.31, 1.0))
        style.set_color_(imgui.Col_.header_active, imgui.ImVec4(0.15, 0.1505, 0.151, 1.0))

        # Buttons
        style.set_color_(imgui.Col_.button, imgui.ImVec4(0.2, 0.205, 0.21, 1.0))
        style.set_color_(imgui.Col_.button_hovered, imgui.ImVec4(0.3, 0.305, 0.31, 1.0))
        style.set_color_(imgui.Col_.button_active, imgui.ImVec4(0.15, 0.1505, 0.151, 1.0))

        # Frame BG
        style.set_color_(imgui.Col_.frame_bg, imgui.ImVec4(0.2, 0.205, 0.21, 1.0))
        style.set_color_(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.3, 0.305, 0.31, 1.0))
        style.set_color_(imgui.Col_.frame_bg_active, imgui.ImVec4(0.15, 0.1505, 0.151, 1.0))

        # Tabs
        style.set_color_(imgui.Col_.tab, imgui.ImVec4(0.15, 0.1505, 0.151, 1.0))
        style.set_color_(imgui.Col_.tab_hovered, imgui.ImVec4(0.38, 0.3805, 0.381, 1.0))
        style.set_color_(imgui.Col_.tab_active, imgui.ImVec4(0.28, 0.2805, 0.281, 1.0))
        style.set_color_(imgui.Col_.tab_unfocused, imgui.ImVec4(0.15, 0.1505, 0.151, 1.0))
        style.set_color_(imgui.Col_.tab_unfocused_active, imgui.ImVec4(0.2, 0.205, 0.21, 1.0))

        # Title
        style.set_color_(imgui.Col_.title_bg, imgui.ImVec4(0.15, 0.1505, 0.151, 1.0))
        style.set_color_(imgui.Col_.title_bg_active, imgui.ImVec4(0.15, 0.1505, 0.151, 1.0))
        style.set_color_(imgui.Col_.title_bg_collapsed, imgui.ImVec4(0.15, 0.1505, 0.151, 1.0))

    def GetActiveWidgetID(self) -> int:
        return imgui.get_current_context().active_id