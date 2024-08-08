
# import OpenGL.GL
# import glfw

from OpenGL.GL import *
from glfw.GLFW import *

from glfw import _GLFWwindow as GLFWwindow

import glm
import numpy as np

from enum import Enum

class Color:
    BLACK   = glm.vec3(0.0)
    RED     = glm.vec3(1.0, 0.0, 0.0)
    GREEN   = glm.vec3(0.0, 1.0, 0.0)
    BLUE    = glm.vec3(0.0, 0.0, 1.0)
    MAGENTA = glm.vec3(1.0, 0.0, 1.0)
    YELLOW  = glm.vec3(1.0, 1.0, 0.0)
    CYAN    = glm.vec3(0.0, 1.0, 1.0)
    WHITE   = glm.vec3(1.0)
