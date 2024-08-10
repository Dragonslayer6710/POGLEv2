from POGLE.Core.Application import *

testQC = QuadCube(
    NMM(glm.vec3(0.0, 0.0, -5.0)),
    [
        Color.RED,
        Color.GREEN,
        Color.BLUE,
        Color.YELLOW,
        Color.CYAN,
        Color.MAGENTA
    ],
    1.0
)

testBlockMesh = QuadCubeMesh(testQC)
testBlockShader = ShaderProgram("block")
