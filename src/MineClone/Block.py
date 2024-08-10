from POGLE.Core.Application import *
blockVertexLayout = VertexLayout(
    FloatVA.Vec3() # Position
)

blockInstanceLayout = VertexLayout([
    FloatVA.Vec3(), # Colour
    FloatVA.Mat4()  # ModelMatrix
])

class BlockMesh(QuadCubeMesh):
    def __init__(self, modelMatrix: glm.mat4, sideColours: list[glm.vec3]):
        super().__init__(Instances(
            [[model, color] for model, color in zip(QuadCubes([modelMatrix]), sideColours)],
            blockInstanceLayout)
        )


testBlockMesh = BlockMesh(
    NMM(glm.vec3(0.0, 0.0, -5.0)),
    [
        Color.RED,
        Color.GREEN,
        Color.BLUE,
        Color.YELLOW,
        Color.CYAN,
        Color.MAGENTA
    ]
)

testBlockShader = ShaderProgram("block")
