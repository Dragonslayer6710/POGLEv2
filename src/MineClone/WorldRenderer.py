from MineClone.World import *
from MineClone.World import _WORLD_SIZE

class WorldRenderer:
    def __init__(self, world: World, originPos: glm.vec3 = glm.vec3(0, _WORLD_SIZE.y, 0), renderDistance: int = 1):
        self.world: World = world
        self.originPos: glm.vec3 = originPos
        self.originChunk: Chunk = None
        self.renderDistance: int = None
        self._chunksInRow: int = None
        self.renderedChunks: list[list[Chunk]] = None
        self.chunk_instances: list = None

        self.worldBlockShader: ShaderProgram = ShaderProgram("block", "block")
        self.worldMesh: Mesh = None

        self._set_origin_chunk()
        self._set_render_distance(renderDistance)

    def _get_origin_chunk(self) -> Chunk:
        return self.world.get_chunk_from_world_block_pos(self.originPos)

    def _set_origin_chunk(self):
        self.originChunk: Chunk = self._get_origin_chunk()

    def update_origin(self, originPos: glm.vec3):
        self.originPos = originPos
        if not self.originChunk.is_pos_in_chunk(originPos):
            lastOrigin: glm.vec2 = self.originChunk.worldChunkPos
            self._set_origin_chunk()
            originDelta: glm.vec2 = self.originChunk.worldChunkPos - lastOrigin
            absDelta = glm.abs(originDelta)
            if absDelta[0] <= 1:
                if absDelta[1] <= 1:
                    self._shift_rendered_chunks(originDelta)
                    return
            self._build_rendered_chunks()

    def _shift_rendered_chunks(self, originDelta: glm.vec2):
        isShiftX = originDelta[0]
        isShiftWest = isShiftX == -1

        isShiftZ = originDelta[1]
        isShiftSouth = isShiftZ == -1
        if isShiftX:
            if isShiftWest:
                # Origin shifted West (lose East most chunks)
                self.renderedChunks.pop()
            else:
                # Origin shifted East (lose West most chunks)
                self.renderedChunks.pop(0)
        if isShiftZ:
            if isShiftSouth:
                # Origin shifted south (lose north most chunks)
                [zAxis.pop() for zAxis in self.renderedChunks]
            else:
                # Origin shifted north (lose south most chunks)
                [zAxis.pop(0) for zAxis in self.renderedChunks]
        if isShiftX:
            if isShiftWest:
                # Insert array of chunks to the West of those remaining in the array to the major array
                self.renderedChunks.insert(
                    0,
                    [chunk.neighbourChunks[Chunk.Cardinal.West] if chunk.is_chunk else CHUNK_NULL for chunk in
                     self.renderedChunks[0]]
                )
            else:
                # Append array of chunks to the East of those remaining in the array to the major array
                self.renderedChunks.append(
                    [chunk.neighbourChunks[Chunk.Cardinal.East] if chunk.is_chunk else CHUNK_NULL for chunk in
                     self.renderedChunks[-1]]
                )
        if isShiftZ:
            if isShiftSouth:
                # Insert chunk into zAxis of chunks to the south of each chunk in each zAxis from the major array
                [
                    zAxis.insert(
                        0,
                        zAxis[0].neighbourChunks[Chunk.Cardinal.South]
                    ) if zAxis[0].is_chunk else zAxis.insert(0, CHUNK_NULL) for zAxis in self.renderedChunks
                ]
            else:
                # Append chunk to zAxis of chunks to the North of each chunk in each zAxis from the major array
                [
                    zAxis.append(
                        zAxis[-1].neighbourChunks[Chunk.Cardinal.North]
                    ) if zAxis[-1].is_chunk else zAxis.append(CHUNK_NULL) for zAxis in self.renderedChunks
                ]
        self.set_instance_data()

    def _set_render_distance(self, renderDistance: int):
        self.renderDistance = renderDistance
        self._chunksInRow = 2 * renderDistance + 1
        self._build_rendered_chunks()

    def _build_rendered_chunks(self):
        self.renderedChunks = [[CHUNK_NULL for i in range(self._chunksInRow)] for j in range(self._chunksInRow)]
        self.originChunk.get_chunks_to_render(self.renderedChunks, self._chunksInRow)
        self.set_instance_data()

    def set_instance_data(self):
        self.chunk_instances = []
        for chunkRow in self.renderedChunks:
            for chunk in chunkRow:
                self.chunk_instances.append(chunk.get_block_face_instance_data())
        self._build_mesh()

    def get_instance_data(self):
        chunk_instances = list(filter((None).__ne__, self.chunk_instances))
        if len(chunk_instances) == 0:
            return np.array([])
        return np.concatenate(chunk_instances, dtype=chunk_instances[0].dtype)

    def _build_mesh(self):
        self.worldMesh = Mesh(Block.vertices, Block.indices, Block._TextureAtlas,
                              Instances(self.get_instance_data(), Block.instanceLayout, True))

    def draw(self, projection: glm.mat4, view: glm.mat4):
        self.worldMesh.draw(self.worldBlockShader, projection, view)

    def __str__(self):
        print_str = ""
        for z in range(self._chunksInRow):
            for x in range(self._chunksInRow):
                if x > 0:
                    print_str += ",\t"
                chunk = self.renderedChunks[x][z]
                if chunk.is_chunk:
                    print_str += f"({chunk.worldChunkPos[0]},\t{chunk.worldChunkPos[1]})"
                else:
                    print_str += f"(NaN,\tNaN)"
                if self._chunksInRow - 1 == x:
                    print_str += "\n"
        return print_str
