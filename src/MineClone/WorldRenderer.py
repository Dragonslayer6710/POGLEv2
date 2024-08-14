from MineClone.World import *


class WorldRenderer:
    from MineClone.World import _CHUNKS_IN_ROW
    def __init__(self, world: World, originPos: glm.vec3, renderDistance: int = _CHUNKS_IN_ROW):
        self.world: World = world
        self.originPos: glm.vec3 = originPos
        self.originChunk: Chunk = None
        self.renderDistance: int = None
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
        isShiftEast = isShiftX == -1

        isShiftZ = originDelta[1]
        isShiftSouth = isShiftZ == -1
        if isShiftX:
            if isShiftEast:
                # Origin shifted east (lose west most chunks)
                self.renderedChunks.pop()
            else:
                # Origin shifted west (lose east most chunks)
                self.renderedChunks.pop(0)
        if isShiftZ:
            if isShiftSouth:
                # Origin shifted south (lose north most chunks)
                [zAxis.pop() for zAxis in self.renderedChunks]
            else:
                # Origin shifted north (lose south most chunks)
                [zAxis.pop(0) for zAxis in self.renderedChunks]
        if isShiftX:
            if isShiftEast:
                # Insert array of chunks to the east of those remaining in the array to the major array
                self.renderedChunks.insert(
                    0,
                    [chunk.adjChunks[Chunk.Side.East] for chunk in self.renderedChunks[0]]
                )
            else:
                # Append array of chunks to the west of those remaining in the array to the major array
                self.renderedChunks.append(
                    [chunk.adjChunks[Chunk.Side.West] for chunk in self.renderedChunks[-1]]
                )
        if isShiftZ:
            if isShiftSouth:
                # Insert chunk into zAxis of chunks to the south of each chunk in each zAxis from the major array
                [
                    zAxis.insert(
                        0,
                        zAxis[0].adjChunks[Chunk.Side.South]
                    ) if zAxis[0].is_chunk else zAxis.insert(0, CHUNK_NULL) for zAxis in self.renderedChunks
                ]
            else:
                # Append chunk to zAxis of chunks to the North of each chunk in each zAxis from the major array
                [
                    zAxis.append(
                        zAxis[-1].adjChunks[Chunk.Side.South]
                    ) if zAxis[-1].is_chunk else zAxis.append(CHUNK_NULL) for zAxis in self.renderedChunks
                ]
        self._set_instance_data()

    def _set_render_distance(self, renderDistance: int):
        self.renderDistance = renderDistance
        self._build_rendered_chunks()


    def _build_rendered_chunks(self):
        self.renderedChunks = [[None for i in range(self.renderDistance)] for i in range(self.renderDistance)]
        self.originChunk.get_chunks_to_render(self.renderedChunks, self.renderDistance - 1)
        self._set_instance_data()

    def _set_instance_data(self):
        self.chunk_instances = []
        for chunkRow in self.renderedChunks:
            for chunk in chunkRow:
                self.chunk_instances.append(chunk.get_instance_data())
        self._build_mesh()

    def get_instance_data(self):
        chunk_instances = list(filter((None).__ne__, self.chunk_instances))
        if len(chunk_instances) == 0:
            return None
        return np.concatenate(chunk_instances, dtype=chunk_instances[0].dtype)

    def _build_mesh(self):
        self.worldMesh = Mesh(Block.vertices, Block.indices, Block._TextureAtlas,
                              Instances(self.get_instance_data(), Block.instanceLayout, True))


    def draw(self):
        self.worldMesh.draw(self.worldBlockShader)