from POGLE.Core.Core import *
from PIL import Image


class Texture:
    _TextureSlots: dict[GLuint, bool] = {}
    _FreedTextureSlots: list[GLuint] = []

    def __init__(self, textureFile: str):
        self.ID = glGenTextures(1)

        texImg = Image.open(f"{cwd}/../assets/textures/{textureFile}").transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        ).convert("RGBA")
        self._Size: glm.vec2 = glm.vec2(texImg.size)

        imgBytes = texImg.tobytes()
        texImg.close()

        self.bind()
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self._Size.x, self._Size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, imgBytes)
        glGenerateMipmap(GL_TEXTURE_2D)
        self.unbind()

    def __del__(self):
        if glIsTexture(self.ID):
            glDeleteTextures(1, self.ID)

    def bind(self):
        self._TextureSlot: GLuint = Texture._get_next_texture_slot()
        glActiveTexture(GL_TEXTURE0 + self._TextureSlot)
        glBindTexture(GL_TEXTURE_2D, self.ID)

    def unbind(self):
        self._clear_texture_slot()
        glBindTexture(GL_TEXTURE_2D, 0)

    def _clear_texture_slot(self):
        Texture._TextureSlots[self._TextureSlot] = False
        lastElem = len(Texture._TextureSlots) - 1

        for i in range(lastElem + 1):
            if i == lastElem:
                Texture._FreedTextureSlots.append(self._TextureSlot)
            elif Texture._FreedTextureSlots[i] > self._TextureSlot:
                Texture._FreedTextureSlots.insert(i + 1, self._TextureSlot)
        self._TextureSlot = None

    def get_texture_slot(self):
        return self._TextureSlot

    @staticmethod
    def _get_next_texture_slot() -> GLuint:
        if Texture._FreedTextureSlots == []:
            textureSlot = len(Texture._TextureSlots)
        else:
            textureSlot = Texture._FreedTextureSlots.pop()
        Texture._TextureSlots[textureSlot] = True
        return textureSlot


class TexDims:
    def __init__(self, position: Optional[glm.vec2] = None, size: Optional[glm.vec2] = None):
        self.pos: glm.vec2 = position
        self.size: glm.vec2 = size


class TextureAtlas(Texture):
    def __init__(self, textureFile: str):
        self._SubTextures: list[TexDims] = []
        super().__init__(textureFile)

    def get_sub_texture(self, id: GLuint) -> TexDims:
        return self._SubTextures[id]

    def add_sub_texture(self, position: glm.vec2, size: glm.vec2):
        self._SubTextures.append(TexDims(position, size))


class UniformTextureAtlas(TextureAtlas):
    def __init__(self, textureFile: str, subTexSize: glm.vec2):
        super().__init__(textureFile)
        relSize = subTexSize / self._Size
        atlasDims = self._Size / subTexSize
        floor = glm.floor(atlasDims)

        for y in range(int(atlasDims.y)):
            for x in range(int(atlasDims.x)):
                self.add_sub_texture(relSize * glm.vec2(x, atlasDims.y - 1 - y), relSize)
