from __future__ import annotations
from POGLE.Geometry.Texture import *
from POGLE.Core.Core import *

_faceTextureAtlas: Optional[UniformTextureAtlas] = None


def initFaceTextureAtlas():
    global _faceTextureAtlas
    if _faceTextureAtlas:
        return
    _faceTextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))


class Side(Enum):
    West = 0
    South = auto()
    East = auto()
    North = auto()
    Top = auto()
    Bottom = auto()


_oppositeSide: dict[Side, Side] = {
    Side.West: Side.East,
    Side.East: Side.West,
    Side.South: Side.North,
    Side.North: Side.South,
    Side.Top: Side.Bottom,
    Side.Bottom: Side.Top
}


class TexRef(Enum):
    Null = -1
    GrassTop = auto()
    Stone = auto()
    Dirt = auto()
    GrassSide = auto()


class Face:
    SERIALIZED_SIZE = struct.calcsize("hh")
    def __init__(self, side: Union[int, Side], tex_ref: Union[int, TexRef] = TexRef.Null):
        if isinstance(side, int):
            side = Side(side)
        self._side = side
        self._tex_ref: TexRef = None
        self._visible: bool = False
        if tex_ref:
            self.setTexID(tex_ref)

    def setTexID(self, tex_ref: Union[int, TexRef] = TexRef.Null):
        if isinstance(tex_ref, int):
            tex_ref = TexRef(tex_ref + 1)
        self._visible: bool = True if tex_ref.value else False
        self._tex_ref = tex_ref

    def _edit_visibility(self, visible) -> bool:
        if self._visible == visible:
            return False
        self._visible = visible
        return True

    def hide(self) -> bool:
        return self._edit_visibility(False)

    def reveal(self) -> bool:
        return self._edit_visibility(True)

    def get_instance(self) -> bytes:
        if self._visible and (self.tex_ref + 1):
            return self.serialize()
        else:
            return struct.pack("hH", -1, self.tex_ref)

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def side(self):
        return self._side.value

    @property
    def tex_ref(self):
        return self._tex_ref.value

    @property
    def opposite(self):
        return _oppositeSide[self._side]

    @property
    def texDims(self) -> Optional[TexDims]:
        if not _faceTextureAtlas:
            raise RuntimeError("Cannot get tex dims when _faceTextureAtlas is None")
        if self._tex_ref.value + 1:
            return _faceTextureAtlas.get_sub_texture(self._tex_ref.value)
        return None

    def serialize(self) -> bytes:
        # Ensure that side and tex_ref are within the range of signed shorts
        return struct.pack("hh", self.side, self.tex_ref)  # Pack side and tex_ref values as signed shorts

    @classmethod
    def deserialize(cls, packed_data: bytes) -> Face:
        # Unpack the side and tex_ref values from the binary data
        side_value, tex_ref_value = struct.unpack("hh", packed_data)

        # Convert the unpacked values back into their enum types
        return cls(
            side=Side(side_value),
            tex_ref=TexRef(tex_ref_value)
        )

    def __repr__(self):
        return f"Face({self._side}, {self._tex_ref}, visible={self.visible})"

# Example face serialization
face = Face(Side.West, TexRef.GrassSide)
serialized_face = face.serialize()

# Deserialization example
deserialized_face = Face.deserialize(serialized_face)
print("Pre-Serialized Face:", face)
print("Serialized Face:", serialized_face)
print("Deserialized Face:", deserialized_face)