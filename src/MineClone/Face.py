from __future__ import annotations

import copy

from POGLE.Geometry.Texture import *
from POGLE.Core.Core import *

_faceTextureAtlas: Optional[UniformTextureAtlas] = None


def initFaceTextureAtlas():
    global _faceTextureAtlas
    if _faceTextureAtlas:
        return
    _faceTextureAtlas = UniformTextureAtlas("terrain.png", glm.vec2(16, 16))


class Side(Enum):
    Null = -1
    West = auto()
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


class FaceID(Enum):
    Null = -1
    GrassTop = auto()
    Stone = auto()
    Dirt = auto()
    GrassSide = auto()


class Face:
    """
    Represents a face of a block with its properties and visibility status.
    """
    SERIALIZED_SIZE = struct.calcsize("hh")

    def __init__(self, face: Union[int, FaceID] = FaceID.Null, side: Union[int, Side] = Side.Null):
        self._face: FaceID = None
        self._visible: bool = False
        if face:
            self.set_face(face)
        self._side: Optional[Side] = None
        self.set_face_in_block(side)

    def set_face(self, face: Union[int, FaceID] = FaceID.Null):
        if isinstance(face, int):
            face = FaceID(face + 1)
        self.reveal() if face != FaceID.Null else self.hide()
        self._face = face

    def _edit_visibility(self, visible) -> bool:
        if self._visible == visible:
            return False
        self._visible = visible
        return True

    def hide(self) -> bool:
        return self._edit_visibility(False)

    def reveal(self) -> bool:
        return self._edit_visibility(True)

    def set_face_in_block(self, side: Union[int, Side] = Side.Null):
        if isinstance(side, int):
            side = Side(side)
        self._side = side

    def get_instance(self) -> bytes:
        if not self._side:
            raise RuntimeError("Cannot Get Face Instance If No Side is Set")
        if self.visible and self.face:
            return self.serialize()
        else:
            return struct.pack("hH", -1, self.face)

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def side(self):
        if not self._side:
            raise RuntimeError("Cannot Get Side Enum Value If No Side is Set")
        return self._side.value

    @property
    def face(self):
        return self._face.value

    @property
    def opposite(self):
        if not self._side:
            raise RuntimeError("Cannot Get Opposite Side If No Side is Set")
        return _oppositeSide[self._side]

    @property
    def texDims(self) -> Optional[TexDims]:
        if not _faceTextureAtlas:
            raise RuntimeError("Cannot get tex dims when _faceTextureAtlas is None")
        if self._face.value + 1:
            return _faceTextureAtlas.get_sub_texture(self._face.value)
        return None

    def serialize(self) -> bytes:
        # Ensure that side and face are within the range of signed shorts
        return struct.pack("hh", self.side, self.face)  # Pack side and face values as signed shorts

    @classmethod
    def deserialize(cls, packed_data: bytes) -> Face:
        # Unpack the side and face values from the binary data
        side_value, face_value = struct.unpack("hh", packed_data)

        # Convert the unpacked values back into their enum types
        return cls(
            side=Side(side_value),
            face=FaceID(face_value)
        )

    def __repr__(self):
        return f"Face({self._side}, {self._face}, visible={self.visible})"


_faces_dict: Dict[FaceID, bytes] = {
    FaceID.Null: Face().serialize(),
    FaceID.GrassTop: Face(FaceID.GrassTop).serialize(),
    FaceID.Stone: Face(FaceID.Stone).serialize(),
    FaceID.Dirt: Face(FaceID.Dirt).serialize(),
    FaceID.GrassSide: Face(FaceID.GrassSide).serialize()
}


def get_face(faceID: FaceID = FaceID.Null) -> Face:
    return Face.deserialize(_faces_dict[faceID])


def NULL_FACE() -> Face:
    return get_face()


if __name__ == "__main__":
    import time

    # Example face serialization
    face = Face(Side.West, FaceID.GrassSide)
    serialized_face = face.serialize()

    # Deserialization example
    deserialized_face = Face.deserialize(serialized_face)
    print("Pre-Serialized Face:", face)
    print("Deserialized Face:", deserialized_face)
