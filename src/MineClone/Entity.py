import nbtlib

from Block import *


@dataclass
class Entity(MCPhys, aabb=glm.vec3()):
    id: Optional[str] = None

    def __init_subclass__(cls, aabb: Union[glm.vec3, AABB], size: Optional[glm.vec3] = None):
        if isinstance(aabb, glm.vec3):
            if size is None:
                size = aabb
            aabb = AABB.from_pos_size(aabb, size)
        elif not isinstance(aabb, AABB):
            raise TypeError("Entity subclasses must receive AABB data in their definitions by giving an AABB or"
                            " position and size")
        cls.__aabb = aabb

    def __post_init__(self):
        super().__post_init__()
        del self.index

        self.velocity: glm.vec3 = glm.vec3()
        self.no_gravity: bool = False
        self.grounded: bool = False
        self.yaw: float = 0.0
        self.pitch: float = 0.0

    def to_nbt(self) -> nbtlib.Compound:
        return nbtlib.Compound({
            "id": nbtlib.String(self.id),
            "Motion": nbtlib.List[nbtlib.Double]([
                nbtlib.Double(self.velocity.x),
                nbtlib.Double(self.velocity.y),
                nbtlib.Double(self.velocity.z)
            ]),  # Velocity
            "NoGravity": nbtlib.Byte(self.no_gravity),  # True if affected by gravity
            "OnGround": nbtlib.Byte(self.grounded),  # True if entity is touching the ground
            "Pos": nbtlib.List[nbtlib.Double]([
                nbtlib.Double(self.pos.x),
                nbtlib.Double(self.pos.y),
                nbtlib.Double(self.pos.z)
            ]),  # Position
            "Rotation": nbtlib.List([
                # The entity's rotation around the Y axis (called yaw). Values vary from -180 (facing due north)
                # to -90 (facing due east) to 0 (facing due south) to +90 (facing due west)
                # to +180 (facing due north again)
                nbtlib.Float(self.yaw),
                # The entity's declination from the horizon (called pitch). Horizontal is 0.
                # Positive values look downward. Does not exceed positive or negative 90 degrees
                nbtlib.Float(self.pitch),
            ]),  # Two TAG_Floats representing rotation in degrees
            "UUID": nbtlib.IntArray([0, 0, 0, 0])
        })


@dataclass
class Mob(Entity, aabb=glm.vec3()):
    def __init_subclass__(cls, aabb: Union[glm.vec3, AABB], size: Optional[glm.vec3] = None):
        if isinstance(aabb, glm.vec3):
            aabb = AABB.from_pos_size(aabb, size)
        elif not isinstance(aabb, AABB):
            raise TypeError("Entity subclasses must receive AABB data in their definitions by giving an AABB or"
                            " position and size")
        cls.__aabb = aabb
    pass


@dataclass
class Player(Entity, aabb=glm.vec3()):
    def __init_subclass__(cls, aabb: Union[glm.vec3, AABB], size: Optional[glm.vec3] = None):
        if isinstance(aabb, glm.vec3):
            aabb = AABB.from_pos_size(aabb, size)
        elif not isinstance(aabb, AABB):
            raise TypeError("Entity subclasses must receive AABB data in their definitions by giving an AABB or"
                            " position and size")
        cls.__aabb = aabb

    def __post_init__(self):
        super().__post_init__()

        self.flying: bool = False
        self.fly_speed: float = 0.05
        self.insta_build: bool = False
        self.may_build: bool = True
        self.may_fly: bool = False
        self.walk_speed: float = 0.1

    def to_nbt(self):
        compound = super().to_nbt()
        del compound["id"]
        compound["abilities"] = nbtlib.Compound({
            "flying": nbtlib.Byte(self.flying),
            "flySpeed": nbtlib.Float(self.fly_speed),
            "instabuild": nbtlib.Byte(self.insta_build),
            "mayBuild": nbtlib.Byte(self.may_build),
            "mayfly": nbtlib.Byte(self.may_fly),
            "walk_speed": nbtlib.Float(self.walk_speed)
        })

        compound["DataVersion"] = nbtlib.Int(0)
        compound["Dimension"] = nbtlib.String("")  # ID of the dimension the player is in
        return compound


class TileEntity:
    pass
