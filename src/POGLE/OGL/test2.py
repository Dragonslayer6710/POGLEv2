from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class GLTFExtension:
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "properties": self.properties}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GLTFExtension":
        name = data.get("name", "")
        properties = data.get("properties", {})
        return cls(name=name, properties=properties)


@dataclass
class GLTFExtra:
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"extra": self.data}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'GLTFExtra':
        extra_data = data.get('extra', {})
        return GLTFExtra(data=extra_data)


@dataclass
class _GLTFBase:
    def __init_subclass__(cls, **kwargs):
        raise TypeError(f"Subclassing is not allowed for {cls.__name__}")

    _extensions: Optional[List[GLTFExtension]] = field(default=None)
    _extras: Optional[List[GLTFExtra]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        ext_dict = {}
        if self._extensions:
            ext_dict["extensions"] = [extension.to_dict() for extension in self._extensions]
        if self._extras:
            ext_dict["extras"] = [extra.to_dict() for extra in self._extras]
        return ext_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> '_GLTFBase':
        exts = [GLTFExtension.from_dict(ext) for ext in data.get("extensions", [])]
        extras = [GLTFExtra.from_dict(extra) for extra in data.get("extras", [])]
        if exts or extras:
            return cls(
                _extensions=exts,
                _extras=extras
            )
        return None


@dataclass
class GLTFAsset:
    version: str
    copyright: Optional[str] = None
    generator: Optional[str] = None
    minVersion: Optional[str] = None
    base: Optional[_GLTFBase] = None

    def to_dict(self) -> Dict[str, Any]:
        asset_dict = {"version": self.version}
        if self.base:
            asset_dict.update(self.base.to_dict())
        if self.copyright:
            asset_dict["copyright"] = self.copyright
        if self.generator:
            asset_dict["generator"] = self.generator
        if self.minVersion:
            asset_dict["minVersion"] = self.minVersion
        return asset_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GLTFAsset":
        base_data = _GLTFBase.from_dict(data)
        return cls(
            version=data["version"],
            copyright=data.get("copyright"),
            generator=data.get("generator"),
            minVersion=data.get("minVersion"),
            base=base_data  # Properly pass the _GLTFBase instance
        )

# Example usage
a = GLTFAsset.from_dict({"version": "", "extensions": [{"name": ""}]})
b = GLTFAsset.from_dict({"version": ""})
print(a)
print(b)
