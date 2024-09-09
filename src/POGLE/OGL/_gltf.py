import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)  # Set your logging level

import unittest
import os

from typing import Optional, List, Dict, Any, Union, Literal
from numbers import Number

from dataclasses import dataclass, field
import glm  # Assuming glm is imported
import json

from OpenGL.GL import *

import glfw
import numpy as np


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
    copyright: Optional[str]
    generator: Optional[str]
    minVersion: Optional[str]
    _base: Optional[_GLTFBase]

    def to_dict(self) -> Dict[str, Any]:
        asset_dict = {"version": self.version}
        if self.copyright:
            asset_dict["copyright"] = self.copyright
        if self.generator:
            asset_dict["generator"] = self.generator
        if self.minVersion:
            asset_dict["minVersion"] = self.minVersion
        if self._base:
            asset_dict.update(self._base.to_dict())
        return asset_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GLTFAsset":
        return cls(
            version=data["version"],
            copyright=data.get("copyright"),
            generator=data.get("generator"),
            minVersion=data.get("minVersion"),
            _base=_GLTFBase.from_dict(data)  # Properly pass the _GLTFBase instance
        )


@dataclass
class Camera:
    index: int
    type: Literal['perspective', 'orthographic']
    name: Optional[str]
    params: Dict[str, float]
    _base: Optional[_GLTFBase]

    def __post_init__(self):
        if self.type == 'perspective':
            required_keys = {'aspectRatio', 'yfov', 'zfar', 'znear'}
            if not required_keys.issubset(self.params.keys()):
                raise ValueError(f"Perspective camera data must include keys: {required_keys}")
        elif self.type == 'orthographic':
            required_keys = {'xmag', 'ymag', 'zfar', 'znear'}
            if not required_keys.issubset(self.params.keys()):
                raise ValueError(f"Orthographic camera data must include keys: {required_keys}")

    def to_dict(self):
        cam_dict = {
            "type": self.type,
            self.type: {
                {k: v for k, v in self.params.items()}
            },

        }
        if self.name:
            cam_dict["name"] = self.name
        if self._base:
            cam_dict.update(self._base.to_dict())
        return cam_dict

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any]) -> 'Camera':
        cam_type = data["type"]
        params = data.get(cam_type)
        if not params:
            raise Exception(
                f"Camera {index} is of type {cam_type} but {cam_type} is not an attribute of camera {index}")
        return cls(
            index=index,
            type=cam_type,
            name=data.get("name"),
            params=params,
            _base=_GLTFBase.from_dict(data)  # Properly pass the _GLTFBase instance
        )


@dataclass
class GLTFBuffer:
    index: int
    uri: str
    byte_length: int
    name: Optional[str]
    _base: Optional[_GLTFBase]

    def is_external(self) -> bool:
        return not self.uri.startswith('data:')

    def is_inline(self) -> bool:
        return self.uri.startswith('data:')

    def to_dict(self) -> Dict[str, Any]:
        buffer_dict = {
            "uri": self.uri,
            "byteLength": self.byte_length,
        }
        if self.name:
            buffer_dict["name"] = self.name

        if self._base:
            buffer_dict.update(self._base.to_dict())
        return buffer_dict

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any]) -> 'GLTFBuffer':
        return cls(
            index=index,
            uri=data.get("uri"),
            byte_length=data["byteLength"],
            name=data.get("name"),
            _base=_GLTFBase.from_dict(data)  # Properly pass the _GLTFBase instance
        )


@dataclass
class GLTFBufferView:
    index: int
    buffer: GLTFBuffer
    byte_offset: GLuint
    byte_length: int
    byte_stride: int
    target: Optional[int]
    name: Optional[str]

    _base: Optional[_GLTFBase]

    def to_dict(self):
        bv_dict = {
            "buffer": self.buffer.index,
            "bufferOffset": self.byte_offset,
            "byteLength": self.byte_length
        }
        if self.byte_stride:
            bv_dict["byteStride"] = self.byte_stride
        if self.target:
            bv_dict["target"] = self.target
        if self.name:
            bv_dict["name"] = self.name
        if self._base:
            bv_dict.update(self._base.to_dict())

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any], buffers: List[GLTFBuffer]) -> 'GLTFBufferView':
        return cls(
            index=index,
            buffer=buffers[data["buffer"]],
            byte_offset=data.get("byteOffset", 0),
            byte_length=data["byteLength"],
            byte_stride=data.get("byteStride"),
            target=data.get("target"),
            name=data.get("name"),
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class Accessor:
    @dataclass
    class Sparse:
        @dataclass
        class Indices:
            buffer_view: GLTFBufferView
            byte_offset: int
            component_type: int

            _base: Optional[_GLTFBase]

            def to_dict(self):
                ind_dict = {
                    "bufferView": self.buffer_view.index,
                    "byteOffset": self.byte_offset,
                    "componentType": self.component_type
                }
                if self._base:
                    ind_dict.update(self._base.to_dict())
                return ind_dict

            @classmethod
            def from_dict(cls, data: Dict[str, Any], buffer_views: List[GLTFBufferView]):
                cls(
                    buffer_view=buffer_views[data["bufferView"]],
                    byte_offset=data.get("byteOffset", 0),
                    component_type=data["componentType"],
                    _base=_GLTFBase.from_dict(data)
                )

        @dataclass
        class Values:
            buffer_view: GLTFBufferView
            byte_offset: int

            _base: Optional[_GLTFBase]

            def to_dict(self):
                val_dict = {
                    "bufferView": self.buffer_view.index,
                    "byteOffset": self.byte_offset,
                }
                if self._base:
                    val_dict.update(self._base.to_dict())
                return val_dict

            @classmethod
            def from_dict(cls, data: Dict[str, Any], buffer_views: List[GLTFBufferView]):
                cls(
                    buffer_view=buffer_views[data["bufferView"]],
                    byte_offset=data.get("byteOffset", 0),
                    _base=_GLTFBase.from_dict(data)
                )

        count: int
        indices: Indices
        values: Values

        _base: Optional[_GLTFBase]

        def to_dict(self) -> Dict[str, Any]:
            sparse_dict = {
                "count": self.count,
                "indices": self.indices.to_dict(),
                "values": self.values.to_dict()
            }
            if self._base:
                sparse_dict.update(self._base.to_dict())
            return sparse_dict

        @classmethod
        def from_dict(cls, data: Dict[str, Any], buffer_views: List[GLTFBufferView]):
            return cls(
                count=data["count"],
                indices=Accessor.Sparse.Indices.from_dict(data, buffer_views),
                values=Accessor.Sparse.Values.from_dict(data, buffer_views),
                _base=_GLTFBase.from_dict(data)
            )
    index: int
    buffer_view: GLTFBufferView
    byte_offset: int
    component_type: int
    normalized: bool
    count: int
    type: Literal['SCALAR', 'VEC2', 'VEC3', 'VEC4', 'MAT2', 'MAT3', 'MAT4']
    max: Optional[List[Number]]
    min: Optional[List[Number]]
    sparse: Optional[Sparse]
    name: Optional[str]

    _base: Optional[_GLTFBase]

    def __post_init__(self):
        if self.max is not None:
            if not 1 <= len(self.max) <= 16:
                raise TypeError("Accessor.max must be a list of numbers (size 1 to 16)")

        if self.min is not None:
            if not 1 <= len(self.min) <= 16:
                raise TypeError("Accessor.min must be a list of numbers (size 1 to 16)")

        if self.sparse:
            if not isinstance(self.sparse.indices, (list, dict)):
                raise ValueError("Sparse indices must be a list or dictionary.")
            if not isinstance(self.sparse.values, (list, dict)):
                raise ValueError("Sparse values must be a list or dictionary.")

    def to_dict(self):
        acc_dict = {
            "byteOffset": self.byte_offset,
            "componentType": self.component_type,
            "normalized": self.normalized,
            "count": self.count,
            "type": self.type
        }
        if self.buffer_view:
            acc_dict["bufferView"] = self.buffer_view.index
        if self.max:
            acc_dict["max"] = self.max
        if self.min:
            acc_dict["min"] = self.min
        if self.sparse:
            acc_dict["sparse"] = self.sparse.to_dict()
        if self.name:
            acc_dict["name"] = self.name
        if self._base:
            acc_dict.update(self._base.to_dict())
        return acc_dict

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any], buffer_views: List[GLTFBufferView]) -> 'Accessor':
        return cls(
            index=index,
            buffer_view=buffer_views[data["bufferView"]],
            byte_offset=data.get("byteOffset", 0),
            component_type=data["componentType"],
            count=data["count"],
            type=data["type"],
            normalized=data.get("normalized", False),
            max=data.get("max"),
            min=data.get("min"),
            sparse=Accessor.Sparse.from_dict(data.get("sparse"), buffer_views),
            name=data.get("name"),
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class Image:
    index: int
    uri: Optional[str]
    mime_type: Optional[str]
    buffer_view: Optional[GLTFBufferView]
    name: Optional[str]

    _base: Optional[_GLTFBase]

    def to_dict(self):
        img_dict = {}
        if self.uri:
            img_dict["uri"] = self.uri
        if self.mime_type:
            img_dict["mimeType"] = self.mime_type
        if self.buffer_view:
            img_dict["bufferView"] = self.buffer_view.index
        if self.name:
            img_dict["name"] = self.name
        if self._base:
            img_dict.update(self._base.to_dict())
        return img_dict

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any], buffer_views: List[GLTFBufferView]) -> 'Image':
        buffer_view = data.get("buffer_view")
        return cls(
            index=index,
            uri=data.get("uri"),
            mime_type=data.get("mime_type"),
            buffer_view=buffer_views[buffer_view] if buffer_view else None,
            name=data.get("name"),
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class Sampler:
    index: int
    mag_filter: Optional[int]
    min_filter: Optional[int]
    wrap_s: Optional[int]
    wrap_t: Optional[int]
    name: Optional[str]

    _base: Optional[_GLTFBase]

    def to_dict(self):
        sampler_dict = {}
        if self.mag_filter:
            sampler_dict["mag_filter"] = self.mag_filter
        if self.min_filter:
            sampler_dict["min_filter"] = self.min_filter
        if self.wrap_s:
            sampler_dict["wrap_s"] = self.wrap_s
        if self.wrap_t:
            sampler_dict["wrap_t"] = self.wrap_t
        if self.name:
            sampler_dict["name"] = self.name
        if self._base:
            sampler_dict.update(self._base.to_dict())
        return sampler_dict

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any]) -> 'Sampler':
        return cls(
            index=index,
            mag_filter=data.get("magFilter"),
            min_filter=data.get("minFilter"),
            wrap_s=data.get("wrapS"),
            wrap_t=data.get("wrapT"),
            name=data.get("name"),
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class GLTFTexture:
    index: int
    sampler: Optional[Sampler]
    source: Optional[int]
    name: Optional[str]

    _base: Optional[_GLTFBase]

    def to_dict(self):
        tex_dict = {}
        if self.sampler:
            tex_dict["sampler"] = self.sampler.index
        if self.source:
            tex_dict["source"] = self.source
        if self.name:
            tex_dict["name"] = self.name
        if self._base:
            tex_dict.update(self._base.to_dict())
        return tex_dict

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any], samplers: List[Sampler]) -> 'GLTFTexture':
        return cls(
            index=index,
            sampler=samplers[data["sampler"]],
            source=data["source"],
            name=data.get("name", f"Texture_{index}"),
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class TextureInfo:
    texture: GLTFTexture
    tex_coord: GLuint

    _base: Optional[_GLTFBase]

    def to_dict(self):
        tex_info_dict = {
            "index": self.texture.index,
            "texCoord": self.tex_coord
        }
        if self._base:
            tex_info_dict.update(self._base.to_dict())
        return tex_info_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], textures: List[GLTFTexture]):
        return cls(
            texture=textures[data["index"]],
            tex_coord=data.get("texCoord", 0),
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class NormalTextureInfo(TextureInfo):
    scale: Number

    def to_dict(self):
        nti_dict = super().to_dict()
        nti_dict["scale"] = self.scale
        return nti_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], textures: List[GLTFTexture]):
        return cls(
            texture=textures[data["index"]],
            tex_coord=data.get("texCoord", 0),
            scale=data.get("scale", 1),
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class OcclusionTextureInfo(TextureInfo):
    strength: Number

    def to_dict(self):
        oti_dict = super().to_dict()
        oti_dict["strength"] = self.strength
        return oti_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], textures: List[GLTFTexture]):
        return cls(
            texture=textures[data["index"]],
            tex_coord=data.get("texCoord", 0),
            strength=data.get("scale", 1),
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class PBRMetallicRoughness:
    # TODO: texture info referencing
    base_color_factor: List[Number]
    base_color_texture: Optional[TextureInfo]
    metallic_factor: Number
    roughness_factor: Number
    metallic_roughness_texture: Optional[TextureInfo]

    _base: Optional[_GLTFBase]

    def to_dict(self):
        pbr_mr_dict = {
            "baseColorFactor": self.base_color_factor,
            "metallicFactor": self.metallic_factor,
            "roughnessFactor": self.roughness_factor
        }
        if self.base_color_texture:
            pbr_mr_dict["baseColorTexture"] = self.base_color_texture.to_dict()
        if self.metallic_roughness_texture:
            pbr_mr_dict["metallicRoughnessTexture"] = self.metallic_roughness_texture.to_dict()
        if self._base:
            pbr_mr_dict.update(self._base.to_dict())
        return pbr_mr_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], textures: List[GLTFTexture]):
        base_color_texture = data.get("baseColorTexture")
        metallic_roughness_texture = data.get("metallicRoughnessTexture")
        return cls(
            base_color_factor=data.get("baseColorFactor", [1, 1, 1, 1]),
            base_color_texture=TextureInfo.from_dict(base_color_texture, textures) if base_color_texture else None,
            metallic_factor=data.get("metallicFactor", 1),
            roughness_factor=data.get("roughnessFactor", 1),
            metallic_roughness_texture=TextureInfo.from_dict(metallic_roughness_texture,
                                                             textures) if metallic_roughness_texture else None,
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class Material:
    index: int
    name: Optional[str]
    _base: Optional[_GLTFBase]
    pbr_metallic_roughness: Optional[PBRMetallicRoughness]
    normal_texture: Optional[NormalTextureInfo]
    occlusion_texture: Optional[OcclusionTextureInfo]
    emissive_texture: Optional[TextureInfo]
    emissive_factor: List[Number]
    alpha_mode: Literal['OPAQUE', 'BLEND', 'MASK']
    alpha_cutoff: Number
    double_sided: bool

    def to_dict(self):
        mat_dict = {
            "emissiveFactor": self.emissive_factor,
            "alphaMode": self.alpha_mode,
            "alphaCutoff": self.alpha_cutoff,
            "doubleSided": self.double_sided
        }
        if self.name:
            mat_dict["name"] = self.name
        if self._base:
            mat_dict.update(self._base.to_dict())
        if self.pbr_metallic_roughness:
            mat_dict["pbrMetallicRoughness"] = self.pbr_metallic_roughness.to_dict()
        if self.normal_texture:
            mat_dict["normalTexture"] = self.normal_texture.to_dict()
        if self.occlusion_texture:
            mat_dict["occlusionTexture"] = self.occlusion_texture.to_dict()
        if self.emissive_texture:
            mat_dict["emissiveTexture"] = self.emissive_texture.to_dict()
        return mat_dict

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any], textures: List[GLTFTexture]) -> 'Material':
        pbr_metallic_roughness = data.get("pbrMetallicRoughness")
        normal_texture = data.get("normalTexture")
        occlusion_texture = data.get("occlusionTexture")
        emissive_texture = data.get("emissiveTexture")
        return cls(
            index=index,
            name=data.get("name"),
            pbr_metallic_roughness=
            PBRMetallicRoughness.from_dict(pbr_metallic_roughness, textures) if pbr_metallic_roughness else None,
            normal_texture=textures[normal_texture] if normal_texture else None,
            occlusion_texture=textures[occlusion_texture] if occlusion_texture else None,
            emissive_texture=textures[emissive_texture] if emissive_texture else None,
            emissive_factor=data.get("emissiveFactor", [0,0,0]),
            alpha_mode=data.get("alphaMode", "OPAQUE"),
            alpha_cutoff=data.get("alphaCutoff", 0.5),
            double_sided=data.get("doubleSided", False)
        )


@dataclass
class Attributes:
    attributes: Dict[str, int] = field(default_factory=dict)

    def get(self, key: str) -> Optional[int]:
        return self.attributes.get(key)

    def set(self, key: str, value: int) -> None:
        self.attributes[key] = value

    def to_dict(self) -> Dict[str, int]:
        return self.attributes

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'Attributes':
        return cls(attributes=data)


@dataclass
class MeshPrimitive:
    attributes: Attributes
    indices: Optional[Accessor]
    material: Optional[Material]
    mode: int
    targets = None #TODO: implement this, morph targets

    _base: Optional[_GLTFBase]

    def to_dict(self):
        mp_dict = {
            "attributes": self.attributes.to_dict(),
            "mode": self.mode
        }
        if self.indices:
            mp_dict["indices"] = self.indices.index
        if self.material:
            mp_dict["material"] = self.material.index
        if self.mode:
            mp_dict["mode"] = self.mode
        if self.targets:
            pass #TODO: morph targets
        if self._base:
            mp_dict.update(self._base.to_dict())
        return mp_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], accessors: List[Accessor], materials: List[Material]) -> 'MeshPrimitive':
        indices = data.get("indices") # index of indices accessor
        material = data.get("material")
        return cls(
            attributes=Attributes.from_dict(data["attributes"]),
            indices=accessors[indices] if indices else None,
            material=materials[material] if material else None,
            mode=data.get("mode", 4),
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class GLTFMesh:
    index: int
    primitives: List[MeshPrimitive]
    weights: List[Number]
    name: Optional[str]

    _base: Optional[_GLTFBase]

    def to_dict(self):
        mesh_dict = {
            "primitives": [prim.to_dict() for prim in self.primitives]
        }
        if self.weights:
            mesh_dict["weights"] = self.weights
        if self.name:
            mesh_dict["name"] = self.name
        if self._base:
            mesh_dict.update(self._base.to_dict())
        return mesh_dict

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any], materials: List[Material]) -> 'GLTFMesh':
        return cls(
            index=index,
            primitives=[MeshPrimitive.from_dict(prim, materials) for prim in data.get("primitives", [])],
            weights=data.get("weights"),
            name=data.get("name"),
            _base=_GLTFBase(data)
        )


@dataclass
class Node:
    index: int
    camera: Optional[Camera]
    children: List['Node']
    skin: Optional['Skin']
    matrix: glm.mat4
    mesh: Optional[GLTFMesh]
    rotation: glm.quat
    scale: glm.vec3
    translation: glm.vec3
    weights: List[Number] # TODO: MORPH TARGET
    name: Optional[str]

    _base: Optional[_GLTFBase]

    def to_dict(self):
        node_dict = {
            "matrix": self.matrix.to_list(),
            "rotation": self.rotation.to_list(),
            "scale": self.scale.to_list(),
            "translation": self.translation.to_list()
        }
        if self.camera:
            node_dict["camera"] = self.camera.index
        if self.children:
            node_dict["children"] = [node.index for node in self.children]
        if self.skin:
            node_dict["skin"] = self.skin.index
        if self.mesh:
            node_dict["mesh"] = self.mesh.index
        if self.weights:
            node_dict["weights"] = self.weights
        if self._base:
            node_dict.update(self._base.to_dict())
        return node_dict
    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any], nodes: Optional[List['Node']] = None,
                  meshes: Optional[List[GLTFMesh]] = None, skins: Optional[List['Skin']] = None,
                  cameras: Optional[List[Camera]] = None) -> 'Node':
        camera = data.get("camera") if cameras else None
        children = data.get("children") if nodes else None
        skin = data.get("skin") if skins else None
        matrix = data.get("matrix")
        mesh = data.get("mesh") if meshes else None
        rotation = data.get("rotation")
        scale = data.get("scale")
        translation = data.get("translation")

        return cls(
            index=index,
            camera=cameras[camera] if camera else None,
            children=nodes[children] if children else [],
            skin=skins[skin] if skin else None,
            matrix=glm.mat4(matrix) if matrix else None,
            mesh=meshes[mesh] if mesh else None,
            rotation=glm.quat(rotation) if rotation else None,
            scale=glm.vec3(scale) if scale else None,
            translation=glm.vec3(translation) if translation else None,
            weights=data.get("weights"),
            name=data.get("name"),
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class Scene:
    index: int
    nodes: Optional[List[Node]]
    name: Optional[str]

    _base: Optional[_GLTFBase]

    def to_dict(self) -> Dict[str, Any]:
        scene_dict = {}
        if self.nodes:
            scene_dict["nodes"] = [node.to_dict() for node in self.nodes]
        if self.name:
            scene_dict["name"] = self.name
        if self._base:
            scene_dict.update(self._base.to_dict())
        return scene_dict

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any], nodes: List[Node]) -> 'Scene':
        node_indices = data.get("nodes")
        return cls(
            index=index,
            name=data.get("name"),
            nodes=[nodes[node] for node in node_indices] if node_indices != [] else None,
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class Skin:
    index: int
    inverse_bind_matrices: Optional[Accessor]
    skeleton: Optional[Node]
    joints: List[Node]
    name: Optional[str]

    _base: Optional[_GLTFBase]

    def to_dict(self):
        skin_dict = {
            "joints": [joint.index for joint in self.joints]
        }
        if self.inverse_bind_matrices:
            skin_dict["inverseBindMatrices"] = self.inverse_bind_matrices.index
        if self.skeleton:
            skin_dict["skeleton"] = self.skeleton.index
        if self.name:
            skin_dict["name"] = self.name
        if self._base:
            skin_dict.update(self._base.to_dict())
        return skin_dict

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any], accessors: List[Accessor], nodes: List[Node]) -> 'Skin':
        inverse_bind_matrices = data.get("inverseBindMatrices")
        skeleton = data.get("skeleton")
        joints = data.get("joints", [])
        return cls(
            index=index,
            inverse_bind_matrices=accessors[inverse_bind_matrices] if inverse_bind_matrices else None,
            skeleton=nodes[skeleton] if skeleton else None,
            joints=[nodes[node] for node in joints] if joints != [] else [],
            name=data.get("name"),
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class AnimationSampler:
    index: int
    input: Accessor
    interpolation: Literal["LINEAR", "STEP", "CUBICSPLINE"]
    output: Accessor

    _base: Optional[_GLTFBase]

    def to_dict(self):
        as_dict = {
            "input": self.input.index,
            "interpolation": self.interpolation,
            "output": self.output.index
        }
        if self._base:
            as_dict.update(self._base.to_dict())
        return as_dict

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any], accessors: List[Accessor]) -> 'AnimationSampler':
        return cls(
            index=index,
            input=accessors[data["input"]],
            output=accessors[data["output"]],
            interpolation=data.get("interpolation", "LINEAR"),
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class AnimationChannelTarget:
    node: Optional[Node]
    path: str

    _base: Optional[_GLTFBase]

    def to_dict(self):
        pass https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.pdf

    @classmethod
    def from_dict(cls, data: Dict[str, Any], nodes: Optional[List[Node]]):
        pass


@dataclass
class AnimationChannel:

    sampler: AnimationSampler
    target: AnimationChannelTarget

    _base: Optional[_GLTFBase]

    def to_dict(self):
        ac_dict = {
            "sampler": self.sampler.index,
            "target": self.target.to_dict()
        }
        if self._base:
            ac_dict.update(self._base.to_dict())
        return self._base

    @classmethod
    def from_dict(cls, data: Dict[str, Any], animation_samplers: List[AnimationSampler]) -> 'AnimationChannel':
        sampler = data["sampler"]
        return cls(
            sampler=animation_samplers[sampler] if sampler else None,
            target=AnimationChannelTarget.from_dict(data["target"]),
            _base=_GLTFBase.from_dict(data)
        )


@dataclass
class Animation:
    index: int
    samplers: List[AnimationSampler] = field(default_factory=list)
    channels: List[AnimationChannel] = field(default_factory=list)

    @classmethod
    def from_dict(cls, index: int, data: Dict[str, Any]) -> 'AnimationChannel':
        samplers = [AnimationSampler.from_dict(sampler) for sampler in data.get("samplers", [])]
        channels = [AnimationChannel.from_dict(channel, samplers) for channel in data.get("channels", [])]

        return cls(
            index=index,
            samplers=samplers,
            channels=channels
        )


@dataclass
class GLTF:
    asset: GLTFAsset

    cameras: List[Camera] = field(default_factory=list)

    buffers: List[GLTFBuffer] = field(default_factory=list)

    buffer_views: List[GLTFBufferView] = field(default_factory=list)

    accessors: List[Accessor] = field(default_factory=list)

    images: List[Image] = field(default_factory=list)

    samplers: List[Sampler] = field(default_factory=list)

    textures: List[GLTFTexture] = field(default_factory=list)

    materials: List[Material] = field(default_factory=list)

    meshes: List[GLTFMesh] = field(default_factory=list)

    skins: List[Skin] = field(default_factory=list)

    nodes: List[Node] = field(default_factory=list)

    scenes: List[Scene] = field(default_factory=list)

    scene: Optional[Scene] = None

    animations: List[Animation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene": self.scene,
            "scenes": [scene.to_dict() for scene in self.scenes],
            "cameras": [camera.to_dict() for camera in self.cameras],
            "nodes": [node.to_dict() for node in self.nodes],
            "images": [image.to_dict() for image in self.images],
            "samplers": [sampler.to_dict() for sampler in self.samplers],
            "textures": [texture.to_dict() for texture in self.textures],
            "materials": [material.to_dict() for material in self.materials],
            "meshes": [mesh.to_dict() for mesh in self.meshes],
            "accessors": [accessor.to_dict() for accessor in self.accessors],
            "bufferViews": [buffer_view.to_dict() for buffer_view in self.buffer_views],
            "buffers": [buffer.to_dict() for buffer in self.buffers],
            "skins": [skin.to_dict() for skin in self.skins],
            "animations": [animation.to_dict() for animation in self.animations]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GLTF':
        asset = GLTFAsset.from_dict(data["asset"])

        cameras = [Camera.from_dict(i, camera) for i, camera in enumerate(data.get("cameras", []))]
        buffers = [GLTFBuffer.from_dict(i, buffer) for i, buffer in enumerate(data.get("buffers", []))]
        buffer_views = [GLTFBufferView.from_dict(i, buffer_view, buffers) for i, buffer_view in
                        enumerate(data.get("bufferViews", []))]
        accessors = [Accessor.from_dict(accessor, buffer_views) for accessor in data.get("accessors", [])]
        imgs = [Image.from_dict(i, image, buffer_views) for i, image in enumerate(data.get("images", []))]
        samplers = [Sampler.from_dict(i, sampler) for i, sampler in enumerate(data.get("samplers", []))]
        textures = [GLTFTexture.from_dict(i, texture, samplers) for i, texture in enumerate(data.get("textures", []))]
        materials = [Material.from_dict(i, material, textures) for i, material in enumerate(data.get("materials", []))]
        meshes = [GLTFMesh.from_dict(i, mesh, materials) for i, mesh in enumerate(data.get("meshes", []))]

        # Collating Skins
        skins_data = data.get("skins", [])

        # First Skins pass: Create Skin objects without resolving joints and skeleton
        skins = [
            Skin.from_dict(
                index=i,
                data=skin_data,
                nodes=None,  # Don't resolve joints and skeleton yet
            ) for i, skin_data in enumerate(skins_data)
        ]

        # Collating Nodes
        nodes_data = data.get("nodes", [])

        # First Nodes pass: Create Node objects without resolving children
        nodes = [
            Node.from_dict(
                index=i,
                data=node_data,
                nodes=None,  # Don't resolve children yet
                meshes=meshes,
                skins=None,  # Don't resolve skins yet
                cameras=cameras
            ) for i, node_data in enumerate(nodes_data)
        ]

        # Second Nodes pass: Resolve children
        for i, node in enumerate(nodes):
            node.children = [nodes[child_index] for child_index in nodes_data[i].get("children", [])]

        # Second Skins pass: Resolve children
        for i, skin in enumerate(skins):
            skeleton_index = skins_data[i].get("skeleton")
            skin.skeleton = nodes[skeleton_index] if skeleton_index else None
            skin.joints = [nodes[joint_index] for joint_index in skins_data[i].get("joints", [])]

        scenes = [Scene.from_dict(i, scene, nodes) for i, scene in enumerate(data.get("scenes", []))]
        scene_index = data.get("scene")
        scene = scenes[scene_index] if scene_index else None

        animations = [Animation.from_dict(i, animation) for i, animation in enumerate(data.get("animations", []))]
        return cls(
            cameras=cameras,
            buffers=buffers,
            buffer_views=buffer_views,
            accessors=accessors,
            samplers=samplers,
            textures=textures,
            materials=materials,
            meshes=meshes,
            skins=skins,
            nodes=nodes,
            scenes=scenes,
            scene=scene,
            animations=animations
        )

    def save_to_file(self, filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_from_file(cls, filename: str) -> 'GLTF':
        with open(filename, 'r') as f:
            data = json.load(f)
            return cls.from_dict(data)


class TestGLTF(unittest.TestCase):

    def test_scene_from_dict(self):
        data = {"name": "Scene 1", "nodes": [1, 2, 3]}
        scene = GLTF.Scene.from_dict(data)
        self.assertEqual(scene.name, "Scene 1")
        self.assertEqual(scene.nodes, [1, 2, 3])

    def test_camera_from_dict(self):
        data = {"type": "perspective", "data": {"aspectRatio": 1.0, "yfov": 0.8, "zfar": 1000.0, "znear": 0.1}}
        camera = GLTF.Camera.from_dict(data)
        self.assertAlmostEqual(camera.params['aspectRatio'], 1.0, places=5)

    def test_node_from_dict(self):
        data = {
            "name": "Node 1",
            "matrix": None,
            "translation": [1.0, 2.0, 3.0],
            "rotation": None,
            "scale": None,
            "children": [1, 2],
            "mesh": 1,
            "skin": None,
            "camera": None
        }
        node = GLTF.Node.from_dict(data)
        self.assertTrue(glm.all(glm.equal(node.translation, glm.vec3(1.0, 2.0, 3.0))))

    def test_image_from_dict(self):
        data = {"uri": "image.png", "buffer_view": 0, "mime_type": "image/png"}
        image = GLTF.Image.from_dict(data)
        self.assertEqual(image.uri, "image.png")
        self.assertEqual(image.buffer_view, 0)
        self.assertEqual(image.mime_type, "image/png")

    def test_sampler_from_dict(self):
        data = {"magFilter": 9729, "minFilter": 9729, "wrapS": 10497, "wrapT": 10497}
        sampler = GLTF.Sampler.from_dict(data)
        self.assertEqual(sampler.mag_filter, 9729)
        self.assertEqual(sampler.min_filter, 9729)
        self.assertEqual(sampler.wrap_s, 10497)
        self.assertEqual(sampler.wrap_t, 10497)

    def test_texture_from_dict(self):
        data = {"name": "Texture 1", "source": 0, "sampler": 1}
        texture = GLTF.Texture.from_dict(data)
        self.assertEqual(texture.name, "Texture 1")
        self.assertEqual(texture.source, 0)
        self.assertEqual(texture.sampler, 1)

    def test_material_from_dict(self):
        data = {
            "name": "Material 1",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 0.0, 0.0, 1.0],
                "baseColorTexture": {"index": 0},
                "metallicFactor": 1.0,
                "roughnessFactor": 1.0
            },
            "normalTexture": {"index": 1},
            "occlusionTexture": {"index": 2},
            "emissiveTexture": {"index": 3},
            "emissiveFactor": [1.0, 1.0, 1.0],
            "alphaMode": "BLEND",
            "alphaCutoff": 0.5,
            "doubleSided": True
        }
        material = GLTF.Material.from_dict(data)
        self.assertEqual(material.name, "Material 1")
        self.assertEqual(material.pbr_metallic_roughness.base_color_factor, [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(material.pbr_metallic_roughness.base_color_texture, {"index": 0})
        self.assertEqual(material.pbr_metallic_roughness.metallic_factor, 1.0)
        self.assertEqual(material.pbr_metallic_roughness.roughness_factor, 1.0)
        self.assertEqual(material.normal_texture, {"index": 1})
        self.assertEqual(material.occlusion_texture, {"index": 2})
        self.assertEqual(material.emissive_texture, {"index": 3})
        self.assertEqual(material.emissive_factor, [1.0, 1.0, 1.0])
        self.assertEqual(material.alpha_mode, "BLEND")
        self.assertEqual(material.alpha_cutoff, 0.5)
        self.assertEqual(material.double_sided, True)

    def test_mesh_from_dict(self):
        data = {
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "TEXCOORD_0": 2
                },
                "mode": 4,
                "indices": 3,
                "material": 0
            }]
        }
        mesh = GLTF.Mesh.from_dict(data)
        self.assertEqual(len(mesh.primitives), 1)
        prim = mesh.primitives[0]
        self.assertEqual(prim.attributes.position, 0)
        self.assertEqual(prim.attributes.normal, 1)
        self.assertEqual(prim.attributes.texcoord_0, 2)
        self.assertEqual(prim.mode, 4)
        self.assertEqual(prim.indices, 3)
        self.assertEqual(prim.material, 0)

    def test_accessor_from_dict(self):
        data = {
            "bufferView": 0,
            "componentType": 5126,
            "count": 100,
            "type": "VEC3",
            "byteOffset": 0,
            "normalized": False,
            "max": [1.0, 1.0, 1.0],
            "min": [0.0, 0.0, 0.0],
            "sparse": {
                "count": 10,
                "indices": {"bufferView": 1, "byteOffset": 0},
                "values": {"bufferView": 2, "byteOffset": 1}
            }
        }
        accessor = GLTF.Accessor.from_dict(data)
        self.assertEqual(accessor.buffer_view, 0)
        self.assertEqual(accessor.component_type, 5126)
        self.assertEqual(accessor.count, 100)
        self.assertEqual(accessor.type, "VEC3")
        self.assertEqual(accessor.byte_offset, 0)
        self.assertEqual(accessor.normalized, False)
        self.assertEqual(accessor.max, [1.0, 1.0, 1.0])
        self.assertEqual(accessor.min, [0.0, 0.0, 0.0])
        self.assertEqual(accessor.sparse.count, 10)

    def test_buffer_view_from_dict(self):
        data = {"buffer": 0, "byteLength": 1024, "byteOffset": 0, "target": 34962}
        buffer_view = GLTF.GLTFBufferView.from_dict(data)
        self.assertEqual(buffer_view.buffer, 0)
        self.assertEqual(buffer_view.byte_length, 1024)
        self.assertEqual(buffer_view.byte_offset, 0)
        self.assertEqual(buffer_view.target, 34962)

    def test_buffer_from_dict(self):
        data = {"uri": "buffer.bin", "byteLength": 2048}
        buffer = GLTF.GLTFBuffer.from_dict(data)
        self.assertEqual(buffer.uri, "buffer.bin")
        self.assertEqual(buffer.byte_length, 2048)

    def test_skin_from_dict(self):
        data = {"inverseBindMatrices": 0, "skeleton": 1, "joints": [2, 3, 4]}
        skin = GLTF.Skin.from_dict(data)
        self.assertEqual(skin.inverse_bind_matrices, 0)
        self.assertEqual(skin.skeleton, 1)
        self.assertEqual(skin.joints, [2, 3, 4])

    def test_animation_from_dict(self):
        data = {
            "channels": [{"sampler": 0, "target": {"node": 1, "path": "translation"}}],
            "samplers": [{"input": 0, "output": 1, "interpolation": "LINEAR"}]
        }
        animation = GLTF.Animation.from_dict(data)
        self.assertEqual(len(animation.channels), 1)
        self.assertEqual(animation.channels[0].sampler, 0)
        self.assertEqual(animation.channels[0].target, {"node": 1, "path": "translation"})
        self.assertEqual(len(animation.samplers), 1)
        self.assertEqual(animation.samplers[0].input, 0)
        self.assertEqual(animation.samplers[0].output, 1)
        self.assertEqual(animation.samplers[0].interpolation, "LINEAR")

    def test_save_load(self):
        gltf = GLTF()
        gltf.scenes.append(GLTF.Scene(name="Test Scene", nodes=[0, 1]))
        gltf.cameras.append(
            GLTF.Camera(type="perspective", data={"aspectRatio": 1.0, "yfov": 0.8, "zfar": 1000.0, "znear": 0.1}))
        gltf.nodes.append(GLTF.Node(name="Test Node", translation=glm.vec3(1.0, 2.0, 3.0), children=[0], mesh=0))

        # Save to file
        with open('test_save_load_gltf.json', 'w') as f:
            json.dump(gltf.to_dict(), f)

        # Load from file
        with open('test_save_load_gltf.json', 'r') as f:
            loaded_data = json.load(f)

        loaded_gltf = GLTF.from_dict(loaded_data)
        self.assertEqual(len(loaded_gltf.scenes), 1)
        self.assertEqual(loaded_gltf.scenes[0].name, "Test Scene")
        self.assertEqual(len(loaded_gltf.cameras), 1)
        self.assertEqual(loaded_gltf.cameras[0].type, "perspective")
        self.assertEqual(len(loaded_gltf.nodes), 1)
        self.assertEqual(loaded_gltf.nodes[0].name, "Test Node")

    def test_scene_from_dict_invalid(self):
        data = {
            # Example of an invalid node that lacks both matrix and translation/rotation/scale
            "nodes": [
                {
                    "name": "Invalid Node"
                    # Missing matrix and all transformation data
                }
            ]
        }

        with self.assertRaises(ValueError):
            GLTF.from_dict(data)

    def test_load_valid_model(self):
        # Load the GLTF file
        loaded_gltf = GLTF.load_from_file('AnimatedCube.gltf')

        # Check if scenes exist
        self.assertTrue(loaded_gltf.scenes)

        # Check if nodes exist
        self.assertTrue(loaded_gltf.nodes)

        # Check if cameras exist (optional)
        if loaded_gltf.cameras:
            self.assertTrue(loaded_gltf.cameras)

        # Check if materials exist
        self.assertTrue(loaded_gltf.materials)

        # Check if meshes exist
        self.assertTrue(loaded_gltf.meshes)

    def tearDown(self):
        if os.path.exists('test_save_load_gltf.json'):
            os.remove('test_save_load_gltf.json')
