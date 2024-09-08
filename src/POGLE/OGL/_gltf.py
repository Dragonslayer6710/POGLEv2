import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)  # Set your logging level

import unittest
import os

from typing import Optional, List, Dict, Any, Union, Literal
from dataclasses import dataclass, field
import glm  # Assuming glm is imported
import json

from OpenGL.GL import *

import glfw
import numpy as np


@dataclass
class GLTF:
    scene: Optional[int] = None

    @dataclass
    class Scene:
        name: Optional[str] = None
        nodes: List[int] = field(default_factory=list)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Scene':
            return cls(
                name=data.get("name"),
                nodes=data.get("nodes", [])
            )

        def to_dict(self) -> Dict[str, Any]:
            return {
                "name": self.name,
                "nodes": self.nodes
            }

    scenes: List[Scene] = field(default_factory=list)

    @dataclass
    class Camera:
        type: Literal['perspective', 'orthographic']
        data: Dict[str, float]

        def __post_init__(self):
            if self.type == 'perspective':
                required_keys = {'aspectRatio', 'yfov', 'zfar', 'znear'}
                if not required_keys.issubset(self.data.keys()):
                    raise ValueError(f"Perspective camera data must include keys: {required_keys}")
            elif self.type == 'orthographic':
                required_keys = {'xmag', 'ymag', 'zfar', 'znear'}
                if not required_keys.issubset(self.data.keys()):
                    raise ValueError(f"Orthographic camera data must include keys: {required_keys}")

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Camera':
            return cls(
                type=data["type"],
                data=data["data"]
            )

        def to_dict(self):
            return {
                "type": self.type,
                "data": self.data
            }
    cameras: List[Camera] = field(default_factory=list)

    @dataclass
    class Node:
        name: str
        matrix: Optional[glm.mat4] = None
        translation: Optional[glm.vec3] = field(default=None)
        rotation: Optional[glm.quat] = field(default=None)
        scale: Optional[glm.vec3] = field(default=None)
        children: List[int] = field(default_factory=list)
        mesh: Optional[int] = None
        skin: Optional[int] = None
        camera: Optional[int] = None

        def __post_init__(self):
            if self.matrix is None:
                if not (self.translation or self.rotation or self.scale):
                    raise ValueError("Either `matrix` or `translation`, `rotation`, and `scale` must be provided.")
            else:
                if (self.translation or self.rotation or self.scale):
                    raise ValueError("If `matrix` is provided, `translation`, `rotation`, and `scale` must be None.")

        @classmethod
        def from_dict(cls, data: Dict[str, Any], index: int = None) -> 'Node':
            return cls(
                name=data.get("name", f"Node_{index}"),
                matrix=glm.mat4(data["matrix"]) if data.get("matrix") else None,
                translation=glm.vec3(data["translation"]) if data.get("translation") else None,
                rotation=glm.quat(data["rotation"]) if data.get("rotation") else None,
                scale=glm.vec3(data["scale"]) if data.get("scale") else None,
                children=data.get("children", []),
                mesh=data.get("mesh"),
                skin=data.get("skin"),
                camera=data.get("camera")
            )

        def to_dict(self):
            return {
                "name": self.name,
                "matrix": self.matrix.to_list() if self.matrix else None,
                "translation": self.translation.to_list() if self.translation else None,
                "rotation": self.rotation.to_list() if self.rotation else None,
                "scale": self.scale.to_list() if self.scale else None,
                "children": self.children,
                "mesh": self.mesh,
                "skin": self.skin,
                "camera": self.camera
            }

    nodes: List[Node] = field(default_factory=list)

    @dataclass
    class Image:
        uri: Optional[str] = None
        buffer_view: Optional[int] = None
        mime_type: Optional[str] = None

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Image':
            return cls(
                uri=data.get("uri"),
                buffer_view=data.get("buffer_view"),
                mime_type=data.get("mime_type")
            )

    images: List[Image] = field(default_factory=list)

    @dataclass
    class Sampler:
        mag_filter: int
        min_filter: int
        wrap_s: int
        wrap_t: int

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Sampler':
            return cls(
                mag_filter=data.get("magFilter"),
                min_filter=data.get("minFilter"),
                wrap_s=data.get("wrapS"),
                wrap_t=data.get("wrapT")
            )

    samplers: List[Sampler] = field(default_factory=list)

    @dataclass
    class Texture:
        name: str
        source: int
        sampler: int

        @classmethod
        def from_dict(cls, data: Dict[str, Any], index: int = None) -> 'Texture':
            return cls(
                name=data.get("name", f"Texture_{index}"),
                source=data["source"],
                sampler=data["sampler"]
            )

    textures: List[Texture] = field(default_factory=list)

    @dataclass
    class Material:
        name: Optional[str] = None

        @dataclass
        class PBRMetallicRoughness:
            base_color_factor: Optional[List[float]] = field(default=None)
            base_color_texture: Optional[Dict[str, Union[int, Dict[str, int]]]] = None
            metallic_factor: Optional[float] = None
            roughness_factor: Optional[float] = None
            metallic_roughness_texture: Optional[Dict[str, Union[int, Dict[str, int]]]] = None

            def to_dict(self) -> Dict[str, Any]:
                result = {}
                if self.base_color_factor is not None:
                    result["baseColorFactor"] = self.base_color_factor
                if self.base_color_texture:
                    result["baseColorTexture"] = self.base_color_texture
                if self.metallic_factor is not None:
                    result["metallicFactor"] = self.metallic_factor
                if self.roughness_factor is not None:
                    result["roughnessFactor"] = self.roughness_factor
                if self.metallic_roughness_texture:
                    result["metallicRoughnessTexture"] = self.metallic_roughness_texture
                return result

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> 'PBRMetallicRoughness':
                return cls(
                    base_color_factor=data.get("baseColorFactor"),
                    base_color_texture=data.get("baseColorTexture"),
                    metallic_factor=data.get("metallicFactor"),
                    roughness_factor=data.get("roughnessFactor"),
                    metallic_roughness_texture=data.get("metallicRoughnessTexture")
                )

        pbr_metallic_roughness: Optional[PBRMetallicRoughness] = None
        normal_texture: Optional[Dict[str, Union[int, Dict[str, int]]]] = None
        occlusion_texture: Optional[Dict[str, Union[int, Dict[str, int]]]] = None
        emissive_texture: Optional[Dict[str, Union[int, Dict[str, int]]]] = None
        emissive_factor: Optional[List[float]] = field(default=None)
        alpha_mode: Optional[Literal['OPAQUE', 'BLEND', 'MASK']] = None
        alpha_cutoff: Optional[float] = None
        double_sided: Optional[bool] = None

        def to_dict(self) -> Dict[str, Any]:
            result = {}
            if self.name:
                result["name"] = self.name
            if self.pbr_metallic_roughness:
                result["pbrMetallicRoughness"] = self.pbr_metallic_roughness.to_dict()
            if self.normal_texture:
                result["normalTexture"] = self.normal_texture
            if self.occlusion_texture:
                result["occlusionTexture"] = self.occlusion_texture
            if self.emissive_texture:
                result["emissiveTexture"] = self.emissive_texture
            if self.emissive_factor is not None:
                result["emissiveFactor"] = self.emissive_factor
            if self.alpha_mode:
                result["alphaMode"] = self.alpha_mode
            if self.alpha_cutoff is not None:
                result["alphaCutoff"] = self.alpha_cutoff
            if self.double_sided is not None:
                result["doubleSided"] = self.double_sided
            return result

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Material':
            pbr_metallic_roughness = data.get("pbrMetallicRoughness")
            return cls(
                name=data.get("name"),
                pbr_metallic_roughness=GLTF.Material.PBRMetallicRoughness.from_dict(pbr_metallic_roughness) if pbr_metallic_roughness else None,
                normal_texture=data.get("normalTexture"),
                occlusion_texture=data.get("occlusionTexture"),
                emissive_texture=data.get("emissiveTexture"),
                emissive_factor=data.get("emissiveFactor"),
                alpha_mode=data.get("alphaMode"),
                alpha_cutoff=data.get("alphaCutoff"),
                double_sided=data.get("doubleSided")
            )

    materials: List[Material] = field(default_factory=list)

    @dataclass
    class Mesh:
        @dataclass
        class Primitive:
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

            attributes: Attributes
            mode: Optional[int] = None
            indices: Optional[int] = None
            material: Optional[int] = None

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> 'Primitive':
                attributes = data["attributes"]
                return cls(
                    attributes=GLTF.Mesh.Primitive.Attributes.from_dict(attributes),
                    mode=data.get("mode"),
                    indices=data.get("indices"),
                    material=data.get("material")
                )

        primitives: List[Primitive] = field(default_factory=list)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Mesh':
            return cls(
                primitives=[GLTF.Mesh.Primitive.from_dict(prim) for prim in data.get("primitives", [])]
            )

    meshes: List[Mesh] = field(default_factory=list)

    @dataclass
    class Accessor:
        @dataclass
        class Sparse:
            count: int
            indices: Dict[str, Union[int, List[int]]]
            values: Dict[str, Union[List[float], List[int]]]

            def to_dict(self) -> Dict[str, Any]:
                return {"count": self.count, "indices": self.indices, "values": self.values}

        buffer_view: int
        component_type: int
        count: int
        type: Literal['SCALAR', 'VEC2', 'VEC3', 'VEC4', 'MAT2', 'MAT3', 'MAT4']
        byte_offset: Optional[int] = 0
        normalized: Optional[bool] = None
        max: Optional[List[float]] = None
        min: Optional[List[float]] = None
        sparse: Optional[Sparse] = None

        def __post_init__(self):
            if self.sparse:
                if not isinstance(self.sparse.indices, (list, dict)):
                    raise ValueError("Sparse indices must be a list or dictionary.")
                if not isinstance(self.sparse.values, (list, dict)):
                    raise ValueError("Sparse values must be a list or dictionary.")

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Accessor':
            sparse_data = data.get("sparse")
            sparse = None
            if sparse_data:
                sparse = cls.Sparse(
                    count=sparse_data["count"],
                    indices=sparse_data["indices"],
                    values=sparse_data["values"]
                )
            return cls(
                buffer_view=data["bufferView"],
                byte_offset=data.get("byteOffset"),
                component_type=data["componentType"],
                count=data["count"],
                type=data["type"],
                normalized=data.get("normalized"),
                max=data.get("max"),
                min=data.get("min"),
                sparse=sparse
            )

    accessors: List[Accessor] = field(default_factory=list)

    @dataclass
    class BufferView:
        buffer: int
        byte_length: int
        byte_offset: Optional[int] = 0
        target: Optional[int] = None

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'BufferView':
            return cls(
                buffer=data["buffer"],
                byte_offset=data.get("byteOffset"),
                byte_length=data["byteLength"],
                target=data.get("target")
            )

    buffer_views: List[BufferView] = field(default_factory=list)

    @dataclass
    class Buffer:
        uri: str
        byte_length: int

        def is_external(self) -> bool:
            return not self.uri.startswith('data:')

        def is_inline(self) -> bool:
            return self.uri.startswith('data:')

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Buffer':
            return cls(
                uri=data.get("uri"),
                byte_length=data["byteLength"]
            )

    buffers: List[Buffer] = field(default_factory=list)

    @dataclass
    class Skin:
        inverse_bind_matrices: int
        skeleton: Optional[int] = None
        joints: List[int] = field(default_factory=list)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Skin':
            return cls(
                inverse_bind_matrices=data["inverseBindMatrices"],
                skeleton=data.get("skeleton"),
                joints=data.get("joints", [])
            )

    skins: List[Skin] = field(default_factory=list)

    @dataclass
    class Animation:
        @dataclass
        class Channel:
            sampler: int
            target: Dict[str, Union[int, str]]

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> 'Channel':
                return cls(
                    sampler=data["sampler"],
                    target=data["target"]
                )

        @dataclass
        class Sampler:
            input: int
            output: int
            interpolation: Optional[str] = None

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> 'Sampler':
                return cls(
                    input=data["input"],
                    output=data["output"],
                    interpolation=data.get("interpolation")
                )

        channels: List[Channel] = field(default_factory=list)
        samplers: List[Sampler] = field(default_factory=list)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Animation':
            return cls(
                channels=[GLTF.Animation.Channel.from_dict(chan) for chan in data.get("channels", [])],
                samplers=[GLTF.Animation.Sampler.from_dict(samp) for samp in data.get("samplers", [])]
            )

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
        return cls(
            scene=data.get("scene"),
            scenes=[GLTF.Scene.from_dict(scene) for scene in data.get("scenes", [])],
            cameras=[GLTF.Camera.from_dict(camera) for camera in data.get("cameras", [])],
            nodes=[GLTF.Node.from_dict(node, i) for i, node in enumerate(data.get("nodes", []))],
            images=[GLTF.Image.from_dict(image) for image in data.get("images", [])],
            samplers=[GLTF.Sampler.from_dict(sampler) for sampler in data.get("samplers", [])],
            textures=[GLTF.Texture.from_dict(texture) for texture in data.get("textures", [])],
            materials=[GLTF.Material.from_dict(material) for material in data.get("materials", [])],
            meshes=[GLTF.Mesh.from_dict(mesh) for mesh in data.get("meshes", [])],
            accessors=[GLTF.Accessor.from_dict(accessor) for accessor in data.get("accessors", [])],
            buffer_views=[GLTF.BufferView.from_dict(buffer_view) for buffer_view in data.get("bufferViews", [])],
            buffers=[GLTF.Buffer.from_dict(buffer) for buffer in data.get("buffers", [])],
            skins=[GLTF.Skin.from_dict(skin) for skin in data.get("skins", [])],
            animations=[GLTF.Animation.from_dict(animation) for animation in data.get("animations", [])]
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
        self.assertAlmostEqual(camera.data['aspectRatio'], 1.0, places=5)

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
        buffer_view = GLTF.BufferView.from_dict(data)
        self.assertEqual(buffer_view.buffer, 0)
        self.assertEqual(buffer_view.byte_length, 1024)
        self.assertEqual(buffer_view.byte_offset, 0)
        self.assertEqual(buffer_view.target, 34962)

    def test_buffer_from_dict(self):
        data = {"uri": "buffer.bin", "byteLength": 2048}
        buffer = GLTF.Buffer.from_dict(data)
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
        gltf.cameras.append(GLTF.Camera(type="perspective", data={"aspectRatio": 1.0, "yfov": 0.8, "zfar": 1000.0, "znear": 0.1}))
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
