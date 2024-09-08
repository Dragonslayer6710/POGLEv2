import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)  # Set your logging level

import unittest

from typing import Optional, List, Dict, Any, Union, Literal
from dataclasses import dataclass, field
import glm  # Assuming glm is imported
import json
from OpenGL.GL import *

@dataclass
class GLTF:
    """
    Represents a GLTF (GL Transmission Format) data structure.

    Attributes:
        scene: Optional[int] - Index of the scene to be used.
        scenes: List[Scene] - List of scenes in the GLTF file.
        cameras: List[Camera] - List of cameras in the GLTF file.
        nodes: List[Node] - List of nodes in the GLTF file.
        images: List[Image] - List of images in the GLTF file.
        samplers: List[Sampler] - List of samplers in the GLTF file.
        textures: List[Texture] - List of textures in the GLTF file.
        materials: List[Material] - List of materials in the GLTF file.
        meshes: List[Mesh] - List of meshes in the GLTF file.
        accessors: List[Accessor] - List of accessors in the GLTF file.
        buffer_views: List[BufferView] - List of buffer views in the GLTF file.
        buffers: List[Buffer] - List of buffers in the GLTF file.
        skins: List[Skin] - List of skins in the GLTF file.
        animations: List[Animation] - List of animations in the GLTF file.
    """
    scene: Optional[int] = None

    @dataclass
    class Scene:
        """
        Represents a scene in a GLTF file.

        Attributes:
            scene_name: Optional[str] - Name of the scene.
            nodes: List[int] - List of node indices in the scene.
        """
        scene_name: Optional[str] = None
        nodes: List[int] = field(default_factory=list)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Scene':
            """Creates a Scene instance from a dictionary."""
            return cls(
                scene_name=data.get("scene_name"),
                nodes=data.get("nodes", [])
            )

    scenes: List[Scene] = field(default_factory=list)

    @dataclass
    class Camera:
        """
        Represents a camera in a GLTF file.

        Attributes:
            type: Literal['perspective', 'orthographic'] - Type of the camera.
            data: Dict[str, float] - Camera data specific to the type.
        """
        type: Literal['perspective', 'orthographic']
        data: Dict[str, float]

        def __post_init__(self):
            """Validates camera data based on the type."""
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
            """Creates a Camera instance from a dictionary."""
            return cls(
                type=data["type"],
                data=data["data"]
            )

    cameras: List[Camera] = field(default_factory=list)

    @dataclass
    class Node:
        """
        Represents a node in a GLTF file.

        Attributes:
            node_name: str - Name of the node.
            matrix: Optional[glm.mat4] - Transformation matrix of the node.
            translation: Optional[glm.vec3] - Translation vector.
            rotation: Optional[glm.quat] - Rotation quaternion.
            scale: Optional[glm.vec3] - Scaling vector.
            children: List[int] - List of child node indices.
            mesh: Optional[int] - Index of the mesh associated with the node.
            skin: Optional[int] - Index of the skin associated with the node.
            camera: Optional[int] - Index of the camera associated with the node.
        """
        node_name: str
        matrix: Optional[glm.mat4] = None
        translation: Optional[glm.vec3] = field(default=None)
        rotation: Optional[glm.quat] = field(default=None)
        scale: Optional[glm.vec3] = field(default=None)
        children: List[int] = field(default_factory=list)
        mesh: Optional[int] = None
        skin: Optional[int] = None
        camera: Optional[int] = None

        def __post_init__(self):
            """Validates node transformations."""
            if self.matrix is None:
                if not (self.translation or self.rotation or self.scale):
                    raise ValueError("Either `matrix` or `translation`, `rotation`, and `scale` must be provided.")
            else:
                if (self.translation or self.rotation or self.scale):
                    raise ValueError("If `matrix` is provided, `translation`, `rotation`, and `scale` must be None.")

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Node':
            """Creates a Node instance from a dictionary."""
            return cls(
                node_name=data["node_name"],
                matrix=data.get("matrix"),
                translation=data.get("translation"),
                rotation=data.get("rotation"),
                scale=data.get("scale"),
                children=data.get("children", []),
                mesh=data.get("mesh"),
                skin=data.get("skin"),
                camera=data.get("camera")
            )

    nodes: List[Node] = field(default_factory=list)

    @dataclass
    class Image:
        """
        Represents an image in a GLTF file.

        Attributes:
            uri: Optional[str] - URI of the image.
            buffer_view: Optional[int] - Index of the buffer view.
            mime_type: Optional[str] - MIME type of the image.
        """
        uri: Optional[str] = None
        buffer_view: Optional[int] = None
        mime_type: Optional[str] = None

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Image':
            """Creates an Image instance from a dictionary."""
            return cls(
                uri=data.get("uri"),
                buffer_view=data.get("buffer_view"),
                mime_type=data.get("mime_type")
            )

    images: List[Image] = field(default_factory=list)

    @dataclass
    class Sampler:
        """
        Represents a sampler in a GLTF file.

        Attributes:
            mag_filter: int - Magnification filter.
            min_filter: int - Minification filter.
            wrap_s: int - Wrapping mode in the s direction.
            wrap_t: int - Wrapping mode in the t direction.
        """
        mag_filter: int
        min_filter: int
        wrap_s: int
        wrap_t: int

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Sampler':
            """Creates a Sampler instance from a dictionary."""
            return cls(
                mag_filter=data["mag_filter"],
                min_filter=data["min_filter"],
                wrap_s=data["wrap_s"],
                wrap_t=data["wrap_t"]
            )

    samplers: List[Sampler] = field(default_factory=list)

    @dataclass
    class Texture:
        """
        Represents a texture in a GLTF file.

        Attributes:
            tex_name: str - Name of the texture.
            source: int - Index of the image source.
            sampler: int - Index of the sampler.
        """
        tex_name: str
        source: int
        sampler: int

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Texture':
            """Creates a Texture instance from a dictionary."""
            return cls(
                tex_name=data["tex_name"],
                source=data["source"],
                sampler=data["sampler"]
            )

    textures: List[Texture] = field(default_factory=list)

    @dataclass
    class Material:
        """
        Represents a material in a GLTF file.

        Attributes:
            mat_name: Optional[str] - Name of the material.
            pbr_metallic_roughness: Optional[PBRMetallicRoughness] - PBR metallic-roughness properties.
            normal_texture: Optional[Dict[str, Union[int, Dict[str, int]]]] - Normal map texture.
            occlusion_texture: Optional[Dict[str, Union[int, Dict[str, int]]]] - Occlusion map texture.
            emissive_texture: Optional[Dict[str, Union[int, Dict[str, int]]]] - Emissive map texture.
            emissive_factor: Optional[List[float]] - Emissive factor.
            alpha_mode: Optional[Literal['OPAQUE', 'BLEND', 'MASK']] - Alpha mode.
            alpha_cutoff: Optional[float] - Alpha cutoff value.
            double_sided: Optional[bool] - Whether the material is double-sided.
        """
        mat_name: Optional[str] = None

        @dataclass
        class PBRMetallicRoughness:
            """
            Represents the PBR metallic-roughness properties of a material.

            Attributes:
                base_color_factor: Optional[List[float]] - Base color factor.
                base_color_texture: Optional[Dict[str, Union[int, Dict[str, int]]]] - Base color texture.
                metallic_factor: Optional[float] - Metallic factor.
                roughness_factor: Optional[float] - Roughness factor.
                metallic_roughness_texture: Optional[Dict[str, Union[int, Dict[str, int]]]] - Metallic-roughness texture.
            """
            base_color_factor: Optional[List[float]] = field(default=None)
            base_color_texture: Optional[Dict[str, Union[int, Dict[str, int]]]] = None
            metallic_factor: Optional[float] = None
            roughness_factor: Optional[float] = None
            metallic_roughness_texture: Optional[Dict[str, Union[int, Dict[str, int]]]] = None

            def to_dict(self) -> Dict[str, Any]:
                """Converts PBRMetallicRoughness to a dictionary."""
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
                """Creates a PBRMetallicRoughness instance from a dictionary."""
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
            """Converts Material to a dictionary."""
            result = {}
            if self.mat_name:
                result["name"] = self.mat_name
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
            """Creates a Material instance from a dictionary."""
            pbr_metallic_roughness = data.get("pbrMetallicRoughness")
            return cls(
                mat_name=data.get("name"),
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
        """
        Represents a mesh in a GLTF file.

        Attributes:
            primitives: List[Primitive] - List of primitives in the mesh.
        """
        @dataclass
        class Primitive:
            """
            Represents a primitive in a GLTF mesh.

            Attributes:
                attributes: Attributes - Attributes of the primitive.
                mode: Optional[int] - Draw mode (e.g., triangles, lines).
                indices: Optional[int] - Index buffer for the primitive.
                material: Optional[int] - Index of the material used.
            """
            @dataclass
            class Attributes:
                """
                Represents attributes for a primitive.

                Attributes:
                    position: int - Attribute index for position data.
                    normal: Optional[int] - Attribute index for normal data.
                    texcoord_0: Optional[int] - Attribute index for texcoord data.
                    color_0: Optional[int] - Attribute index for color data.
                    joints_0: Optional[int] - Attribute index for joint data.
                    weights_0: Optional[int] - Attribute index for weights data.
                    custom: Dict[str, int] - Custom attributes.
                """
                position: int
                normal: Optional[int] = None
                texcoord_0: Optional[int] = None
                color_0: Optional[int] = None
                joints_0: Optional[int] = None
                weights_0: Optional[int] = None
                custom: Dict[str, int] = field(default_factory=dict)

                def to_dict(self) -> Dict[str, int]:
                    """Converts Attributes to a dictionary."""
                    attributes = {
                        "POSITION": self.position,
                    }
                    if self.normal is not None:
                        attributes["NORMAL"] = self.normal
                    if self.texcoord_0 is not None:
                        attributes["TEXCOORD_0"] = self.texcoord_0
                    if self.color_0 is not None:
                        attributes["COLOR_0"] = self.color_0
                    if self.joints_0 is not None:
                        attributes["JOINTS_0"] = self.joints_0
                    if self.weights_0 is not None:
                        attributes["WEIGHTS_0"] = self.weights_0
                    attributes.update(self.custom)
                    return attributes

                @classmethod
                def from_dict(cls, data: Dict[str, int]) -> 'Attributes':
                    """Creates an Attributes instance from a dictionary."""
                    return cls(
                        position=data["POSITION"],
                        normal=data.get("NORMAL"),
                        texcoord_0=data.get("TEXCOORD_0"),
                        color_0=data.get("COLOR_0"),
                        joints_0=data.get("JOINTS_0"),
                        weights_0=data.get("WEIGHTS_0"),
                        custom={k: v for k, v in data.items() if k not in {"POSITION", "NORMAL", "TEXCOORD_0", "COLOR_0", "JOINTS_0", "WEIGHTS_0"}}
                    )

            attributes: Attributes
            mode: Optional[int] = None
            indices: Optional[int] = None
            material: Optional[int] = None

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> 'Primitive':
                """Creates a Primitive instance from a dictionary."""
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
            """Creates a Mesh instance from a dictionary."""
            return cls(
                primitives=[GLTF.Mesh.Primitive.from_dict(prim) for prim in data.get("primitives", [])]
            )

    meshes: List[Mesh] = field(default_factory=list)

    @dataclass
    class Accessor:
        """
        Represents an accessor in a GLTF file.

        Attributes:
            buffer_view: int - Index of the buffer view.
            component_type: int - Data type of the components.
            count: int - Number of elements in the accessor.
            type: Literal['SCALAR', 'VEC2', 'VEC3', 'VEC4', 'MAT2', 'MAT3', 'MAT4'] - Type of the accessor data.
            byte_offset: Optional[int] - Offset in bytes from the start of the buffer view.
            normalized: Optional[bool] - Whether the accessor data is normalized.
            max: Optional[List[float]] - Maximum values of the accessor data.
            min: Optional[List[float]] - Minimum values of the accessor data.
            sparse: Optional[Sparse] - Sparse data for the accessor.
        """
        @dataclass
        class Sparse:
            """
            Represents sparse data in an accessor.

            Attributes:
                count: int - Number of sparse values.
                indices: Dict[str, Union[int, List[int]]] - Sparse indices.
                values: Dict[str, Union[List[float], List[int]]] - Sparse values.
            """
            count: int
            indices: Dict[str, Union[int, List[int]]]
            values: Dict[str, Union[List[float], List[int]]]

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> 'Sparse':
                """Creates a Sparse instance from a dictionary."""
                return cls(
                    count=data["count"],
                    indices=data["indices"],
                    values=data["values"]
                )

        buffer_view: int
        component_type: int
        count: int
        type: Literal['SCALAR', 'VEC2', 'VEC3', 'VEC4', 'MAT2', 'MAT3', 'MAT4']
        byte_offset: Optional[int] = None
        normalized: Optional[bool] = None
        max: Optional[List[float]] = None
        min: Optional[List[float]] = None
        sparse: Optional[Sparse] = None

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Accessor':
            """Creates an Accessor instance from a dictionary."""
            sparse = data.get("sparse")
            return cls(
                buffer_view=data["bufferView"],
                component_type=data["componentType"],
                count=data["count"],
                type=data["type"],
                byte_offset=data.get("byteOffset"),
                normalized=data.get("normalized"),
                max=data.get("max"),
                min=data.get("min"),
                sparse=GLTF.Accessor.Sparse.from_dict(sparse) if sparse else None
            )

    accessors: List[Accessor] = field(default_factory=list)

    @dataclass
    class BufferView:
        """
        Represents a buffer view in a GLTF file.

        Attributes:
            buffer: int - Index of the buffer.
            byte_offset: int - Byte offset into the buffer.
            byte_length: int - Byte length of the view.
            target: Optional[int] - Target of the buffer view.
        """
        buffer: int
        byte_offset: int
        byte_length: int
        target: Optional[int] = None

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'BufferView':
            """Creates a BufferView instance from a dictionary."""
            return cls(
                buffer=data["buffer"],
                byte_offset=data["byteOffset"],
                byte_length=data["byteLength"],
                target=data.get("target")
            )

    buffer_views: List[BufferView] = field(default_factory=list)

    @dataclass
    class Buffer:
        """
        Represents a buffer in a GLTF file.

        Attributes:
            byte_length: int - Length of the buffer in bytes.
            uri: Optional[str] - URI of the buffer.
        """
        byte_length: int
        uri: Optional[str] = None

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Buffer':
            """Creates a Buffer instance from a dictionary."""
            return cls(
                byte_length=data["byteLength"],
                uri=data.get("uri")
            )

    buffers: List[Buffer] = field(default_factory=list)

    @dataclass
    class Skin:
        """
        Represents a skin in a GLTF file.

        Attributes:
            joints: List[int] - List of joint indices.
            skeleton: Optional[List[int]] - Index of the skeleton.
            inverse_bind_matrices: Optional[int] - Index of the inverse bind matrices buffer view.
        """
        joints: List[int]
        skeleton: Optional[List[int]] = None
        inverse_bind_matrices: Optional[int] = None

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Skin':
            """Creates a Skin instance from a dictionary."""
            return cls(
                joints=data["joints"],
                skeleton=data.get("skeleton"),
                inverse_bind_matrices=data.get("inverse_bind_matrices")
            )

    skins: List[Skin] = field(default_factory=list)

    @dataclass
    class Animation:
        """
        Represents an animation in a GLTF file.

        Attributes:
            name: Optional[str] - Name of the animation.
            channels: List[Channel] - List of channels in the animation.
            samplers: List[Sampler] - List of samplers in the animation.
        """
        @dataclass
        class Channel:
            """
            Represents a channel in an animation.

            Attributes:
                sampler: int - Index of the sampler.
                target: Dict[str, Union[int, str]] - Target of the animation channel.
            """
            sampler: int
            target: Dict[str, Union[int, str]]

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> 'Channel':
                """Creates a Channel instance from a dictionary."""
                return cls(
                    sampler=data["sampler"],
                    target=data["target"]
                )

        @dataclass
        class Sampler:
            """
            Represents a sampler in an animation.

            Attributes:
                input: int - Index of the input accessor.
                output: int - Index of the output accessor.
                interpolation: Optional[str] - Interpolation method.
            """
            input: int
            output: int
            interpolation: Optional[str] = None

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> 'Sampler':
                """Creates a Sampler instance from a dictionary."""
                return cls(
                    input=data["input"],
                    output=data["output"],
                    interpolation=data.get("interpolation")
                )

        name: Optional[str] = None
        channels: List[Channel] = field(default_factory=list)
        samplers: List[Sampler] = field(default_factory=list)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Animation':
            """Creates an Animation instance from a dictionary."""
            return cls(
                name=data.get("name"),
                channels=[GLTF.Animation.Channel.from_dict(chan) for chan in data.get("channels", [])],
                samplers=[GLTF.Animation.Sampler.from_dict(samp) for samp in data.get("samplers", [])]
            )

    animations: List[Animation] = field(default_factory=list)
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the GLTF instance to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the GLTF instance.
        """
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
        """
        Creates a GLTF instance from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary containing the GLTF data.

        Returns:
            GLTF: An instance of GLTF created from the dictionary.
        """
        return cls(
            scene=data.get("scene"),
            scenes=[GLTF.Scene.from_dict(scene) for scene in data.get("scenes", [])],
            cameras=[GLTF.Camera.from_dict(camera) for camera in data.get("cameras", [])],
            nodes=[GLTF.Node.from_dict(node) for node in data.get("nodes", [])],
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
        """
        Saves the GLTF instance to a file as JSON.

        Args:
            filename (str): The path to the file where the GLTF instance will be saved.
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_from_file(cls, filename: str) -> 'GLTF':
        """
        Loads a GLTF instance from a JSON file.

        Args:
            filename (str): The path to the file containing the GLTF data.

        Returns:
            GLTF: An instance of GLTF loaded from the file.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
            return cls.from_dict(data)

class TestGLTF(unittest.TestCase):

    def test_scene_from_dict(self):
        data = {"scene_name": "Scene 1", "nodes": [1, 2, 3]}
        scene = GLTF.Scene.from_dict(data)
        self.assertEqual(scene.name, "Scene 1")
        self.assertEqual(scene.nodes, [1, 2, 3])

    def test_camera_from_dict(self):
        data = {"type": "perspective", "data": {"aspectRatio": 1.0, "yfov": 0.8, "zfar": 1000.0, "znear": 0.1}}
        camera = GLTF.Camera.from_dict(data)
        self.assertEqual(camera.type, "perspective")
        self.assertEqual(camera.data, {"aspectRatio": 1.0, "yfov": 0.8, "zfar": 1000.0, "znear": 0.1})

    def test_node_from_dict(self):
        data = {
            "node_name": "Node 1",
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
        self.assertEqual(node.name, "Node 1")
        self.assertEqual(node.translation, glm.vec3(1.0, 2.0, 3.0))
        self.assertEqual(node.children, [1, 2])
        self.assertEqual(node.mesh, 1)

    def test_image_from_dict(self):
        data = {"uri": "image.png", "buffer_view": 0, "mime_type": "image/png"}
        image = GLTF.Image.from_dict(data)
        self.assertEqual(image.uri, "image.png")
        self.assertEqual(image.buffer_view, 0)
        self.assertEqual(image.mime_type, "image/png")

    def test_sampler_from_dict(self):
        data = {"mag_filter": 9729, "min_filter": 9729, "wrap_s": 10497, "wrap_t": 10497}
        sampler = GLTF.Sampler.from_dict(data)
        self.assertEqual(sampler.mag_filter, 9729)
        self.assertEqual(sampler.min_filter, 9729)
        self.assertEqual(sampler.wrap_s, 10497)
        self.assertEqual(sampler.wrap_t, 10497)

    def test_texture_from_dict(self):
        data = {"tex_name": "Texture 1", "source": 0, "sampler": 1}
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
        gltf.scenes.append(GLTF.Scene(scene_name="Test Scene", nodes=[0, 1]))
        gltf.cameras.append(
            GLTF.Camera(type="perspective", data={"aspectRatio": 1.0, "yfov": 0.8, "zfar": 1000.0, "znear": 0.1}))
        gltf.nodes.append(GLTF.Node(node_name="Test Node", translation=glm.vec3(1.0, 2.0, 3.0), children=[0], mesh=0))

        # Save to file
        with open('test_save_load_gltf.json', 'w') as f:
            json.dump(gltf.to_dict(), f)

        # Load from file
        with open('test_save_load_gltf.json', 'r') as f:
            loaded_data = json.load(f)

        loaded_gltf = GLTF.from_dict(loaded_data)
        self.assertEqual(len(loaded_gltf.scenes), 1)
        self.assertEqual(loaded_gltf.scenes[0].scene_name, "Test Scene")
        self.assertEqual(len(loaded_gltf.cameras), 1)
        self.assertEqual(loaded_gltf.cameras[0].type, "perspective")
        self.assertEqual(len(loaded_gltf.nodes), 1)
        self.assertEqual(loaded_gltf.nodes[0].node_name, "Test Node")


if __name__ == "__main__":
    unittest.main()
