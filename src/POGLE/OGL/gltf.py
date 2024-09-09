import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)  # Set your logging level

import unittest

from typing import Optional, List, Dict, Any, Union, Literal
from dataclasses import dataclass, field
import glm  # Assuming glm is imported
import json
from OpenGL.GL import *

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Literal
import glm
import json

@dataclass
class GLTF:
    """
    A class to represent a GLTF model.

    Attributes:
        scene (Optional[int]): The index of the default scene to be displayed.
        scenes (List[Scene]): A list of scenes in the GLTF file.
        cameras (List[Camera]): A list of cameras in the GLTF file.
        nodes (List[Node]): A list of nodes in the GLTF file.
        images (List[Image]): A list of images in the GLTF file.
        samplers (List[Sampler]): A list of samplers in the GLTF file.
        textures (List[Texture]): A list of textures in the GLTF file.
        materials (List[Material]): A list of materials in the GLTF file.
        meshes (List[Mesh]): A list of meshes in the GLTF file.
        accessors (List[Accessor]): A list of accessors in the GLTF file.
        buffer_views (List[BufferView]): A list of buffer views in the GLTF file.
        buffers (List[Buffer]): A list of buffers in the GLTF file.
        skins (List[Skin]): A list of skins in the GLTF file.
        animations (List[Animation]): A list of animations in the GLTF file.
    """
    scene: Optional[int] = None

    @dataclass
    class Scene:
        """
        A class to represent a Scene in the GLTF model.

        Attributes:
            name (Optional[str]): The name of the scene.
            nodes (List[int]): The list of node indices that are part of the scene.
        """
        name: Optional[str] = None
        nodes: List[int] = field(default_factory=list)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Scene':
            """
            Create a Scene object from a dictionary.

            Args:
                data (Dict[str, Any]): The dictionary to create the Scene from.

            Returns:
                Scene: The created Scene object.
            """
            return cls(
                name=data.get("name"),
                nodes=data.get("nodes", [])
            )

        def to_dict(self) -> Dict[str, Any]:
            """
            Convert the Scene object to a dictionary.

            Returns:
                Dict[str, Any]: The dictionary representation of the Scene.
            """
            return {
                "name": self.name,
                "nodes": self.nodes
            }

    scenes: List[Scene] = field(default_factory=list)

    @dataclass
    class Camera:
        """
        A class to represent a Camera in the GLTF model.

        Attributes:
            type (Literal['perspective', 'orthographic']): The type of the camera.
            data (Dict[str, float]): The camera data, which varies depending on the camera type.
        """
        type: Literal['perspective', 'orthographic']
        data: Dict[str, float]

        def __post_init__(self):
            """
            Validates the camera data depending on the camera type.
            Raises:
                ValueError: If required keys for the camera type are missing.
            """
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
            """
            Create a Camera object from a dictionary.

            Args:
                data (Dict[str, Any]): The dictionary to create the Camera from.

            Returns:
                Camera: The created Camera object.
            """
            return cls(
                type=data["type"],
                data=data["data"]
            )

        def to_dict(self) -> Dict[str, Any]:
            """
            Convert the Camera object to a dictionary.

            Returns:
                Dict[str, Any]: The dictionary representation of the Camera.
            """
            return {
                "type": self.type,
                "data": self.data
            }

    cameras: List[Camera] = field(default_factory=list)

    @dataclass
    class Node:
        """
        A class to represent a Node in the GLTF model.

        Attributes:
            name (str): The name of the node.
            matrix (Optional[glm.mat4]): The transformation matrix of the node.
            translation (Optional[glm.vec3]): The translation vector of the node.
            rotation (Optional[glm.quat]): The rotation quaternion of the node.
            scale (Optional[glm.vec3]): The scale vector of the node.
            children (List[int]): The indices of the node's children.
            mesh (Optional[int]): The index of the mesh associated with this node.
            skin (Optional[int]): The index of the skin associated with this node.
            camera (Optional[int]): The index of the camera associated with this node.
        """
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
            """
            Validates the Node attributes, ensuring that either `matrix` or the combination of `translation`, `rotation`,
            and `scale` are provided, but not both.
            Raises:
                ValueError: If neither or both `matrix` and the other transformation attributes are provided.
            """
            if self.matrix is None:
                if not (self.translation or self.rotation or self.scale):
                    raise ValueError("Either `matrix` or `translation`, `rotation`, and `scale` must be provided.")
            else:
                if (self.translation or self.rotation or self.scale):
                    raise ValueError("If `matrix` is provided, `translation`, `rotation`, and `scale` must be None.")

        @classmethod
        def from_dict(cls, data: Dict[str, Any], index: int = None) -> 'Node':
            """
            Create a Node object from a dictionary.

            Args:
                data (Dict[str, Any]): The dictionary to create the Node from.
                index (int, optional): The index of the node, used for naming if name is not provided.

            Returns:
                Node: The created Node object.
            """
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

        def to_dict(self) -> Dict[str, Any]:
            """
            Convert the Node object to a dictionary.

            Returns:
                Dict[str, Any]: The dictionary representation of the Node.
            """
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
        """
        A class to represent an Image in the GLTF model.

        Attributes:
            uri (Optional[str]): The URI of the image.
            buffer_view (Optional[int]): The index of the buffer view containing the image.
            mime_type (Optional[str]): The MIME type of the image.
        """
        uri: Optional[str] = None
        buffer_view: Optional[int] = None
        mime_type: Optional[str] = None

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Image':
            """
            Create an Image object from a dictionary.

            Args:
                data (Dict[str, Any]): The dictionary to create the Image from.

            Returns:
                Image: The created Image object.
            """
            return cls(
                uri=data.get("uri"),
                buffer_view=data.get("buffer_view"),
                mime_type=data.get("mime_type")
            )

    images: List[Image] = field(default_factory=list)

    @dataclass
    class Sampler:
        """
        A class to represent a Sampler in the GLTF model.

        Attributes:
            mag_filter (int): The magnification filter.
            min_filter (int): The minification filter.
            wrap_s (int): The wrapping mode for the S (U) coordinate.
            wrap_t (int): The wrapping mode for the T (V) coordinate.
        """
        mag_filter: int
        min_filter: int
        wrap_s: int
        wrap_t: int

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Sampler':
            """
            Create a Sampler object from a dictionary.

            Args:
                data (Dict[str, Any]): The dictionary to create the Sampler from.

            Returns:
                Sampler: The created Sampler object.
            """
            return cls(
                mag_filter=data["magFilter"],
                min_filter=data["minFilter"],
                wrap_s=data["wrapS"],
                wrap_t=data["wrapT"]
            )

    samplers: List[Sampler] = field(default_factory=list)

    @dataclass
    class Texture:
        """
        A class to represent a Texture in the GLTF model.

        Attributes:
            sampler (Optional[int]): The index of the sampler to use.
            source (Optional[int]): The index of the image source to use.
        """
        sampler: Optional[int] = None
        source: Optional[int] = None

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Texture':
            """
            Create a Texture object from a dictionary.

            Args:
                data (Dict[str, Any]): The dictionary to create the Texture from.

            Returns:
                Texture: The created Texture object.
            """
            return cls(
                sampler=data.get("sampler"),
                source=data.get("source")
            )

    textures: List[Texture] = field(default_factory=list)

    @dataclass
    class Material:
        """
        A class to represent a Material in the GLTF model.

        Attributes:
            name (Optional[str]): The name of the material.
            pbr_metallic_roughness (Dict[str, Any]): The PBR metallic-roughness properties of the material.
            normal_texture (Optional[int]): The index of the normal texture.
            occlusion_texture (Optional[int]): The index of the occlusion texture.
            emissive_texture (Optional[int]): The index of the emissive texture.
            emissive_factor (Optional[List[float]]): The emissive factor of the material.
            alpha_mode (Optional[str]): The alpha mode of the material.
            alpha_cutoff (Optional[float]): The alpha cutoff value.
            double_sided (Optional[bool]): Whether the material is double-sided.
        """
        name: Optional[str] = None
        pbr_metallic_roughness: Dict[str, Any] = field(default_factory=dict)
        normal_texture: Optional[int] = None
        occlusion_texture: Optional[int] = None
        emissive_texture: Optional[int] = None
        emissive_factor: Optional[List[float]] = None
        alpha_mode: Optional[str] = None
        alpha_cutoff: Optional[float] = None
        double_sided: Optional[bool] = None

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Material':
            """
            Create a Material object from a dictionary.

            Args:
                data (Dict[str, Any]): The dictionary to create the Material from.

            Returns:
                Material: The created Material object.
            """
            return cls(
                name=data.get("name"),
                pbr_metallic_roughness=data.get("pbrMetallicRoughness", {}),
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
        A class to represent a Mesh in the GLTF model.

        Attributes:
            name (Optional[str]): The name of the mesh.
            primitives (List[Dict[str, Any]]): A list of primitives making up the mesh.
        """
        name: Optional[str] = None
        primitives: List[Dict[str, Any]] = field(default_factory=list)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Mesh':
            """
            Create a Mesh object from a dictionary.

            Args:
                data (Dict[str, Any]): The dictionary to create the Mesh from.

            Returns:
                Mesh: The created Mesh object.
            """
            return cls(
                name=data.get("name"),
                primitives=data.get("primitives", [])
            )

    meshes: List[Mesh] = field(default_factory=list)

    @dataclass
    class Accessor:
        """
        A class to represent an Accessor in the GLTF model.

        Attributes:
            buffer_view (int): The index of the buffer view.
            byte_offset (int): The byte offset into the buffer view.
            component_type (int): The data type of the components.
            count (int): The number of elements.
            type (str): The type of the accessor.
            max (List[float]): The maximum values of the accessor.
            min (List[float]): The minimum values of the accessor.
        """
        buffer_view: int
        byte_offset: int
        component_type: int
        count: int
        type: str
        max: List[float] = field(default_factory=list)
        min: List[float] = field(default_factory=list)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Accessor':
            """
            Create an Accessor object from a dictionary.

            Args:
                data (Dict[str, Any]): The dictionary to create the Accessor from.

            Returns:
                Accessor: The created Accessor object.
            """
            return cls(
                buffer_view=data["bufferView"],
                byte_offset=data["byteOffset"],
                component_type=data["componentType"],
                count=data["count"],
                type=data["type"],
                max=data.get("max", []),
                min=data.get("min", [])
            )

    accessors: List[Accessor] = field(default_factory=list)

    @dataclass
    class BufferView:
        """
        A class to represent a BufferView in the GLTF model.

        Attributes:
            buffer (int): The index of the buffer.
            byte_offset (int): The byte offset into the buffer.
            byte_length (int): The length of the buffer view.
            byte_stride (Optional[int]): The stride between elements in the buffer view.
        """
        buffer: int
        byte_offset: int
        byte_length: int
        byte_stride: Optional[int] = None

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'BufferView':
            """
            Create a BufferView object from a dictionary.

            Args:
                data (Dict[str, Any]): The dictionary to create the BufferView from.

            Returns:
                BufferView: The created BufferView object.
            """
            return cls(
                buffer=data["buffer"],
                byte_offset=data["byteOffset"],
                byte_length=data["byteLength"],
                byte_stride=data.get("byteStride")
            )

    buffer_views: List[BufferView] = field(default_factory=list)

    @dataclass
    class Buffer:
        """
        A class to represent a Buffer in the GLTF model.

        Attributes:
            uri (Optional[str]): The URI of the buffer.
            byte_length (int): The length of the buffer in bytes.
        """
        uri: Optional[str] = None
        byte_length: int = 0

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Buffer':
            """
            Create a Buffer object from a dictionary.

            Args:
                data (Dict[str, Any]): The dictionary to create the Buffer from.

            Returns:
                Buffer: The created Buffer object.
            """
            return cls(
                uri=data.get("uri"),
                byte_length=data["byteLength"]
            )

    buffers: List[Buffer] = field(default_factory=list)

    @dataclass
    class Skin:
        """
        A class to represent a Skin in the GLTF model.

        Attributes:
            inverse_bind_matrices (Optional[int]): The index of the accessor containing the inverse bind matrices.
            skeleton (Optional[int]): The index of the skeleton node.
            joints (List[int]): The list of joints in the skin.
        """
        inverse_bind_matrices: Optional[int] = None
        skeleton: Optional[int] = None
        joints: List[int] = field(default_factory=list)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Skin':
            """
            Create a Skin object from a dictionary.

            Args:
                data (Dict[str, Any]): The dictionary to create the Skin from.

            Returns:
                Skin: The created Skin object.
            """
            return cls(
                inverse_bind_matrices=data.get("inverseBindMatrices"),
                skeleton=data.get("skeleton"),
                joints=data.get("joints", [])
            )

    skins: List[Skin] = field(default_factory=list)

    @dataclass
    class Animation:
        """
        A class to represent an Animation in the GLTF model.

        Attributes:
            name (Optional[str]): The name of the animation.
            channels (List[Dict[str, Any]]): A list of animation channels.
            samplers (List[Dict[str, Any]]): A list of animation samplers.
        """
        name: Optional[str] = None
        channels: List[Dict[str, Any]] = field(default_factory=list)
        samplers: List[Dict[str, Any]] = field(default_factory=list)

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> 'Animation':
            """
            Create an Animation object from a dictionary.

            Args:
                data (Dict[str, Any]): The dictionary to create the Animation from.

            Returns:
                Animation: The created Animation object.
            """
            return cls(
                name=data.get("name"),
                channels=data.get("channels", []),
                samplers=data.get("samplers", [])
            )

    animations: List[Animation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the GLTF object to a dictionary.

        Returns:
            Dict[str, Any]: The dictionary representation of the GLTF object.
        """
        return {
            "scene": self.scene,
            "scenes": [scene.to_dict() for scene in self.scenes],
            "cameras": [camera.to_dict() for camera in self.cameras],
            "nodes": [node.to_dict() for node in self.nodes],
            "images": [image.__dict__ for image in self.images],
            "samplers": [sampler.__dict__ for sampler in self.samplers],
            "textures": [texture.__dict__ for texture in self.textures],
            "materials": [material.__dict__ for material in self.materials],
            "meshes": [mesh.__dict__ for mesh in self.meshes],
            "accessors": [accessor.__dict__ for accessor in self.accessors],
            "bufferViews": [buffer_view.__dict__ for buffer_view in self.buffer_views],
            "buffers": [buffer.__dict__ for buffer in self.buffers],
            "skins": [skin.__dict__ for skin in self.skins],
            "animations": [animation.__dict__ for animation in self.animations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GLTF':
        """
        Create a GLTF object from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary to create the GLTF object from.

        Returns:
            GLTF: The created GLTF object.
        """
        return cls(
            scene=data.get("scene"),
            scenes=[GLTF.Scene.from_dict(s) for s in data.get("scenes", [])],
            cameras=[GLTF.Camera.from_dict(c) for c in data.get("cameras", [])],
            nodes=[GLTF.Node.from_dict(n, i) for i, n in enumerate(data.get("nodes", []))],
            images=[GLTF.Image.from_dict(i) for i in data.get("images", [])],
            samplers=[GLTF.Sampler.from_dict(s) for s in data.get("samplers", [])],
            textures=[GLTF.Texture.from_dict(t) for t in data.get("textures", [])],
            materials=[GLTF.Material.from_dict(m) for m in data.get("materials", [])],
            meshes=[GLTF.Mesh.from_dict(m) for m in data.get("meshes", [])],
            accessors=[GLTF.Accessor.from_dict(a) for a in data.get("accessors", [])],
            buffer_views=[GLTF.BufferView.from_dict(bv) for bv in data.get("bufferViews", [])],
            buffers=[GLTF.Buffer.from_dict(b) for b in data.get("buffers", [])],
            skins=[GLTF.Skin.from_dict(s) for s in data.get("skins", [])],
            animations=[GLTF.Animation.from_dict(a) for a in data.get("animations", [])],
        )

    @classmethod
    def from_file(cls, filename: str) -> 'GLTF':
        """
        Load a GLTF object from a JSON file.

        Args:
            filename (str): The path to the GLTF file.

        Returns:
            GLTF: The loaded GLTF object.
        """
        with open(filename, 'r') as file:
            data = json.load(file)
        return cls.from_dict(data)

    def to_file(self, filename: str):
        """
        Save the GLTF object to a JSON file.

        Args:
            filename (str): The path to the output file.
        """
        with open(filename, 'w') as file:
            json.dump(self.to_dict(), file, indent=2)


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
        self.assertEqual(camera.params, {"aspectRatio": 1.0, "yfov": 0.8, "zfar": 1000.0, "znear": 0.1})

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
