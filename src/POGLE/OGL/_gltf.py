from __future__ import annotations

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

import inspect


@dataclass
class Extension:
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "properties": self.properties}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Extension":
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            name=data.get("name", ""),
            properties=data.get("properties", {})
        )


class TestExtension(unittest.TestCase):
    def test_extension_from_init(self):
        extension = Extension(name="test", properties={"test": 1})
        self.assertEqual(extension.name, "test")
        self.assertEqual(extension.properties, {"test": 1})

    def test_extension_from_dict(self):
        data = {"name": "test", "properties": {"test": 1}}
        extension = Extension.from_dict(data)
        self.assertEqual(extension.name, "test")
        self.assertEqual(extension.properties, {"test": 1})

    def test_extension_to_dict(self):
        extension = Extension(name="test", properties={"test": 1})
        data = extension.to_dict()
        self.assertEqual(data, {"name": "test", "properties": {"test": 1}})


@dataclass
class Extra:
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"extra": self.data}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Extra':
        extra_data = data.get('extra', {})
        return Extra(data=extra_data)


class TestExtra(unittest.TestCase):
    def test_extra_from_init(self):
        extra = Extra(data={"test": 1})
        self.assertEqual(extra.data, {"test": 1})

    def test_extra_from_dict(self):
        data = {"extra": {"test": 1}}
        extra = Extra.from_dict(data)
        self.assertEqual(extra.data, {"test": 1})

    def test_extra_to_dict(self):
        extra = Extra(data={"test": 1})
        data = extra.to_dict()
        self.assertEqual(data, {"extra": {"test": 1}})


@dataclass
class Asset:
    version: str
    copyright: Optional[str]
    generator: Optional[str]
    minVersion: Optional[str]
    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self) -> Dict[str, Any]:
        asset_dict = {"version": self.version}
        if self.copyright:
            asset_dict["copyright"] = self.copyright
        if self.generator:
            asset_dict["generator"] = self.generator
        if self.minVersion:
            asset_dict["minVersion"] = self.minVersion
        if self.extensions:
            asset_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            asset_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return asset_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Asset":
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            version=data["version"],
            copyright=data.get("copyright"),
            generator=data.get("generator"),
            minVersion=data.get("minVersion"),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


@dataclass
class Camera:
    id: int
    type: Literal['perspective', 'orthographic']
    params: Dict[str, float]

    name: Optional[str] = None
    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

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
        if self.extensions:
            cam_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            cam_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return cam_dict

    @classmethod
    def from_dict(cls, id: int, data: Dict[str, Any]) -> 'Camera':
        cam_type = data["type"]
        params = data.get(cam_type)
        if not params:
            raise Exception(
                f"Camera id is of type {cam_type} but {cam_type} is not an attribute of camera id")
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            id=id,
            type=cam_type,
            name=data.get("name"),
            params=params,
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestCamera(unittest.TestCase):
    def test_perspective_from_init(self):
        cam = Camera(id=1, type="perspective", params={"aspectRatio": 1, "yfov": 1, "zfar": 1, "znear": 1})
        self.assertEqual(cam.params, {"aspectRatio": 1, "yfov": 1, "zfar": 1, "znear": 1})

    def test_orthographic_from_init(self):
        cam = Camera(id=1, type="orthographic", params={"xmag": 1, "ymag": 1, "zfar": 1, "znear": 1})
        self.assertEqual(cam.params, {"xmag": 1, "ymag": 1, "zfar": 1, "znear": 1})

    def test_perspective_to_dict(self):
        cam = Camera(id=1, type="perspective", params={"aspectRatio": 1, "yfov": 1, "zfar": 1, "znear": 1})
        self.assertEqual(cam.to_dict(),
                         {"type": "perspective", "perspective": {"aspectRatio": 1, "yfov": 1, "zfar": 1, "znear": 1}})

    def test_orthographic_to_dict(self):
        cam = Camera(id=1, type="orthographic", params={"xmag": 1, "ymag": 1, "zfar": 1, "znear": 1})
        self.assertEqual(cam.to_dict(),
                         {"type": "orthographic", "orthographic": {"xmag": 1, "ymag": 1, "zfar": 1, "znear": 1}})

    def test_perspective_from_dict(self):
        cam = Camera.from_dict(1, {"type": "perspective",
                                   "perspective": {"aspectRatio": 1, "yfov": 1, "zfar": 1, "znear": 1}})
        self.assertEqual(cam.params, {"aspectRatio": 1, "yfov": 1, "zfar": 1, "znear": 1})

    def test_orthographic_from_dict(self):
        cam = Camera.from_dict(1,
                               {"type": "orthographic", "orthographic": {"xmag": 1, "ymag": 1, "zfar": 1, "znear": 1}})
        self.assertEqual(cam.params, {"xmag": 1, "ymag": 1, "zfar": 1, "znear": 1})


@dataclass
class Buffer:
    id: int
    uri: str
    byte_length: int
    name: Optional[str] = None
    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

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

        if self.extensions:
            buffer_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            buffer_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return buffer_dict

    @classmethod
    def from_dict(cls, id: int, data: Dict[str, Any]) -> 'Buffer':
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            id=id,
            uri=data.get("uri"),
            byte_length=data["byteLength"],
            name=data.get("name"),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = Buffer(id=1, uri="foo", byte_length=100)
        self.compare_dict = {"uri": "foo", "byteLength": 100}

    def test_to_dict(self):
        self.assertEqual(self.buffer.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(Buffer.from_dict(1, self.compare_dict), self.buffer)

    def test_is_external(self):
        self.assertTrue(self.buffer.is_external())

    def test_is_inline(self):
        self.assertFalse(self.buffer.is_inline())


@dataclass
class BufferView:
    id: int
    buffer: Buffer
    byte_offset: GLuint
    byte_length: int
    target: Optional[int]
    byte_stride: int = 0
    name: Optional[str] = None

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self):
        bv_dict = {
            "buffer": self.buffer.id,
            "bufferOffset": self.byte_offset,
            "byteLength": self.byte_length
        }
        if self.byte_stride:
            bv_dict["byteStride"] = self.byte_stride
        if self.target:
            bv_dict["target"] = self.target
        if self.name:
            bv_dict["name"] = self.name
        if self.extensions:
            bv_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            bv_dict["extras"] = [extras.to_dict() for extras in self.extras]

    @classmethod
    def from_dict(cls, id: int, data: Dict[str, Any], buffers: List[Buffer]) -> 'BufferView':
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            id=id,
            buffer=buffers[data["buffer"]],
            byte_offset=data.get("byteOffset", 0),
            byte_length=data["byteLength"],
            byte_stride=data.get("byteStride"),
            target=data.get("target"),
            name=data.get("name"),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestBufferView(unittest.TestCase):

    def setUp(self):
        self.buffers = [Buffer(id=1, uri="foo", byte_length=100)]
        self.buffer_view = BufferView(id=1, buffer=self.buffers[0], byte_offset=0, byte_length=100)
        self.compare_dict = {"buffer": 1, "bufferOffset": 0, "byteLength": 100}

    def test_to_dict(self):
        self.assertEqual(self.buffer_view.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(BufferView.from_dict(1, self.compare_dict, self.buffers).to_dict(), self.compare_dict)


@dataclass
class SparseIndices:
    buffer_view: BufferView
    byte_offset: int
    component_type: int

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self):
        ind_dict = {
            "bufferView": self.buffer_view.id,
            "byteOffset": self.byte_offset,
            "componentType": self.component_type
        }
        if self.extensions:
            ind_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            ind_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return ind_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], buffer_views: List[BufferView]):
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            buffer_view=buffer_views[data["bufferView"]],
            byte_offset=data.get("byteOffset", 0),
            component_type=data["componentType"],
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestSparseIndices(unittest.TestCase):
    def setUp(self):
        self.buffers = [Buffer(id=1, uri="foo", byte_length=100)]
        self.buffer_views = [BufferView(id=1, buffer=self.buffers[0], byte_offset=0, byte_length=100)]
        self.sparse_indices = SparseIndices(buffer_view=self.buffer_views[0], byte_offset=0, component_type=1)
        self.compare_dict = {"bufferView": 1, "byteOffset": 0, "componentType": 1}

    def test_to_dict(self):
        self.assertEqual(self.sparse_indices.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(SparseIndices.from_dict(self.compare_dict, self.buffer_views).to_dict(), self.compare_dict)


@dataclass
class SparseValues:
    buffer_view: BufferView
    byte_offset: int

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self):
        val_dict = {
            "bufferView": self.buffer_view.id,
            "byteOffset": self.byte_offset,
        }
        if self.extensions:
            val_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            val_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return val_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], buffer_views: List[BufferView]):
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            buffer_view=buffer_views[data["bufferView"]],
            byte_offset=data.get("byteOffset", 0),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestSparseValues(unittest.TestCase):
    def setUp(self):
        self.buffers = [Buffer(id=1, uri="foo", byte_length=100)]
        self.buffer_views = [BufferView(id=1, buffer=self.buffers[0], byte_offset=0, byte_length=100)]
        self.sparse_values = SparseValues(buffer_view=self.buffer_views[0], byte_offset=0)
        self.compare_dict = {"bufferView": 1, "byteOffset": 0}

    def test_from_init(self):
        self.assertEqual(self.sparse_values.buffer_view, self.buffer_views[0])
        self.assertEqual(self.sparse_values.byte_offset, 0)

    def test_to_dict(self):
        self.assertEqual(self.sparse_values.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(self.sparse_values.from_dict({"bufferView": 1, "byteOffset": 0}, self.buffer_views),
                         self.sparse_values)


@dataclass
class Sparse:
    count: int
    indices: SparseIndices
    values: SparseValues

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self) -> Dict[str, Any]:
        sparse_dict = {
            "count": self.count,
            "indices": self.indices.to_dict(),
            "values": self.values.to_dict()
        }
        if self.extensions:
            sparse_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            sparse_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return sparse_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], buffer_views: List[BufferView]):
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            count=data["count"],
            indices=SparseIndices.from_dict(data, buffer_views),
            values=SparseValues.from_dict(data, buffer_views),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestSparse(unittest.TestCase):
    def setUp(self):
        self.buffers = [Buffer(id=1, uri="foo", byte_length=100)]
        self.buffer_views = [BufferView(id=1, buffer=self.buffers[0], byte_offset=0, byte_length=100)]
        self.sparse_indices = SparseIndices(buffer_view=self.buffer_views[0], byte_offset=0, component_type=1)
        self.sparse_values = SparseValues(buffer_view=self.buffer_views[0], byte_offset=0)
        self.sparse = Sparse(count=1, indices=self.sparse_indices, values=self.sparse_values)
        self.compare_dict = {"count": 1, "indices": self.sparse_indices.to_dict(),
                             "values": self.sparse_values.to_dict()}

    def test_to_dict(self):
        self.assertEqual(self.sparse.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(Sparse.from_dict(self.compare_dict, self.buffer_views).to_dict(), self.compare_dict)


@dataclass
class Accessor:
    id: int
    component_type: GLenum
    count: int
    type: Literal['SCALAR', 'VEC2', 'VEC3', 'VEC4', 'MAT2', 'MAT3', 'MAT4']
    buffer_view: Optional[BufferView] = None
    byte_offset: int = 0
    normalized: bool = False
    max: Optional[List[Number]] = None
    min: Optional[List[Number]] = None
    sparse: Optional[Sparse] = None
    name: Optional[str] = None

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

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
            acc_dict["bufferView"] = self.buffer_view.id
        if self.max:
            acc_dict["max"] = self.max
        if self.min:
            acc_dict["min"] = self.min
        if self.sparse:
            acc_dict["sparse"] = self.sparse.to_dict()
        if self.name:
            acc_dict["name"] = self.name
        if self.extensions:
            acc_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            acc_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return acc_dict

    @classmethod
    def from_dict(cls, id: int, data: Dict[str, Any], buffer_views: List[BufferView]) -> 'Accessor':
        extensions = data.get("extensions")
        extras = data.get("extras")
        sparse_data = data.get("sparse")
        return cls(
            id=id,
            buffer_view=buffer_views[data["bufferView"]],
            byte_offset=data.get("byteOffset", 0),
            component_type=data["componentType"],
            count=data["count"],
            type=data["type"],
            normalized=data.get("normalized", False),
            max=data.get("max"),
            min=data.get("min"),
            sparse=Sparse.from_dict(sparse_data, buffer_views) if sparse_data else None,
            name=data.get("name"),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestAccessor(unittest.TestCase):
    def setUp(self):
        self.buffers = [Buffer(id=1, uri="foo", byte_length=100)]
        self.buffer_views = [BufferView(id=1, buffer=self.buffers[0], byte_offset=0, byte_length=100)]
        self.accessor = Accessor(id=1, buffer_view=self.buffer_views[0], byte_offset=0, component_type=1, count=1,
                                 type="VEC3")
        self.compare_dict = {"bufferView": 1, "byteOffset": 0, "componentType": 1, "count": 1, "type": "VEC3"}

    def test_to_dict(self):
        self.assertEqual(self.accessor.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(Accessor.from_dict(self.compare_dict, self.buffer_views).to_dict(), self.compare_dict)


@dataclass
class Image:
    id: int
    uri: Optional[str]
    mime_type: Optional[str]
    buffer_view: Optional[BufferView]
    name: Optional[str] = None

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self):
        image_dict = {}
        if self.uri:
            image_dict["uri"] = self.uri
        if self.mime_type:
            image_dict["mimeType"] = self.mime_type
        if self.buffer_view:
            image_dict["bufferView"] = self.buffer_view.id
        if self.name:
            image_dict["name"] = self.name
        if self.extensions:
            image_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            image_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return image_dict

    @classmethod
    def from_dict(cls, id: int, data: Dict[str, Any], buffer_views: List[BufferView]) -> 'Image':
        buffer_view = data.get("buffer_view")
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            id=id,
            uri=data.get("uri"),
            mime_type=data.get("mime_type"),
            buffer_view=buffer_views[buffer_view] if buffer_view else None,
            name=data.get("name"),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestImage(unittest.TestCase):
    def setUp(self):
        self.buffers = [Buffer(id=1, uri="foo", byte_length=100)]
        self.buffer_views = [BufferView(id=1, buffer=self.buffers[0], byte_offset=0, byte_length=100)]
        self.image = Image(id=1, uri="foo", mime_type="bar", buffer_view=self.buffer_views[0])
        self.compare_dict = {"uri": "foo", "mimeType": "bar", "bufferView": 1}

    def test_to_dict(self):
        self.assertEqual(self.image.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(Image.from_dict(self.compare_dict, self.buffer_views).to_dict(), self.compare_dict)


@dataclass
class Sampler:
    id: int
    mag_filter: Optional[int]
    min_filter: Optional[int]
    wrap_s: Optional[int]
    wrap_t: Optional[int]
    name: Optional[str] = None

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

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
        if self.extensions:
            sampler_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            sampler_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return sampler_dict

    @classmethod
    def from_dict(cls, id: int, data: Dict[str, Any]) -> 'Sampler':
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            id=id,
            mag_filter=data.get("magFilter"),
            min_filter=data.get("minFilter"),
            wrap_s=data.get("wrapS"),
            wrap_t=data.get("wrapT"),
            name=data.get("name"),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = Sampler(id=1, mag_filter=2, min_filter=3, wrap_s=4, wrap_t=5)
        self.compare_dict = {"magFilter": 2, "minFilter": 3, "wrapS": 4, "wrapT": 5}

    def test_to_dict(self):
        self.assertEqual(self.sampler.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(Sampler.from_dict(self.compare_dict).to_dict(), self.compare_dict)


@dataclass
class Texture:
    id: int
    sampler: Optional[Sampler]
    source: Optional[int]
    name: Optional[str] = None

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self):
        tex_dict = {}
        if self.sampler:
            tex_dict["sampler"] = self.sampler.id
        if self.source:
            tex_dict["source"] = self.source
        if self.name:
            tex_dict["name"] = self.name
        if self.extensions:
            tex_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            tex_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return tex_dict

    @classmethod
    def from_dict(cls, id: int, data: Dict[str, Any], samplers: List[Sampler]) -> 'Texture':
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            id=id,
            sampler=samplers[data["sampler"]],
            source=data["source"],
            name=data.get("name", f"Texture_id"),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestTexture(unittest.TestCase):
    def setUp(self):
        self.sampler = Sampler(id=1, mag_filter=2, min_filter=3, wrap_s=4, wrap_t=5)
        self.buffers = [Buffer(id=1, uri="foo", byte_length=100)]
        self.buffer_views = [BufferView(id=1, buffer=self.buffers[0], byte_offset=0, byte_length=100)]
        self.image = Image(id=1, uri="foo", mime_type="bar", buffer_view=self.buffer_views[0])
        self.texture = Texture(id=1, sampler=self.sampler, source=self.image.id)
        self.compare_dict = {"sampler": 1, "source": 1}

    def test_to_dict(self):
        self.assertEqual(self.texture.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(Texture.from_dict(self.compare_dict, samplers=[self.sampler]).to_dict(), self.compare_dict)


@dataclass
class TextureInfo:
    texture: Texture
    tex_coord: GLuint

    extensions: Optional[List[Extension]]
    extras: Optional[List[Extra]]

    def to_dict(self):
        tex_info_dict = {
            "index": self.texture.id,
            "texCoord": self.tex_coord
        }
        if self.extensions:
            tex_info_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            tex_info_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return tex_info_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], textures: List[Texture]):
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            texture=textures[data["index"]],
            tex_coord=data.get("texCoord", 0),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestTextureInfo(unittest.TestCase):
    def setUp(self):
        self.texture = Texture(id=1, sampler=Sampler(id=1, mag_filter=2, min_filter=3, wrap_s=4, wrap_t=5))
        self.texture_info = TextureInfo(texture=self.texture, tex_coord=0)
        self.compare_dict = {"index": 1, "texCoord": 0}

    def test_to_dict(self):
        self.assertEqual(self.texture_info.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(TextureInfo.from_dict(self.compare_dict, textures=[self.texture]).to_dict(), self.compare_dict)


@dataclass
class NormalTextureInfo(TextureInfo):
    scale: Number

    def to_dict(self):
        nti_dict = super().to_dict()
        nti_dict["scale"] = self.scale
        return nti_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], textures: List[Texture]):
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            texture=textures[data["index"]],
            tex_coord=data.get("texCoord", 0),
            scale=data.get("scale", 1),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestNormalTextureInfo(unittest.TestCase):
    def setUp(self):
        self.texture = Texture(id=1, sampler=Sampler(id=1, mag_filter=2, min_filter=3, wrap_s=4, wrap_t=5))
        self.texture_info = NormalTextureInfo(texture=self.texture, tex_coord=0, scale=1)
        self.compare_dict = {"index": 1, "texCoord": 0, "scale": 1}

    def test_to_dict(self):
        self.assertEqual(self.texture_info.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(NormalTextureInfo.from_dict(self.compare_dict, textures=[self.texture]).to_dict(),
                         self.compare_dict)


@dataclass
class OcclusionTextureInfo(TextureInfo):
    strength: Number

    def to_dict(self):
        oti_dict = super().to_dict()
        oti_dict["strength"] = self.strength
        return oti_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], textures: List[Texture]):
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            texture=textures[data["index"]],
            tex_coord=data.get("texCoord", 0),
            strength=data.get("scale", 1),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestOcclusionTextureInfo(unittest.TestCase):
    def setUp(self):
        self.texture = Texture(id=1, sampler=Sampler(id=1, mag_filter=2, min_filter=3, wrap_s=4, wrap_t=5))
        self.texture_info = OcclusionTextureInfo(texture=self.texture, tex_coord=0, strength=1)
        self.compare_dict = {"index": 1, "texCoord": 0, "strength": 1}

    def test_to_dict(self):
        self.assertEqual(self.texture_info.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(OcclusionTextureInfo.from_dict(self.compare_dict, textures=[self.texture]).to_dict(),
                         self.compare_dict)


@dataclass
class PBRMetallicRoughness:
    # TODO: texture info referencing
    base_color_factor: List[Number]
    base_color_texture: Optional[TextureInfo]
    metallic_factor: Number
    roughness_factor: Number
    metallic_roughness_texture: Optional[TextureInfo]

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

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
        if self.extensions:
            pbr_mr_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            pbr_mr_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return pbr_mr_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], textures: List[Texture]):
        base_color_texture = data.get("baseColorTexture")
        metallic_roughness_texture = data.get("metallicRoughnessTexture")
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            base_color_factor=data.get("baseColorFactor", [1, 1, 1, 1]),
            base_color_texture=TextureInfo.from_dict(base_color_texture, textures) if base_color_texture else None,
            metallic_factor=data.get("metallicFactor", 1),
            roughness_factor=data.get("roughnessFactor", 1),
            metallic_roughness_texture=TextureInfo.from_dict(metallic_roughness_texture,
                                                             textures) if metallic_roughness_texture else None,
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestPBRMetallicRoughness(unittest.TestCase):
    def setUp(self):
        self.texture = Texture(id=1, sampler=Sampler(id=1, mag_filter=2, min_filter=3, wrap_s=4, wrap_t=5))
        self.pbr_metallic_roughness = PBRMetallicRoughness(
            base_color_factor=[1, 1, 1, 1],
            base_color_texture=TextureInfo(texture=self.texture, tex_coord=0),
            metallic_factor=1,
            roughness_factor=1,
            metallic_roughness_texture=TextureInfo(texture=self.texture, tex_coord=0)
        )
        self.compare_dict = {
            "baseColorFactor": [1, 1, 1, 1],
            "baseColorTexture": {"index": 1, "texCoord": 0},
            "metallicFactor": 1,
            "roughnessFactor": 1,
            "metallicRoughnessTexture": {"index": 1, "texCoord": 0}
        }

    def test_to_dict(self):
        self.assertEqual(self.pbr_metallic_roughness.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(PBRMetallicRoughness.from_dict(self.compare_dict, textures=[self.texture]).to_dict(),
                         self.compare_dict)


@dataclass
class Material:
    id: int
    pbr_metallic_roughness: Optional[PBRMetallicRoughness]
    normal_texture: Optional[NormalTextureInfo]
    occlusion_texture: Optional[OcclusionTextureInfo]
    emissive_texture: Optional[TextureInfo]
    emissive_factor: List[Number]
    alpha_mode: Literal['OPAQUE', 'BLEND', 'MASK']
    alpha_cutoff: Number
    double_sided: bool

    name: Optional[str] = None
    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self):
        mat_dict = {
            "emissiveFactor": self.emissive_factor,
            "alphaMode": self.alpha_mode,
            "alphaCutoff": self.alpha_cutoff,
            "doubleSided": self.double_sided
        }
        if self.name:
            mat_dict["name"] = self.name
        if self.extensions:
            mat_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            mat_dict["extras"] = [extras.to_dict() for extras in self.extras]
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
    def from_dict(cls, id: int, data: Dict[str, Any], textures: List[Texture]) -> 'Material':
        pbr_metallic_roughness = data.get("pbrMetallicRoughness")
        normal_texture = data.get("normalTexture")
        occlusion_texture = data.get("occlusionTexture")
        emissive_texture = data.get("emissiveTexture")
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            id=id,
            name=data.get("name"),
            pbr_metallic_roughness=
            PBRMetallicRoughness.from_dict(pbr_metallic_roughness, textures) if pbr_metallic_roughness else None,
            normal_texture=textures[normal_texture] if normal_texture else None,
            occlusion_texture=textures[occlusion_texture] if occlusion_texture else None,
            emissive_texture=textures[emissive_texture] if emissive_texture else None,
            emissive_factor=data.get("emissiveFactor", [0, 0, 0]),
            alpha_mode=data.get("alphaMode", "OPAQUE"),
            alpha_cutoff=data.get("alphaCutoff", 0.5),
            double_sided=data.get("doubleSided", False)
        )


class TestMaterial(unittest.TestCase):
    def setUp(self):
        self.texture = Texture(id=1, sampler=Sampler(id=1, mag_filter=2, min_filter=3, wrap_s=4, wrap_t=5))
        self.material = Material(
            id=1,
            pbr_metallic_roughness=PBRMetallicRoughness(
                base_color_factor=[1, 1, 1, 1],
                base_color_texture=TextureInfo(texture=self.texture, tex_coord=0),
                metallic_factor=1,
                roughness_factor=1,
                metallic_roughness_texture=TextureInfo(texture=self.texture, tex_coord=0)
            ),
            normal_texture=NormalTextureInfo(texture=self.texture, tex_coord=0),
            occlusion_texture=OcclusionTextureInfo(texture=self.texture, tex_coord=0),
            emissive_texture=TextureInfo(texture=self.texture, tex_coord=0),
            emissive_factor=[1, 1, 1],
            alpha_mode="OPAQUE",
            alpha_cutoff=0.5,
            double_sided=True
        )
        self.compare_dict = {
            "pbrMetallicRoughness": {
                "baseColorFactor": [1, 1, 1, 1],
                "baseColorTexture": {"index": 1, "texCoord": 0},
                "metallicFactor": 1,
                "roughnessFactor": 1,
                "metallicRoughnessTexture": {"index": 1, "texCoord": 0}
            },
            "normalTexture": {"index": 1, "texCoord": 0},
            "occlusionTexture": {"index": 1, "texCoord": 0},
            "emissiveTexture": {"index": 1, "texCoord": 0},
            "emissiveFactor": [1, 1, 1],
            "alphaMode": "OPAQUE",
            "alphaCutoff": 0.5,
            "doubleSided": True
        }

    def test_to_dict(self):
        self.assertEqual(self.material.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(Material.from_dict(1, self.compare_dict, [self.texture]).to_dict(), self.compare_dict)


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
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(attributes=data)


class TestAttributes(unittest.TestCase):
    def setUp(self):
        self.attributes = Attributes(attributes={"POSITION": 0, "NORMAL": 1})
        self.compare_dict = {"POSITION": 0, "NORMAL": 1}

    def test_to_dict(self):
        self.assertEqual(self.attributes, self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(Attributes.from_dict(self.compare_dict).to_dict(), self.compare_dict)


@dataclass
class MeshPrimitive:
    attributes: Attributes
    indices: Optional[Accessor] = None
    material: Optional[Material] = None
    mode: int = 4
    targets = None  # TODO: implement this, morph targets

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def __post_init__(self):
        if not isinstance(self.indices, Accessor) and self.indices is not None:
            raise TypeError("MeshPrimitive.indices must be an Accessor")

    def to_dict(self):
        mp_dict = {
            "attributes": self.attributes,
            "mode": self.mode
        }
        if self.indices:
            mp_dict["indices"] = self.indices.id
        if self.material:
            mp_dict["material"] = self.material.id
        if self.mode:
            mp_dict["mode"] = self.mode
        if self.targets:
            pass  # TODO: morph targets
        if self.extensions:
            mp_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            mp_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return mp_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], accessors: List[Accessor],
                  materials: Optional[List[Material]] = None) -> 'MeshPrimitive':
        indices = data.get("indices")  # index of indices accessor
        material = data.get("material")
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            attributes=Attributes.from_dict(data["attributes"]),
            indices=accessors[indices] if accessors else None,
            material=materials[material] if materials else None,
            mode=data.get("mode", 4),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestMeshPrimitive(unittest.TestCase):
    def setUp(self):
        self.attributes = Attributes(attributes={"POSITION": 0, "NORMAL": 1})
        self.compare_dict = {
            "attributes": {"POSITION": 0, "NORMAL": 1},
            "mode": 4
        }

    def test_to_dict(self):
        self.assertEqual(MeshPrimitive(attributes=self.attributes, mode=4).to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(MeshPrimitive.from_dict(self.compare_dict, [Accessor(0, "VEC3", "VEC3", 0),
                                                                     Accessor(1, "VEC3", "VEC3", 0)]).to_dict(),
                         self.compare_dict)


@dataclass
class Mesh:
    id: int
    primitives: List[MeshPrimitive]
    weights: Optional[List[Number]] = None
    name: Optional[str] = None

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert a Mesh into a dictionary

        Returns:
            A dictionary representing the Mesh
        """
        mesh_dict: Dict[str, Any] = {
            "primitives": [prim.to_dict() for prim in self.primitives]
        }
        if self.weights:
            # Store the morph target weights
            mesh_dict["weights"] = self.weights
        if self.name:
            # Store the name of the mesh
            mesh_dict["name"] = self.name
        if self.extensions:
            # Store any extensions
            mesh_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            # Store any extra data
            mesh_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return mesh_dict

    @classmethod
    def from_dict(cls, id: int, data: Dict[str, Any], accessors: Optional[List[Accessor]] = None,
                  materials: Optional[List[Material]] = None) -> 'Mesh':
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            id=id,
            primitives=[MeshPrimitive.from_dict(prim, accessors, materials) for prim in data["primitives"]],
            weights=data.get("weights"),
            name=data.get("name"),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestMesh(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh(0, [MeshPrimitive(attributes=Attributes(attributes={"POSITION": 0, "NORMAL": 1}), mode=4)])

    def test_to_dict(self):
        self.assertEqual(self.mesh.to_dict(), {"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "mode": 4}]})

    def test_from_dict(self):
        self.assertEqual(Mesh.from_dict(0, self.mesh.to_dict()).to_dict(), self.mesh.to_dict())


@dataclass
class Node:
    id: int
    camera: Optional[Union[int, Camera]] = None
    children: Optional[List[Union[int, Node]]] = None
    skin: Optional[Union[int, Skin]] = None
    matrix: Optional[glm.mat4] = None
    mesh: Optional[Mesh] = None
    rotation: Optional[glm.quat] = None
    scale: Optional[glm.vec3] = None
    translation: Optional[glm.vec3] = None
    weights: Optional[List[Number]] = None  # TODO: MORPH TARGET
    name: Optional[str] = None
    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self):
        node_dict = {
            "matrix": self.matrix.to_list(),
            "rotation": self.rotation.to_list(),
            "scale": self.scale.to_list(),
            "translation": self.translation.to_list()
        }
        if self.camera:
            node_dict["camera"] = self.camera.id
        if self.children:
            node_dict["children"] = [node.id for node in self.children]
        if self.skin:
            node_dict["skin"] = self.skin.id
        if self.mesh:
            node_dict["mesh"] = self.mesh.id
        if self.weights:
            node_dict["weights"] = self.weights
        if self.extensions:
            node_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            node_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return node_dict

    @classmethod
    def from_dict(cls, id: int, data: Dict[str, Any],
                  meshes: Optional[List[Mesh]] = None,
                  cameras: Optional[List[Camera]] = None,
                  nodes: Optional[List['Node']] = None,
                  skins: Optional[List['Skin']] = None) -> 'Node':
        camera = data.get("camera") if cameras else None
        children = data.get("children")
        skin = data.get("skin")
        matrix = data.get("matrix")
        mesh = data.get("mesh") if meshes else None
        rotation = data.get("rotation")
        scale = data.get("scale")
        translation = data.get("translation")

        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            id=id,
            camera=cameras[camera] if camera else None,
            children=[nodes[child] for child in children] if nodes else children,
            skin=skins[skin] if skins else skin,
            matrix=glm.mat4(matrix) if matrix else None,
            mesh=meshes[mesh] if mesh else None,
            rotation=glm.quat(rotation) if rotation else None,
            scale=glm.vec3(scale) if scale else None,
            translation=glm.vec3(translation) if translation else None,
            weights=data.get("weights"),
            name=data.get("name"),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )

    def second_pass(self, nodes: Optional[List['Node']] = None, skins: Optional[List['Skin']] = None):
        if skins:
            if self.skin is not None and isinstance(self.skin, int):
                self.skin = skins[self.skin]
        if nodes:
            if self.children and isinstance(self.children[0], int):
                for id, child_index in enumerate(self.children):
                    self.children[id] = nodes[child_index]
                    self.children[id].second_pass(nodes, skins)


@dataclass
class Scene:
    id: int
    nodes: Optional[List[Node]]
    name: Optional[str] = None

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self) -> Dict[str, Any]:
        scene_dict = {}
        if self.nodes:
            scene_dict["nodes"] = [node.to_dict() for node in self.nodes]
        if self.name:
            scene_dict["name"] = self.name
        if self.extensions:
            scene_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            scene_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return scene_dict

    @classmethod
    def from_dict(cls, id: int, data: Dict[str, Any], nodes: List[Node]) -> 'Scene':
        node_indices = data.get("nodes")
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            id=id,
            name=data.get("name"),
            nodes=[nodes[node] for node in node_indices] if node_indices != [] else None,
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestScene(unittest.TestCase):
    def setUp(self):
        self.nodes = [Node(0, name="node0"), Node(1, name="node1")]
        self.scene = Scene(0, name="scene0", nodes=self.nodes)
        self.compare_dict = {"nodes": [0, 1], "name": "scene0"}

    def test_to_dict(self):
        self.assertEqual(self.scene.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(Scene.from_dict(0, self.scene.to_dict(), self.nodes).to_dict(), self.compare_dict)


@dataclass
class Skin:
    id: int
    inverse_bind_matrices: Optional[Accessor]
    skeleton: Optional[Union[int, Node]]
    joints: List[Union[int, Node]]
    name: Optional[str] = None

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self):
        skin_dict = {
            "joints": [joint.id for joint in self.joints]
        }
        if self.inverse_bind_matrices:
            skin_dict["inverseBindMatrices"] = self.inverse_bind_matrices.id
        if self.skeleton:
            skin_dict["skeleton"] = self.skeleton.id
        if self.name:
            skin_dict["name"] = self.name
        if self.extensions:
            skin_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            skin_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return skin_dict

    @classmethod
    def from_dict(cls, id: int, data: Dict[str, Any], accessors: List[Accessor],
                  nodes: Optional[List[Node]] = None) -> 'Skin':
        inverse_bind_matrices = data.get("inverseBindMatrices")
        joints = data["joints"]
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            id=id,
            inverse_bind_matrices=accessors[inverse_bind_matrices] if inverse_bind_matrices else None,
            skeleton=data.get("skeleton"),
            joints=[nodes[joint_id] for joint_id in joints] if nodes else joints,
            name=data.get("name"),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )

    def second_pass(self, nodes: Optional[List['Node']] = None):
        if nodes:
            if self.skeleton is not None:
                self.skeleton = nodes[self.skeleton]
            if self.joints:
                for id, joint_index in enumerate(self.joints):
                    self.joints[id] = nodes[joint_index]


class TestSkin(unittest.TestCase):
    def setUp(self):
        self.skeleton = Node(0, name="skeleton")
        self.joints = [Node(1, name="joint1"), Node(2, name="joint2")]
        self.inverse_bind_matrices = Accessor(0, name="inverseBindMatrices", component_type=GL_FLOAT, count=16,
                                              type="MAT4")
        self.skin = Skin(0, skeleton=self.skeleton, joints=self.joints,
                         inverse_bind_matrices=self.inverse_bind_matrices, name="skin0")
        self.compare_dict = {"inverseBindMatrices": 0, "joints": [1, 2], "skeleton": 0, "name": "skin0"}

    def test_to_dict(self):
        self.assertEqual(self.skin.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(Skin.from_dict(0, self.skin.to_dict(), self.inverse_bind_matrices).to_dict(),
                         self.compare_dict)


class TestNode(unittest.TestCase):
    def setUp(self):
        self.existing_nodes = [Node(0, name="node0"), Node(1, name="node1"), Node(2, name="node2"),
                               Node(3, name="node3")]
        self.children = [self.existing_nodes[0], self.existing_nodes[1], self.existing_nodes[2]]
        self.joints = [[self.existing_nodes[0], self.existing_nodes[1]],
                       [self.existing_nodes[2], self.existing_nodes[3]]]
        self.skins = [Skin(0, joints=self.joints[0]), Skin(1, joints=self.joints[1])]
        self.root_a = Node(0, name="root_a", children=[self.existing_nodes[0], self.existing_nodes[1]],
                           skin=self.skins[0])
        self.root_b = Node(1, name="root_b", children=[self.existing_nodes[2], self.existing_nodes[3]],
                           skin=self.skins[1])
        self.compare_dicts = [{"children": [0, 1], "name": "root_a", "skin": 0},
                              {"children": [2, 3], "name": "root_b", "skin": 1}]

    def test_to_dict(self):
        self.assertEqual(self.root_a.to_dict(), self.compare_dicts[0])
        self.assertEqual(self.root_b.to_dict(), self.compare_dicts[1])

    def test_from_dict(self):
        self.assertEqual(Node.from_dict(0, self.root_a.to_dict(), self.existing_nodes).to_dict(), self.compare_dicts[0])
        self.assertEqual(Node.from_dict(1, self.root_b.to_dict(), self.existing_nodes).to_dict(), self.compare_dicts[1])


@dataclass
class AnimationSampler:
    id: int
    input: Accessor
    interpolation: Literal["LINEAR", "STEP", "CUBICSPLINE"]
    output: Accessor

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self):
        as_dict = {
            "input": self.input.id,
            "interpolation": self.interpolation,
            "output": self.output.id
        }
        if self.extensions:
            as_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            as_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return as_dict

    @classmethod
    def from_dict(cls, id: int, data: Dict[str, Any], accessors: List[Accessor]) -> 'AnimationSampler':
        extensions = data.get("extensions")
        extras = data.get("extras")
        input_index = data["input"]
        try:
            input_accessor = accessors[input_index]
        except:
            raise ValueError("Provided accessor index for input doesn't link to an existing accessor")
        output_index = data["output"]
        try:
            output_accessor = accessors[output_index]
        except:
            raise ValueError("Provided accessor index for output doesn't link to an existing accessor")
        return cls(
            id=id,
            input=input_accessor,
            output=output_accessor,
            interpolation=data.get("interpolation", "LINEAR"),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestAnimationSampler(unittest.TestCase):
    def setUp(self):
        self.input = Accessor(0, name="input", component_type=GL_FLOAT, count=4, type="VEC4")
        self.output = Accessor(1, name="output", component_type=GL_FLOAT, count=4, type="VEC4")
        self.compare_dict = {"input": 0, "output": 1, "interpolation": "LINEAR"}

    def test_to_dict(self):
        self.assertEqual(self.input.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(AnimationSampler.from_dict(0, self.input.to_dict(), [self.input, self.output]).to_dict(),
                         self.compare_dict)


@dataclass
class AnimationChannelTarget:
    node: Optional[Node]
    path: str

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self):
        act_dict = {
            "path": self.path
        }
        if self.node:
            act_dict["node"] = self.node.id
        if self.extensions:
            act_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            act_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return act_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], nodes: List[Node]):
        node = data.get("node")
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            node=nodes[node] if node else None,
            path=data["path"],
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestAnimationChannelTarget(unittest.TestCase):
    def setUp(self):
        self.node = Node(0, name="node")
        self.compare_dict = {"path": "translation", "node": 0}

    def test_to_dict(self):
        self.assertEqual(self.node.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(AnimationChannelTarget.from_dict(self.compare_dict, [self.node]).to_dict(), self.compare_dict)


@dataclass
class AnimationChannel:
    sampler: AnimationSampler
    target: AnimationChannelTarget

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self):
        ac_dict = {
            "sampler": self.sampler.id,
            "target": self.target.to_dict()
        }
        if self.extensions:
            ac_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            ac_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return self.extensions

    @classmethod
    def from_dict(cls, data: Dict[str, Any], animation_samplers: List[AnimationSampler],
                  nodes: List[Node]) -> 'AnimationChannel':
        sampler = data["sampler"]
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            sampler=animation_samplers[sampler] if sampler else None,
            target=AnimationChannelTarget.from_dict(data["target"], nodes),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestAnimationChannel(unittest.TestCase):
    def setUp(self):
        self.sampler = AnimationSampler(0)
        self.target = AnimationChannelTarget(node=None, path="translation")
        self.compare_dict = {"sampler": 0, "target": self.target.to_dict()}

    def test_to_dict(self):
        self.assertEqual(self.sampler.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(AnimationChannel.from_dict(self.compare_dict, [self.sampler], [self.target]).to_dict(),
                         self.compare_dict)


@dataclass
class Animation:
    id: int
    channels: List[AnimationChannel]
    samplers: List[AnimationSampler]
    name: Optional[str] = None

    extensions: Optional[List[Extension]] = None
    extras: Optional[List[Extra]] = None

    def to_dict(self):
        anim_dict = {
            "channels": [channel.to_dict() for channel in self.channels],
            "samplers": [sampler.to_dict() for sampler in self.samplers],
        }
        if self.name:
            anim_dict["name"] = self.name
        if self.extensions:
            anim_dict["extensions"] = [extension.to_dict() for extension in self.extensions]
        if self.extras:
            anim_dict["extras"] = [extras.to_dict() for extras in self.extras]
        return anim_dict

    @classmethod
    def from_dict(cls, id: int, data: Dict[str, Any], accessors: List[Accessor], nodes: List[Node]) -> 'Animation':
        channels = data.get("channels")
        if not channels:
            raise ValueError("Animations require at least one sampler")
        samplers = data.get("samplers")
        if not samplers:
            raise ValueError("Animations require at least one sampler")

        samplers = [AnimationSampler.from_dict(id, sampler, accessors) for id, sampler in enumerate(samplers)]
        channels = [AnimationChannel.from_dict(channel, samplers, nodes) for channel in channels]
        extensions = data.get("extensions")
        extras = data.get("extras")
        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            id=id,
            channels=channels,
            samplers=samplers,
            name=data.get("name"),
            extensions=[Extension.from_dict(extension_data) for extension_data in extensions] if extensions else None,
            extras=[Extra.from_dict(extra_data) for extra_data in extras] if extras else None
        )


class TestAnimation(unittest.TestCase):
    def setUp(self):
        self.sampler = AnimationSampler(0)
        self.target = AnimationChannelTarget(node=None, path="translation")
        self.channel = AnimationChannel(sampler=self.sampler, target=self.target)
        self.animation = Animation(0, [self.channel], [self.sampler])
        self.compare_dict = {"channels": [self.channel.to_dict()], "samplers": [self.sampler.to_dict()]}

    def test_to_dict(self):
        self.assertEqual(self.animation.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(Animation.from_dict(0, self.compare_dict, [self.sampler], [self.target]).to_dict(),
                         self.compare_dict)


@dataclass
class GLTF:
    extensions_used: Optional[List[str]]

    extensions_required: Optional[List[str]]

    accessors: Optional[List[Accessor]]

    animations: Optional[List[Animation]]

    asset: Asset

    buffers: Optional[List[Buffer]]

    buffer_views: Optional[List[BufferView]]

    cameras: Optional[List[Camera]]

    images: Optional[List[Image]]

    materials: Optional[List[Material]]

    meshes: Optional[List[Mesh]]

    nodes: Optional[List[Node]]

    samplers: Optional[List[Sampler]]

    scene: Optional[Scene]

    scenes: Optional[List[Scene]]

    skins: Optional[List[Skin]]

    textures: Optional[List[Texture]]

    extensions: Optional[List[Extension]] = None

    extras: Optional[List[Extra]] = None

    def _new_from_dict(self, cls, data: Dict[str, Any], id=0):
        if cls == Accessor:
            self.new_accessor(Accessor.from_dict(id, data, self.buffer_views), None, None)
        elif cls == Animation:
            self.new_animation(Animation.from_dict(id, data, self.accessors, self.nodes), None)
        elif cls == Buffer:
            self.new_buffer(Buffer.from_dict(id, data))
        elif cls == BufferView:
            self.new_buffer_view(BufferView.from_dict(id, data, self.buffers), None)
        elif cls == Camera:
            self.new_camera(Camera.from_dict(id, data), None)
        elif cls == Image:
            self.new_image(Image.from_dict(id, data, self.buffer_views))
        elif cls == Material:
            self.new_material(Material.from_dict(id, data, self.textures))
        elif cls == Mesh:
            self.new_mesh(Mesh.from_dict(id, data, self.materials))
        elif cls == Node:
            self.new_node(Node.from_dict(id, data, self.meshes, self.cameras, self.nodes, self.skins))
        elif cls == Sampler:
            self.new_sampler(Sampler.from_dict(id, data))
        elif cls == Scene:
            self.new_scene(Scene.from_dict(id, data, self.nodes))
        elif cls == Skin:
            self.new_skin(Skin.from_dict(id, data, self.accessors, self.nodes))
        elif cls == Texture:
            self.new_texture(Texture.from_dict(id, data, self.samplers))

    def new_accessor(self, component_type: Union[int, Accessor], count: int, type: str,
                     buffer_view: Optional[Union[int, BufferView]] = None, byte_offset: int = 0,
                     normalized: bool = False, max: Optional[List[Number]] = None, min: Optional[List[Number]] = None,
                     sparse: Optional[Sparse] = None, name: Optional[str] = None,
                     extensions: Optional[List[Extension]] = None, extras: Optional[List[Extra]] = None):
        num_accessors = len(self.accessors)
        if not isinstance(component_type, Accessor):
            is_bv_obj = isinstance(buffer_view, BufferView)
            if is_bv_obj:
                bvi = buffer_view.id
            else:
                bvi = buffer_view
            if bvi >= len(self.buffer_views):
                if is_bv_obj:
                    self.new_buffer_view(buffer_view)
                else:
                    raise ValueError("Accessor Being Added Links to a non-existent buffer view")
            elif not is_bv_obj:
                buffer_view = self.buffer_views[bvi]

            accessor = Accessor(
                id=num_accessors,
                buffer_view=buffer_view,
                byte_offset=byte_offset,
                component_type=component_type,
                normalized=normalized,
                count=count, type=type,
                max=max, min=min,
                sparse=sparse, name=name,
                extensions=extensions, extras=extras
            )
        else:
            accessor = component_type
            if accessor.id < num_accessors:
                raise ValueError(f"Accessor {accessor.id} is being added to gltf despite"
                                 f" an accessor already existing at this id")
        self.accessors.append(accessor)

    def new_animation(self, channels: Union[Animation, List[AnimationChannel]], samplers: List[AnimationSampler],
                      name: Optional[str] = None,
                      extensions: Optional[List[Extension]] = None, extras: Optional[List[Extra]] = None):
        num_animations = len(self.animations)
        if not isinstance(channels, Animation):
            animation = Animation(
                id=num_animations, channels=channels,
                samplers=samplers, name=name,
                extensions=extensions, extras=extras
            )
        else:
            animation = channels
            if animation.id < num_animations:
                raise ValueError(f"Animation {animation.id} is being added to gltf despite"
                                 f" an animation already existing at this id")
        self.animations.append(animation)

    def new_buffer(self, byte_length: Union[int, Buffer], uri: Optional[str] = None, name: Optional[str] = None,
                   extensions: Optional[List[Extension]] = None, extras: Optional[List[Extra]] = None):
        num_buffers = len(self.buffers)
        if not isinstance(byte_length, Buffer):
            buffer = Buffer(
                id=num_buffers, uri=uri,
                byte_length=byte_length, name=name,
                extensions=extensions, extras=extras
            )
        else:
            buffer = byte_length
            if buffer.id < num_buffers:
                raise ValueError(f"Buffer {buffer.id} is being added to gltf despite"
                                 f" a buffer already existing at this id")
        self.buffers.append(buffer)

    def new_buffer_view(self, buffer: Union[int, Buffer, BufferView], byte_length: int, byte_offset: int = 0,
                        byte_stride: Optional[int] = None, target: Optional[GLenum] = None, name: Optional[str] = None,
                        extensions: Optional[List[Extension]] = None, extras: Optional[List[Extra]] = None):
        num_buffer_views = len(self.buffer_views)
        if not isinstance(buffer, BufferView):
            is_buf_obj = isinstance(buffer, Buffer)
            if is_buf_obj:
                bid = buffer.id
            else:
                bid = buffer
            if bid >= len(self.buffers):
                if is_buf_obj:
                    self.new_buffer(buffer)
                else:
                    raise ValueError("Buffer View Being Added Links to a non-existent buffer")
            elif not is_buf_obj:
                buffer = self.buffers[bid]

            buffer_view = BufferView(
                id=num_buffer_views,
                buffer=buffer,
                byte_offset=byte_offset,
                byte_length=byte_length,
                byte_stride=byte_stride,
                target=target, name=name,
                extensions=extensions, extras=extras
            )
        else:
            buffer_view = buffer
            if buffer_view.id < num_buffer_views:
                raise ValueError(f"Buffer View {buffer_view.id} is being added to gltf despite"
                                 f" a buffer view already existing at this id")
        self.buffer_views.append(buffer_view)

    def new_camera(self, type: Union[Camera, Literal["orthographic", "perspective"]], params: Dict[str, float],
                   name: Optional[str] = None,
                   extensions: Optional[List[Extension]] = None, extras: Optional[List[Extra]] = None):
        num_cameras = len(self.cameras)
        if not isinstance(type, Camera):
            if type not in ["orthographic", "perspective"]:
                raise TypeError("Invalid camera type provided")
            camera = Camera(
                id=num_cameras, type=type,
                params=params, name=name,
                extensions=extensions, extras=extras
            )
        else:
            camera = type
            if camera.id < num_cameras:
                raise ValueError(f"Camera {camera.id} is being added to gltf despite"
                                 f" a camera already existing at this id")
        self.cameras.append(camera)

    def new_image(self, uri: Optional[Union[str, Image]], mime_type: Optional[str] = None,
                  buffer_view: Optional[Union[int, BufferView]] = None, name: Optional[str] = None,
                  extensions: Optional[List[Extension]] = None, extras: Optional[List[Extra]] = None):
        num_images = len(self.images)
        if not isinstance(uri, Image):
            is_bv_obj = isinstance(buffer_view, BufferView)
            if is_bv_obj:
                bvi = buffer_view.id
            else:
                bvi = buffer_view
            if bvi >= len(self.buffer_views):
                if is_bv_obj:
                    self.new_buffer_view(buffer_view, None)
                else:
                    raise ValueError("Image Being Added Links to a non-existent Buffer View")
            elif not is_bv_obj:
                buffer_view = self.buffer_views[bvi]

            image = Image(
                id=num_images, uri=uri, mime_type=mime_type,
                buffer_view=buffer_view, name=name,
                extensions=extensions, extras=extras
            )
        else:
            image = uri
            if image.id < num_images:
                raise ValueError(f"Image {image.id} is being added to gltf despite"
                                 f" a image already existing at this id")
        self.images.append(image)

    def new_material(self, name: Optional[Union[Material, str]] = None, extensions: Optional[List[Extension]] = None,
                     extras: Optional[List[Extra]] = None,
                     pbr_metallic_roughness: Optional[PBRMetallicRoughness] = None,
                     normal_texture: Optional[NormalTextureInfo] = None,
                     occlusion_texture: Optional[OcclusionTextureInfo] = None,
                     emissive_texure: Optional[TextureInfo] = None, emissive_factor: glm.vec3 = glm.vec3(),
                     alpha_mode: str = "OPAQUE", alpha_cutoff: Number = 0.5, double_sided: bool = False):
        num_materials = len(self.materials)
        if not isinstance(name, Material):
            material = Material(
                id=num_materials, name=name,
                extensions=extensions, extras=extras,
                pbr_metallic_roughness=pbr_metallic_roughness,
                normal_texture=normal_texture,
                occlusion_texture=occlusion_texture,
                emissive_texture=emissive_texure,
                emissive_factor=emissive_factor,
                alpha_mode=alpha_mode, alpha_cutoff=alpha_cutoff,
                double_sided=double_sided
            )
        else:
            material = name
            if material.id < num_materials:
                raise ValueError(f"Material {material.id} is being added to gltf despite"
                                 f" a material already existing at this id")
        self.materials.append(material)

    def new_mesh(self, primitives: Union[Mesh, List[MeshPrimitive]], weights: Optional[List[Number]] = None,
                 name: Optional[str] = None,
                 extensions: Optional[List[Extension]] = None, extras: Optional[List[Extra]] = None):
        num_meshes = len(self.meshes)
        if not isinstance(primitives, Mesh):
            mesh = Mesh(
                id=num_meshes, primitives=primitives,
                weights=weights, name=name,
                extensions=extensions, extras=extras
            )
        else:
            mesh = primitives
            if mesh.id < num_meshes:
                raise ValueError(f"Mesh {mesh.id} is being added to gltf despite"
                                 f" a mesh already existing at this id")
        self.meshes.append(mesh)

    def new_node(self, camera: Optional[Union[int, Camera, Node]], children: Optional[List[Union[int, Node]]] = None,
                 skin: Optional[Union[int, Skin]] = None, matrix: Optional[glm.mat4] = glm.mat4(),
                 mesh: Optional[Union[int, Mesh]] = None, rotation: Optional[glm.quat] = glm.quat(),
                 scale: Optional[glm.vec3] = glm.vec3(1), translation: Optional[glm.vec3] = glm.vec3(1),
                 weights: Optional[List[Number]] = None, name: Optional[str] = None,
                 extensions: Optional[List[Extension]] = None, extras: Optional[List[Extra]] = None):
        num_nodes = len(self.nodes)
        if not isinstance(camera, Node):
            if camera:
                is_cam_obj = isinstance(camera, Camera)
                if is_cam_obj:
                    cid = camera.id
                else:
                    cid = camera
                if cid >= len(self.cameras):
                    if is_cam_obj:
                        self.new_camera(camera, None)
                    else:
                        raise ValueError("Node Being Added Links to a non-existent Camera")
                elif not is_cam_obj:
                    camera = self.cameras[cid]
            if children:
                is_child_node_obj = isinstance(children[0], Node)
                for i, child in enumerate(children):
                    if is_child_node_obj:
                        cni = child.id
                    else:
                        cni = child
                    if cni >= len(self.nodes):
                        if is_child_node_obj:
                            self.new_node(child, None)
                        else:
                            raise ValueError("Node Being Added Links to a non-existent Child Node")
                    elif not is_child_node_obj:
                        children[i] = self.nodes[cni]
            if skin:
                is_skin_obj = isinstance(skin, Camera)
                if is_skin_obj:
                    sid = skin.id
                else:
                    sid = skin
                if sid >= len(self.skins):
                    if is_skin_obj:
                        self.new_skin(skin, None)
                    else:
                        raise ValueError("Node Being Added Links to a non-existent skin")
                elif not is_skin_obj:
                    skin = self.skins[sid]
            if mesh:
                is_mesh_obj = isinstance(mesh, Camera)
                if is_mesh_obj:
                    mid = mesh.id
                else:
                    mid = mesh
                if mid >= len(self.meshes):
                    if is_mesh_obj:
                        self.new_mesh(mesh, None)
                    else:
                        raise ValueError("Node Being Added Links to a non-existent mesh")
                elif not is_mesh_obj:
                    mesh = self.meshes[mid]
            node = Node(
                id=num_nodes, camera=camera, children=children,
                skin=skin, matrix=matrix, mesh=mesh,
                rotation=rotation, scale=scale,
                translation=translation,
                weights=weights, name=name,
                extensions=extensions, extras=extras
            )
        else:
            node = camera
            if node.id < num_nodes:
                raise ValueError(f"Node {node.id} is being added to gltf despite"
                                 f" a node already existing at this id")
        self.nodes.append(node)

    def new_sampler(self, mag_filter: Optional[Union[int, Sampler]], min_filter: Optional[int] = None,
                    wrap_s: Optional[GLenum] = GL_REPEAT, wrap_t: Optional[GLenum] = GL_REPEAT,
                    name: Optional[str] = None,
                    extensions: Optional[List[Extension]] = None, extras: Optional[List[Extra]] = None):
        num_samplers = len(self.samplers)
        if not isinstance(mag_filter, Image):
            sampler = Sampler(
                id=num_samplers, mag_filter=mag_filter, min_filter=min_filter,
                wrap_s=wrap_s, wrap_t=wrap_t, name=name,
                extensions=extensions, extras=extras
            )
        else:
            sampler = mag_filter
            if sampler.id < num_samplers:
                raise ValueError(f"Sampler {sampler.id} is being added to gltf despite"
                                 f" a sampler already existing at this id")
        self.samplers.append(sampler)

    def new_scene(self, nodes: Optional[Union[List[Union[int, Scene]], Scene]] = None, name: Optional[str] = None,
                  extensions: Optional[List[Extension]] = None, extras: Optional[List[Extra]] = None):
        num_scenes = len(self.scenes)
        if not isinstance(nodes, Scene):
            if nodes:
                is_node_obj = isinstance(nodes[0], Node)
                for i, node in enumerate(nodes):
                    if is_node_obj:
                        nid = node.id
                    else:
                        nid = node
                    if nid >= len(self.nodes):
                        if is_node_obj:
                            self.new_node(node, None)
                        else:
                            raise ValueError("Scene Being Added Links to a non-existent Node")
                    elif not is_node_obj:
                        nodes[i] = self.nodes[nid]
            scene = Scene(
                id=num_scenes, nodes=nodes, name=name,
                extensions=extensions, extras=extras
            )
        else:
            scene = nodes
            if scene.id < num_scenes:
                raise ValueError(f"Scene {scene.id} is being added to gltf despite"
                                 f" a scene already existing at this id")
        self.scenes.append(scene)

    def new_skin(self, joints: Union[List[Union[int, Node]], Skin] = None,
                 inverse_bind_matrices: Optional[Union[int, Accessor]] = None,
                 skeleton: Optional[Union[int, Node]] = None,
                 name: Optional[str] = None, extensions: Optional[List[Extension]] = None,
                 extras: Optional[List[Extra]] = None):
        num_skins = len(self.skins)
        if not isinstance(joints, Skin):
            if inverse_bind_matrices:
                is_ibm_acc_obj = isinstance(inverse_bind_matrices, Accessor)
                if is_ibm_acc_obj:
                    ibmaid = inverse_bind_matrices.id
                else:
                    ibmaid = inverse_bind_matrices
                if ibmaid >= len(self.accessors):
                    if is_ibm_acc_obj:
                        self.new_accessor(inverse_bind_matrices, None, None)
                    else:
                        raise ValueError("Skin Being Added Links to a non-existent accessor")
                elif not is_ibm_acc_obj:
                    inverse_bind_matrices = self.accessors[ibmaid]
            if skeleton:
                is_sknode_obj = isinstance(skeleton, Node)
                if is_sknode_obj:
                    cid = skeleton.id
                else:
                    cid = skeleton
                if cid >= len(self.nodes):
                    if is_sknode_obj:
                        self.new_node(skeleton)
                    else:
                        raise ValueError("Skin Being Added Links to a non-existent Node")
                elif not is_sknode_obj:
                    skeleton = self.nodes[cid]

            is_joint_obj = isinstance(joints[0], Node)
            for i, node in enumerate(joints):
                if is_joint_obj:
                    jid = node.id
                else:
                    jid = node
                if jid >= len(self.nodes):
                    if is_joint_obj:
                        self.new_node(node, None)
                    else:
                        raise ValueError("Skin Being Added Links to a non-existent Node")
                elif not is_joint_obj:
                    joints[i] = self.nodes[jid]
            skin = Skin(
                id=num_skins, inverse_bind_matrices=inverse_bind_matrices,
                skeleton=skeleton, joints=joints, name=name,
                extensions=extensions, extras=extras
            )
        else:
            skin = joints
            if skin.id < num_skins:
                raise ValueError(f"Skin {skin.id} is being added to gltf despite"
                                 f" a skin already existing at this id")
        self.skins.append(skin)

    def new_texture(self, sampler: Optional[Union[int, Sampler, Texture]] = None,
                    source: Optional[Union[int, Image]] = None, name: Optional[str] = None,
                    extensions: Optional[List[Extension]] = None, extras: Optional[List[Extra]] = None):
        num_textures = len(self.textures)
        if not isinstance(sampler, Texture):
            if sampler:
                is_sam_obj = isinstance(sampler, Sampler)
                if is_sam_obj:
                    sid = sampler.id
                else:
                    sid = sampler
                if sid >= len(self.samplers):
                    if is_sam_obj:
                        self.new_sampler(sampler)
                    else:
                        raise ValueError("Texture Being Added Links to a non-existent sampler")
                elif not is_sam_obj:
                    sampler = self.samplers[sid]
            if source:
                is_img_obj = isinstance(source, Image)
                if is_img_obj:
                    imgid = source.id
                else:
                    imgid = source
                if imgid >= len(self.nodes):
                    if is_img_obj:
                        self.new_image(source)
                    else:
                        raise ValueError("Texture Being Added Links to a non-existent Image")
                elif not is_img_obj:
                    source = self.images[imgid]
            texture = Texture(
                id=num_textures, sampler=sampler,
                source=source, name=name,
                extensions=extensions, extras=extras
            )
        else:
            texture = sampler
            if texture.id < num_textures:
                raise ValueError(f"Texture {texture.id} is being added to gltf despite"
                                 f" a texture already existing at this id")
        self.textures.append(texture)

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
    def from_dict(cls, data: Dict[str, Any]) -> '':
        extensions_used = data.get("extensionsUsed")
        extensions_required = data.get("extensionsRequired")
        asset_data = data["asset"]
        cameras_data = data.get("cameras")
        buffers_data = data.get("buffers")
        buffer_views_data = data.get("bufferViews")
        accessors_data = data.get("accessors")
        images_data = data.get("images")
        samplers_data = data.get("samplers")
        textures_data = data.get("textures")
        materials_data = data.get("materials")
        meshes_data = data.get("meshes")
        skins_data = data.get("skins")
        nodes_data = data.get("nodes")
        scenes_data = data.get("scenes")
        default_scene_index = data.get("scene")
        animations_data = data.get("animations")

        asset = Asset.from_dict(asset_data)

        cameras = None
        if cameras_data:
            cameras = [
                Camera.from_dict(
                    id=id,
                    data=camera_data
                ) for id, camera_data in enumerate(cameras_data)
            ]

        buffers = None
        if buffers_data:
            buffers = [
                Buffer.from_dict(
                    id=id,
                    data=buffer_data
                ) for id, buffer_data in enumerate(buffers_data)
            ]

        buffer_views = None
        if buffer_views_data:
            buffer_views = [
                BufferView.from_dict(
                    id=id,
                    data=buffer_view_data,
                    buffers=buffers
                ) for id, buffer_view_data in enumerate(buffer_views_data)
            ]

        accessors = None
        if accessors_data:
            accessors = [
                Accessor.from_dict(
                    id=id,
                    data=accessor_data,
                    buffer_views=buffer_views
                ) for id, accessor_data in accessors_data
            ]

        images = None
        if images_data:
            images = [
                Image.from_dict(
                    id=id,
                    data=image_data,
                    buffer_views=buffer_views
                ) for id, image_data in enumerate(images_data)
            ]

        samplers = None
        if samplers_data:
            samplers = [
                Sampler.from_dict(
                    id=id,
                    data=sampler_data
                ) for id, sampler_data in enumerate(samplers_data)
            ]

        textures = None
        if textures_data:
            textures = [
                Texture.from_dict(
                    id=id,
                    data=texture_data,
                    samplers=samplers
                ) for id, texture_data in enumerate(textures_data)
            ]

        materials = None
        if materials_data:
            materials = [
                Material.from_dict(
                    id=id,
                    data=material,
                    textures=textures
                ) for id, material in enumerate(materials_data)
            ]

        meshes = None
        if meshes_data:
            meshes = [
                Mesh.from_dict(
                    id=id,
                    data=mesh_data,
                    materials=materials
                ) for id, mesh_data in enumerate(meshes_data)
            ]

        skins = None

        nodes = None
        if nodes_data:
            if skins_data:
                # First Skins pass: Create Skin objects without resolving joints and skeleton
                skins = [
                    Skin.from_dict(
                        id=id,
                        data=skin_data,
                        accessors=accessors
                    ) for id, skin_data in enumerate(skins_data)
                ]

            # First Nodes pass: Create Node objects without resolving children
            nodes = [
                Node.from_dict(
                    id=id,
                    data=node_data,
                    meshes=meshes,
                    cameras=cameras
                ) for id, node_data in enumerate(nodes_data)
            ]
            if nodes:
                # Second Nodes pass: Resolve children
                [node.second_pass(nodes, skins) for node in nodes]

                if skins:
                    # Second Skins pass: Resolve children
                    [skin.second_pass(nodes) for skin in skins]

        scenes = None
        if scenes_data:
            scenes = [
                Scene.from_dict(
                    id=id,
                    data=scene_data,
                    nodes=nodes
                ) for id, scene_data in enumerate(scenes_data)
            ]

        scene = scenes[default_scene_index] if default_scene_index else None

        animations = None
        if animations_data:
            animations = [
                Animation.from_dict(
                    id=id,
                    data=animation_data,
                    accessors=accessors,
                    nodes=nodes
                ) for id, animation_data in enumerate(animations_data)
            ]

        extensions = data.get("extensions")
        extras = data.get("extras")
        return cls(
            extensions_used=extensions_used,
            extensions_required=extensions_required,
            asset=asset,
            cameras=cameras,
            buffers=buffers,
            buffer_views=buffer_views,
            accessors=accessors,
            images=images,
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
    def load_from_file(cls, filename: str) -> '':
        with open(filename, 'r') as f:
            data = json.load(f)
            return cls.from_dict(data)


class TestGLTF(unittest.TestCase):
    def setUp(self):
        asset = {"generator": "TestGLTF", "version": "2.0"}
        buffers = [Buffer(id=0, uri="foo", byte_length=100), Buffer(id=1, uri="bar", byte_length=200),
                   Buffer(id=2, uri="baz", byte_length=300)]
        buffer_views = [BufferView(id=0, buffer=buffers[0], byte_offset=0, byte_length=100),
                        BufferView(id=1, buffer=buffers[1], byte_offset=0, byte_length=200),
                        BufferView(id=2, buffer=buffers[2], byte_offset=0, byte_length=300)]
        accessors = [Accessor(id=0, buffer_view=buffer_views[0], byte_offset=0, component_type=1, count=1, type="VEC3"),
                     Accessor(id=1, buffer_view=buffer_views[1], byte_offset=0, component_type=1, count=1, type="VEC3"),
                     Accessor(id=2, buffer_view=buffer_views[2], byte_offset=0, component_type=1, count=1, type="VEC3")]
        images = [Image(id=0, uri="foo"), Image(id=1, uri="bar"), Image(id=2, uri="baz")]
        samplers = [Sampler(id=0), Sampler(id=1), Sampler(id=2)]
        textures = [Texture(id=0, sampler=samplers[0], source=images[0]), Texture(id=1, sampler=samplers[1], source=images[1]),
                    Texture(id=2, sampler=samplers[2], source=images[2])]
        materials = [Material(id=0, name="foo"), Material(id=1, name="bar"), Material(id=2, name="baz")]
        meshes = [Mesh(id=0, name="foo"), Mesh(id=1, name="bar"), Mesh(id=2, name="baz")]
        skins = [Skin(id=0, name="foo"), Skin(id=1, name="bar"), Skin(id=2, name="baz")]
        cameras = [Camera(id=0, name="foo", type="orthographic", params={"xmag": 1, "ymag": 1, "znear": 1, "zfar": 1}),
                   Camera(id=1, name="bar", type="perspective", params={"xfov": 1, "yfov": 1, "znear": 1, "zfar": 1})]
        nodes = [Node(id=0, name="foo", camera=cameras[0], skin=skins[0]), Node(id=1, name="bar", camera=cameras[1]),
                 Node(id=2, name="baz", skin=skins[1], children=[0, 1]), Node(id=3, name="qux", children=[2]),
                 Node(id=4, name="quux", children=[3], mesh=meshes[0]), Node(id=5, name="corge", children=[4]),
                 Node(id=6, name="grault", children=[5], mesh=meshes[1]), Node(id=7, name="garply", children=[6])]
        scenes = [Scene(id=0, name="foo", nodes=[0, 1]), Scene(id=1, name="bar", nodes=[2, 3]),
                  Scene(id=2, name="baz", nodes=[4, 5]), Scene(id=3, name="qux", nodes=[6, 7])]
        scene = scenes[0]
        self.gltf = GLTF(asset=asset, buffers=buffers, buffer_views=buffer_views, accessors=accessors, images=images,
                          samplers=samplers, textures=textures, materials=materials, meshes=meshes, skins=skins,
                          cameras=cameras, nodes=nodes, scenes=scenes, scene=scene)
        self.compare_dict = {
            "asset": {"generator": "TestGLTF", "version": "2.0"},
            "buffers": [{"uri": "foo", "byteLength": 100}, {"uri": "bar", "byteLength": 200}, {"uri": "baz", "byteLength": 300}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 100}, {"buffer": 1, "byteOffset": 0, "byteLength": 200}, {"buffer": 2, "byteOffset": 0, "byteLength": 300}],
            "accessors": [{"bufferView": 0, "byteOffset": 0, "componentType": 1, "count": 1, "type": "VEC3"},
                           {"bufferView": 1, "byteOffset": 0, "componentType": 1, "count": 1, "type": "VEC3"},
                           {"bufferView": 2, "byteOffset": 0, "componentType": 1, "count": 1, "type": "VEC3"}],
            "images": [{"uri": "foo"}, {"uri": "bar"}, {"uri": "baz"}],
            "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 10497, "wrapT": 10497}],
            "textures": [{"sampler": 0, "source": 0}, {"sampler": 1, "source": 1}, {"sampler": 2, "source": 2}],
            "materials": [{"name": "foo"}, {"name": "bar"}, {"name": "baz"}],
            "meshes": [{"name": "foo"}, {"name": "bar"}, {"name": "baz"}],
            "skins": [{"name": "foo"}, {"name": "bar"}, {"name": "baz"}],
            "cameras": [{"name": "foo", "type": "orthographic", "params": {"xmag": 1, "ymag": 1, "znear": 1, "zfar": 1}},
                        {"name": "bar", "type": "perspective", "params": {"xfov": 1, "yfov": 1, "znear": 1, "zfar": 1}}],
            "nodes": [{"name": "foo", "camera": 0, "skin": 0}, {"name": "bar", "camera": 1}, {"name": "baz", "skin": 1, "children": [0, 1]},
                      {"name": "qux", "children": [2]}, {"name": "quux", "children": [3], "mesh": 0}, {"name": "corge", "children": [4]},
                      {"name": "grault", "children": [5], "mesh": 1}, {"name": "garply", "children": [6]}],
            "scenes": [{"name": "foo", "nodes": [0, 1]}, {"name": "bar", "nodes": [2, 3]}, {"name": "baz", "nodes": [4, 5]},
                       {"name": "qux", "nodes": [6, 7]}],
            "scene": 0
        }

    def test_to_dict(self):
        self.assertEqual(self.gltf.to_dict(), self.compare_dict)

    def test_from_dict(self):
        self.assertEqual(GLTF.from_dict(self.compare_dict), self.gltf)

    def test_save_to_file(self):
        self.gltf.save_to_file("test.gltf")
        loaded_gltf = GLTF.load_from_file("test.gltf")
        self.assertEqual(self.gltf, loaded_gltf)
        os.remove("test.gltf")

    def test_load_valid_file(self):
        self.gltf.load_from_file("AnimatedCube.gltf")
        self.assertIsInstance(self.gltf.asset, Asset)
        [self.assertIsInstance(buffer, Buffer) for buffer in self.gltf.buffers]
        [self.assertIsInstance(buffer_view, BufferView) for buffer_view in self.gltf.buffer_views]
        [self.assertIsInstance(accessor, Accessor) for accessor in self.gltf.accessors]
        [self.assertIsInstance(image, Image) for image in self.gltf.images]
        [self.assertIsInstance(sampler, Sampler) for sampler in self.gltf.samplers]
        [self.assertIsInstance(texture, Texture) for texture in self.gltf.textures]
        [self.assertIsInstance(material, Material) for material in self.gltf.materials]
        [self.assertIsInstance(mesh, Mesh) for mesh in self.gltf.meshes]
        [self.assertIsInstance(skin, Skin) for skin in self.gltf.skins]
        [self.assertIsInstance(camera, Camera) for camera in self.gltf.cameras]
        [self.assertIsInstance(node, Node) for node in self.gltf.nodes]
        [self.assertIsInstance(scene, Scene) for scene in self.gltf.scenes]
        self.assertIsInstance(self.gltf.scene, Scene)

    def test_load_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            self.gltf.load_from_file("invalid_file.gltf")
