import ctypes

from POGLE.Core.Core import *


def NMM(t: glm.vec3, r: glm.vec3 = glm.vec3(), s: glm.vec3 = glm.vec3(1)) -> glm.mat4:
    return NewModelMatrix(t, r, s)


def NewModelMatrix(translation: glm.vec3 = glm.vec3(),
                   rotation: glm.vec3 = glm.vec3(),
                   scale: glm.vec3 = glm.vec3(1.0)) -> glm.mat4:
    # Create the rotation matrix using Euler angles
    rotation_matrix = glm.mat4_cast(glm.quat(glm.vec3(glm.radians(rotation))))

    # Create the scale and translation matrices
    scale_matrix = glm.scale(glm.mat4(), scale)
    translation_matrix = glm.translate(glm.mat4(), translation)

    # Combine the matrices: translation * (rotation * scale)
    model_matrix = translation_matrix * rotation_matrix * scale_matrix

    return model_matrix

class Vector:
    __create_key = object()

    def __init__(self, create_key, vector_type, initial_capacity=10):
        if create_key != Vector.__create_key:
            raise ValueError("Vector objects must be created using Vector._new")
        """
        Initialize the Vector with a given structure class and initial capacity.

        :param vector_type: The ctypes.Structure class that the vector will manage.
        :param initial_capacity: Initial capacity of the vector.
        """
        self._vector_type = vector_type
        self._initialize(initial_capacity)

    @classmethod
    def _new(cls, vector_type, initial_capacity=10):
        return cls(cls.__create_key, vector_type, initial_capacity)

    @classmethod
    def Struct(cls, initial_capacity=10):
        return cls._new(ctypes.Structure, initial_capacity)

    @classmethod
    def from_iterable(cls, vector_type, iterable):
        """
        Create a Vector from an iterable of elements.

        :param vector_type: The ctypes.Structure class that the vector will manage.
        :param iterable: An iterable of elements to initialize the vector with.
        """
        instance = cls._new(vector_type, len(iterable))
        for element in iterable:
            instance.push(element)
        return instance

    def _initialize(self, capacity):
        """
        Initialize or reinitialize the vector with a given capacity.
        """
        self.capacity = capacity
        self.size = 0
        # Allocate memory for the array of vector_type instances
        self.data = (self._vector_type * capacity)()

    def push(self, vector_element):
        """
        Add an element to the end of the vector, resizing if necessary.
        """
        if not isinstance(vector_element, self._vector_type):
            raise TypeError("Vector Element Is Of Wrong Type")
        if self.size >= self.capacity:
            self._resize(self.capacity * 2)
        self.data[self.size] = vector_element
        self.size += 1

    def pop(self):
        """
        Remove and return the last structure instance in the vector.
        """
        if self.size == 0:
            raise IndexError("Pop from empty vector")
        self.size -= 1
        return self.data[self.size]

    def _resize(self, new_capacity):
        """
        Resize the vector to a new capacity.
        """
        new_data = (self._vector_type * new_capacity)()
        for i in range(self.size):
            new_data[i] = self.data[i]
        self.data = new_data
        self.capacity = new_capacity

    def __getitem__(self, index):
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")
        return self.data[index]

    def __len__(self):
        return self.size

    @property
    def bytes(self) -> int:
        return self.size * c_sizeof(self._vector_type)

    def data_pointer(self):
        """
        Get a ctypes pointer to the underlying data.
        """
        return ctypes.cast(self.data, ctypes.POINTER(self._vector_type))

    def __iter__(self):
        """
        Return an iterator for the vector.
        """
        self._iter_index = 0
        return self

    def __next__(self):
        """
        Return the next element in the iteration.
        """
        if self._iter_index >= self.size:
            raise StopIteration
        result = self.data[self._iter_index]
        self._iter_index += 1
        return result

    def reserve(self, new_capacity):
        """
        Reserve space for at least new_capacity elements.
        """
        if new_capacity > self.capacity:
            self._resize(new_capacity)

    def clear(self):
        """
        Clear all elements in the vector.
        """
        self.size = 0

class Data:
    from ctypes import c_bool, c_ubyte, c_byte, c_ushort, c_short, c_uint32, c_int32, c_float, c_double

    _typeDict = {
        GL_BOOL: c_bool,
        GL_UNSIGNED_BYTE: c_ubyte,
        GL_BYTE: c_byte,
        GL_UNSIGNED_SHORT: c_ushort,
        GL_SHORT: c_short,
        GL_UNSIGNED_INT: c_uint32,
        GL_INT: c_int32,
        GL_FLOAT: c_float,
        GL_DOUBLE: c_double
    }
    class Attribute:
        class _MetaClass(type):
            T = TypeVar("T")
            def __getitem__(cls, key: Type[T]) -> Type['NewClass']:
                # Define a new class with the 'gl_type' attribute set to the key
                class NewClass(cls, Generic[cls.T]):
                    gl_type = key

                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)

                return NewClass

        class _Base(metaclass=_MetaClass):
            __create_key = object()
            @classmethod
            def _new(cls, normalized: GLboolean, subAttribs: int, divisor: int = 0, attrName: Optional[str] = None):
                """
                Create a new _DataAttribute instance.
                """
                c_dtype = Data._typeDict[cls.gl_type]
                return cls(cls.__create_key, c_sizeof(c_dtype) * subAttribs, c_dtype, normalized,
                                subAttribs,
                                divisor, attrName)

            def __init__(self, create_key, numBytes: GLint, ctype_base: GLenum, normalized: GLboolean, numElements: GLsizei, divisor: int,
                         attrName: Optional[str]):
                if create_key != Data.Attribute._Base.__create_key:
                    raise ValueError("_DataAttribute objects must be created using _DataAttribute._new")
                self.numBytes: GLint = numBytes
                self.ctype_base = ctype_base
                self.normalized: GLboolean = normalized
                self.numElements: GLsizei = numElements
                self.ctype = self.numElements * self.ctype_base
                self.divisor = divisor
                self.attrName = attrName

            @classmethod
            def Single(cls, divisor: int = 0, normalized: GLboolean = GL_FALSE, attrName: Optional[str] = None):
                return cls._new(normalized, 1, divisor, attrName)

            @classmethod
            def Array(cls, normalized: GLboolean, arraySize: int, divisor: int = 0, attrName: Optional[str] = None):
                if arraySize > 4:
                    divisor = divisor or 1
                return cls._new(normalized, arraySize, divisor, attrName)

            @classmethod
            def Vec2(cls, divisor: int = 0, normalized: GLboolean = GL_FALSE, attrName: Optional[str] = None):
                return cls.Array(normalized, 2, divisor, attrName)

            @classmethod
            def Vec3(cls, divisor: int = 0, normalized: GLboolean = GL_FALSE, attrName: Optional[str] = None):
                return cls.Array(normalized, 3, divisor, attrName)

            @classmethod
            def Vec4(cls, divisor: int = 0, normalized: GLboolean = GL_FALSE, attrName: Optional[str] = None):
                return cls.Array(normalized, 4, divisor, attrName)

            @classmethod
            def _Mat(cls, divisor: int = 0, matRows: int = 2, matCols: int = 2, normalized: GLboolean = GL_FALSE,
                     attrName: Optional[str] = None):
                total_elements = matRows * matCols
                return cls.Array(normalized, total_elements, divisor, attrName=attrName)

            @classmethod
            def Mat4(cls, divisor: int = 1, normalized: GLboolean = GL_FALSE, attrName: Optional[str] = None):
                return cls._Mat(divisor, 4, 4, normalized, attrName)

        class _AttribChild(_Base):
            @classmethod
            def Bool(cls):
                return cls[GL_BOOL]
            @classmethod
            def UByte(cls):
                return cls[GL_UNSIGNED_BYTE]
            @classmethod
            def Byte(cls):
                return cls[GL_BYTE]
            @classmethod
            def UShort(cls):
                return cls[GL_UNSIGNED_SHORT]
            @classmethod
            def Short(cls):
                return cls[GL_SHORT]
            @classmethod
            def UInt(cls):
                return cls[GL_UNSIGNED_INT]
            @classmethod
            def Int(cls):
                return cls[GL_INT]
            @classmethod
            def Float(cls):
                return cls[GL_FLOAT]
            @classmethod
            def Double(cls):
                return cls[GL_DOUBLE]

        class Vertex(_AttribChild):
            def set_attribute_array_pointer(self, id: GLuint, stride: GLsizei, offset: GLsizei):
                """
                Set the vertex attribute pointer for this attribute.
                """
                num_components = 4 if self.numElements > 4 else self.numElements
                num_attributes = (self.numElements + num_components - 1) // num_components

                for c in range(num_attributes):
                    glEnableVertexAttribArray(id + c)

                    component_size = min(num_components, self.numElements - c * num_components)
                    component_bytes = (self.numBytes // self.numElements) * component_size
                    component_offset = offset + component_bytes * c

                    if self.gl_type == GL_INT:
                        glVertexAttribIPointer(id + c, component_size, self.gl_type, stride,
                                               ctypes.c_void_p(component_offset))
                    else:
                        glVertexAttribPointer(id + c, component_size, self.gl_type, self.normalized, stride,
                                              ctypes.c_void_p(component_offset))

                    glVertexAttribDivisor(id + c, self.divisor)
                    print(
                        f"\nPointer Set:\t{{id: {id + c} | elements: {component_size} | bytes: {component_bytes} | dtype: {self.gl_type} | normalized: {self.normalized} | stride: {stride} | offset: {component_offset} | divisor: {self.divisor}}}")

        class UniformBlock(_AttribChild):
            def set_uniform_block_binding(self, program: GLuint, blockIndex: GLuint, bindingPoint: GLuint):
                """
                Set the uniform block binding for this uniform attribute.
                """
                glUniformBlockBinding(program, blockIndex, bindingPoint)
                print(
                    f"Uniform block binding set: {{program: {program} | blockIndex: {blockIndex} | bindingPoint: {bindingPoint}}}")

    def _struct(attributes: Union[List[Attribute.Vertex], List[Attribute.UniformBlock]]) -> ctypes.Structure:
        fields = []
        for i, attribute in enumerate(attributes):
            attributeName = attribute.attrName if attribute.attrName is not None else f"attribute_{i}"
            fields.append((attributeName, attribute.ctype))

        class Struct(ctypes.Structure):
            _fields_ = fields

            def __init__(self, data: list):
                super().__init__()
                num_fields = len(self)
                if num_fields != len(data):
                    raise ValueError(
                        "data must have same amount of attributes as the number of fields in the struct")
                for fieldID in range(num_fields):
                    self.set_attribute(fieldID, data[fieldID])

            def set_attribute(self, fieldID: int, fieldData):
                field = self[fieldID]
                field_length = len(field)
                field_data_length = 1
                try:
                    field_data_length = len(fieldData)
                except:
                    pass
                if field_length != field_data_length:
                    raise ValueError("fieldData must have the same number of elements as the struct field")
                elif 1 == field_length:
                    field = fieldData
                else:
                    for elementID in range(field_length):
                        field[elementID] = fieldData[elementID]

            def __getitem__(self, key):
                if isinstance(key, str):
                    for name, _ in self._fields_:
                        if name == key:
                            return getattr(self, name)
                    raise KeyError(f"Field '{key}' not found in DataStruct.")
                elif isinstance(key, int):
                    if 0 <= key < len(self._fields_):
                        name, _ = self._fields_[key]
                        return getattr(self, name)
                    else:
                        raise IndexError("Index out of range.")
                else:
                    raise TypeError("Key must be an integer or string.")

            def __iter__(self):
                for name, _ in self._fields_:
                    yield getattr(self, name)

            def __len__(self):
                return len(self._fields_)

        return Struct

    class _Layout:
        def __init__(self, attributes: List, startID: int = 0):
            super().__init__()
            if not isinstance(attributes, List):
                if not isinstance(attributes, Data.Attribute._Base):
                    raise TypeError("attributes should be a list of _DataAttribute instances.")
                attributes = [attributes]

            if not all(isinstance(attribute, Data.Attribute._Base) for attribute in attributes):
                raise TypeError("All elements in attributes must be instances of _DataAttribute.")

            self.attributes: Union[List[Data.Attribute.Vertex], List[Data.Attribute.UniformBlock]] = attributes
            self.struct = Data._struct(attributes)
            self.numElements = sum(attribute.numElements for attribute in self.attributes)

            self.stride = c_sizeof(self.struct)
            self.nextID = startID

        def __getitem__(self, key):
            return self.struct[key]

        def __iter__(self):
            return iter(self.struct)

    class VertexLayout(_Layout):
        def set_attribute_array_pointers(self, extraOffset: int = 0):
            """
            Set the attribute pointers for the data layout.

            Args:
                extraOffset (int): Additional offset to apply to each attribute's offset.
            """
            offset = 0
            for attribute in self.attributes:
                attribute.set_attribute_array_pointer(self.nextID, self.stride, offset + extraOffset)
                offset += attribute.numBytes
                if attribute.numElements < 5:
                    self.nextID += 1
                else:
                    self.nextID += attribute.numElements // 4

    class UniformBlockLayout(_Layout):
        def __init__(self, blockName: str, attributes: List, startID: int = 0):
            super().__init__(attributes, startID)
            self.blockName = blockName

    class Set:
        def __init__(self, layout, data: Union[Vector, ctypes.Structure]):
            self.layout: Data._Layout = layout
            if isinstance(data, ctypes.Structure):
                data = Vector.from_iterable(ctypes.Structure, [data])
            self.data: Vector = data

VA = Data.Attribute.Vertex
UBA = Data.Attribute.UniformBlock
_AttributeBase = Data.Attribute._Base

DVL = Data.VertexLayout
DUBL = Data.UniformBlockLayout

DataSet = Data.Set

aLocalPosXYZ = VA.Float().Vec3(attrName="aLocalPosXYZ")
aColorRGB = VA.Float().Vec3(attrName="aColorRGB")
aInstColorRGB = VA.Float().Vec3(1, attrName="aColorRGB")
aAlpha = VA.Float().Single(attrName="aAlpha")
aInstAlpha = VA.Float().Single(1, attrName="aAlpha")
aModel = VA.Float().Mat4(attrName="aModel")

aLocalPosXY = VA.Float().Vec2(attrName="aLocalPosXY")
aTexUV = VA.Float().Vec2(attrName="aTexUV")
aTexPos = VA.Float().Vec2(1, attrName="aTexPos")
aTexSize = VA.Float().Vec2(1, attrName="aTexSize")

aSideID = VA.UShort().Single(1, attrName="aSideID")

aWorldPos = VA.Float().Vec3(6, attrName="aWorldPos")


# Define a default vertex layout for standard vertex attributes
defaultLayout = DVL([
    aLocalPosXYZ,  # Vertex position (x, y, z)
    aColorRGB,  # Vertex color (r, g, b)
    aAlpha  # Vertex alpha (a)
])

# Define a default instance layout for per-instance data
defaultInstanceLayout = DVL([
    aModel  # Model matrix (4x4 matrix for transformations)
])