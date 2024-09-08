import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from _gltf import GLTF  # Assuming GLTF is your data class and properly imported


# Shader Creation and Compilation
def create_and_compile_shaders() -> int:
    print("Creating and compiling shaders")

    vertex_shader_code = """
    #version 330 core
    layout(location = 0) in vec3 aPosition;
    layout(location = 1) in vec3 aNormal;
    layout(location = 2) in vec2 aTexCoord;
    layout(location = 3) in vec4 aTangent;  // Added for tangents

    out vec3 FragPosition;
    out vec3 Normal;
    out vec2 TexCoord;
    out vec3 Tangent;  // Output tangent to fragment shader

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        FragPosition = vec3(model * vec4(aPosition, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;
        TexCoord = aTexCoord;
        Tangent = mat3(transpose(inverse(model))) * aTangent.xyz;  // Transform tangent
        gl_Position = projection * view * vec4(FragPosition, 1.0);
    }
    """

    fragment_shader_code = """
    #version 330 core
    out vec4 FragColor;

    in vec3 FragPosition;
    in vec3 Normal;
    in vec2 TexCoord;
    in vec3 Tangent;  // Receive tangent from vertex shader

    uniform sampler2D baseColorTexture;
    uniform vec3 baseColorFactor;

    void main() {
        vec4 textureColor = texture(baseColorTexture, TexCoord);
        // Simple fragment shader, you can expand to use Tangent for advanced effects
        FragColor = vec4(baseColorFactor, 1.0) * textureColor;
    }
    """

    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_code)
    glCompileShader(vertex_shader)
    if not glGetShaderiv(vertex_shader, GL_COMPILE_STATUS):
        print(f"Vertex shader compilation failed: {glGetShaderInfoLog(vertex_shader).decode()}")
        raise RuntimeError("Vertex shader compilation failed")

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_code)
    glCompileShader(fragment_shader)
    if not glGetShaderiv(fragment_shader, GL_COMPILE_STATUS):
        print(f"Fragment shader compilation failed: {glGetShaderInfoLog(fragment_shader).decode()}")
        raise RuntimeError("Fragment shader compilation failed")

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)
    if not glGetProgramiv(shader_program, GL_LINK_STATUS):
        print(f"Shader program linking failed: {glGetProgramInfoLog(shader_program).decode()}")
        raise RuntimeError("Shader program linking failed")

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program


# Buffer and Attribute Setup
def get_data_from_buffer(buffer, buffer_view, accessor):
    buffer_data = buffer.data[buffer_view.byte_offset:buffer_view.byte_offset + buffer_view.byte_length]
    data = np.frombuffer(buffer_data, dtype=np.float32).reshape(accessor.count, accessor.type_size)
    return data


def setup_buffers_and_attributes(gltf_data: GLTF):
    vao_map = {}  # Dictionary to map mesh indices to their VAOs

    attribute_mapping = {
        "POSITION": (3, 0),
        "NORMAL": (3, 1),
        "TEXCOORD_0": (2, 2),
        "TANGENT": (4, 3)
    }

    for i, mesh in enumerate(gltf_data.meshes):
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)
        print(f"Created VAO {vao} for mesh {i}")

        for primitive in mesh.primitives:
            for attr_name, accessor_index in primitive.attributes.attributes.items():
                accessor = gltf_data.accessors[accessor_index]
                buffer_view = gltf_data.buffer_views[accessor.buffer_view]
                buffer = gltf_data.buffers[buffer_view.buffer]
                attribute_data = get_data_from_buffer(buffer, buffer_view, accessor)

                if attr_name in attribute_mapping:
                    size, location = attribute_mapping[attr_name]

                    buffer_id = glGenBuffers(1)
                    glBindBuffer(GL_ARRAY_BUFFER, buffer_id)
                    glBufferData(GL_ARRAY_BUFFER, attribute_data, GL_STATIC_DRAW)
                    glVertexAttribPointer(location, size, GL_FLOAT, GL_FALSE, 0, None)
                    glEnableVertexAttribArray(location)
                    print(f"Set up buffer for attribute {attr_name} at location {location}")
                else:
                    print(f"Warning: Attribute '{attr_name}' not recognized or not supported.")

            if primitive.indices is not None:
                indices_accessor = gltf_data.accessors[primitive.indices]
                buffer_view = gltf_data.buffer_views[indices_accessor.buffer_view]
                buffer = gltf_data.buffers[buffer_view.buffer]
                index_data = get_data_from_buffer(buffer, buffer_view, indices_accessor)
                index_buffer = glGenBuffers(1)
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer)
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data, GL_STATIC_DRAW)
                print("Set up index buffer")

            # Check for OpenGL errors
            error = glGetError()
            if error != GL_NO_ERROR:
                print(f"OpenGL error: {error}")

        glBindVertexArray(0)
        vao_map[i] = vao  # Map the mesh index to its VAO
        print(f"VAO {vao} set up for mesh {i}")

    return vao_map


# Material Setup
def setup_material(material: GLTF.Material, shader_program: int, gltf_data: GLTF):
    print(f"Setting up material: {material.name}")
    if material.pbr_metallic_roughness:
        pbr = material.pbr_metallic_roughness
        base_color_factor_location = glGetUniformLocation(shader_program, "baseColorFactor")
        if base_color_factor_location != -1:
            glUniform3fv(base_color_factor_location, 1, pbr.base_color_factor)
        else:
            print("Uniform baseColorFactor not found.")

        if pbr.base_color_texture:
            texture_index = pbr.base_color_texture['index']
            print(f"Binding texture {texture_index}")
            texture = gltf_data.textures[texture_index]
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, texture.id)
            texture_location = glGetUniformLocation(shader_program, "baseColorTexture")
            if texture_location != -1:
                glUniform1i(texture_location, 0)
            else:
                print("Uniform baseColorTexture not found.")


# Rendering Loop
def render_model(gltf_data: GLTF, vao_map: dict, shader_program: int):
    print("Rendering model")
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader_program)

    for i, mesh in enumerate(gltf_data.meshes):
        vao = vao_map.get(i)  # Retrieve the VAO for the mesh index
        if vao is None:
            print(f"No VAO found for mesh {i}, skipping")
            continue

        glBindVertexArray(vao)

        # Set up material for each mesh
        material = gltf_data.materials[i] if i < len(gltf_data.materials) else None
        if material:
            setup_material(material, shader_program, gltf_data)

        for primitive in mesh.primitives:
            if primitive.indices:
                num_indices = gltf_data.accessors[primitive.indices].count
                print(f"Drawing elements for mesh {i}, num_indices: {num_indices}")
                glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
            else:
                num_vertices = gltf_data.accessors[primitive.attributes.position].count
                print(f"Drawing arrays for mesh {i}, num_vertices: {num_vertices}")
                glDrawArrays(GL_TRIANGLES, 0, num_vertices)

        glBindVertexArray(0)

    # Check for OpenGL errors after rendering
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL error after rendering: {error}")


# Main Function to Tie Everything Together
def main():
    gltf_data = GLTF.load_from_file("AnimatedCube.gltf")  # Implement this function as needed
    shader_program = create_and_compile_shaders()
    vao_map = setup_buffers_and_attributes(gltf_data)

    # Render the model
    render_model(gltf_data, vao_map, shader_program)

    # Clean up
    glDeleteProgram(shader_program)
    print("Done.")


if __name__ == "__main__":
    main()
