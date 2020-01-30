from glumpy import app, gl, glm, gloo
import numpy as np

vertex = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

attribute vec3 a_position;
attribute vec4 a_color;
attribute vec2 a_texcoord;
attribute vec3 a_normal;

varying float depth;
varying vec3 color;
varying vec3 v_pos;
varying vec3 v_normal;
varying vec2 v_texcoord;

void main()
{
    gl_Position = u_projection * u_view * u_model * vec4(a_position, 1.0);
    // gl_Position = vec4(a_position, 1.0);
    // depth = gl_Position.z / gl_Position.w;
    color = a_color.rgb;
    v_texcoord = a_texcoord;
    v_pos = a_position;
    v_normal = a_normal;
}
"""

fragment = """
varying float depth;
varying vec3 color;
varying vec2 v_texcoord;
varying vec3 v_pos;
varying vec3 v_normal;

uniform mat4      u_model;           // Model matrix
uniform mat4      u_view;            // View matrix
uniform mat4      u_model_it;        // Normal matrix
uniform mat4      u_projection;      // Projection matrix

uniform vec3 u_linecolor;
uniform sampler2D u_texture;
uniform float u_outline;
uniform vec3 u_lightpos;
uniform vec3 u_lightint;

uniform vec3 u_camerapos;
uniform vec3 u_specular;
uniform float u_shininess;

void main()
{
    vec3 normal = normalize((u_model_it * vec4(v_normal, 0.0)).xyz);
    vec3 position = vec3(u_model * vec4(v_pos, 1.0));
    vec3 lightdir = normalize(u_lightpos - position);

    float lambertian = clamp(dot(normal, lightdir), 0.0, 1.0);
    float specular = 0.0;

    if (lambertian > 0.0)
    {
        vec3 viewdir = normalize(u_camerapos - position);
        vec3 H = normalize(lightdir + viewdir);
        specular = pow(max(dot(H, normal), 0.0), u_shininess);
    }

    vec3 f_diffuse = texture2D(u_texture, v_texcoord).rrr * lambertian;
    vec3 f_specular = u_specular * specular * u_lightint;

    gl_FragColor = vec4(mix((f_diffuse + f_specular), u_linecolor, u_outline), 1.0);
}
"""

cube = gloo.Program(vertex, fragment)
window = app.Window(width=1024, height=768, color=(0.3, 0.3, 0.35, 1.0))

# V = np.zeros(8, [('a_position', np.float32, 3), ("a_color",    np.float32, 4)])
# V['a_position'] = [[ 1, 1, 1], [-1, 1, 1], [-1,-1, 1], [ 1,-1, 1],
#                    [1, -1,-1], [ 1, 1,-1], [-1, 1,-1], [-1,-1,-1]]

# V["a_color"]    = [[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
#                    [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]]

# I = np.array([0,1,2, 0,2,3, 0,3,4, 0,4,5, 0,5,6, 0,6,1,
#               1,6,7, 1,7,2, 7,4,3, 7,3,2, 4,7,6, 4,6,5], dtype=np.uint32)

# V = V.view(gloo.VertexBuffer)
# I = I.view(gloo.IndexBuffer)

# O = np.array([0,1, 1,2, 2,3, 3,0,
#               4, 7, 7, 6, 6, 5, 5, 4,
#               0, 5, 1, 6, 2, 7, 3, 4], dtype=np.uint32)
# O = O.view(gloo.IndexBuffer)

def gen_cube():
    vtype = [('a_position', np.float32, 3),
             ('a_texcoord', np.float32, 2),
             ('a_normal',   np.float32, 3),
             ('a_color',    np.float32, 4)]
    itype = np.uint32

    # Vertices positions
    p = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                  [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=float)

    # Face Normals
    n = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0],
                  [-1, 0, 1], [0, -1, 0], [0, 0, -1]])

    # Vertice colors
    c = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
                  [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]])

    # Texture coords
    t = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

    faces_p = [0, 1, 2, 3,  0, 3, 4, 5,   0, 5, 6, 1,
               1, 6, 7, 2,  7, 4, 3, 2,   4, 7, 6, 5]
    faces_c = [0, 1, 2, 3,  0, 3, 4, 5,   0, 5, 6, 1,
               1, 6, 7, 2,  7, 4, 3, 2,   4, 7, 6, 5]
    faces_n = [0, 0, 0, 0,  1, 1, 1, 1,   2, 2, 2, 2,
               3, 3, 3, 3,  4, 4, 4, 4,   5, 5, 5, 5]
    faces_t = [0, 1, 2, 3,  0, 1, 2, 3,   0, 1, 2, 3,
               3, 2, 1, 0,  0, 1, 2, 3,   0, 1, 2, 3]

    vertices = np.zeros(24, vtype)
    vertices['a_position'] = p[faces_p]
    vertices['a_normal']   = n[faces_n]
    vertices['a_color']    = c[faces_c]
    vertices['a_texcoord'] = t[faces_t]

    filled = np.resize(
       np.array([0, 1, 2, 0, 2, 3], dtype=itype), 6 * (2 * 3))
    filled += np.repeat(4 * np.arange(6, dtype=itype), 6)

    outline = np.resize(
        np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=itype), 6 * (2 * 4))
    outline += np.repeat(4 * np.arange(6, dtype=itype), 8)

    vertices = vertices.view(gloo.VertexBuffer)
    filled   = filled.view(gloo.IndexBuffer)
    outline  = outline.view(gloo.IndexBuffer)

    return vertices, filled, outline

def gen_cube_light():
    vtype = [('a_position', np.float32, 3), ('a_texcoord', np.float32, 2),
             ('a_normal',   np.float32, 3), ('a_color',    np.float32, 4)]
    itype = np.uint32

    # Vertices positions
    p = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                  [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=float)
    # Face Normals
    n = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0],
                  [-1, 0, 1], [0, -1, 0], [0, 0, -1]])
    # Vertice colors
    c = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 0, 1],
                  [1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1]])
    # Texture coords
    t = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

    faces_p = [0, 1, 2, 3,  0, 3, 4, 5,   0, 5, 6, 1,
               1, 6, 7, 2,  7, 4, 3, 2,   4, 7, 6, 5]
    faces_c = [0, 1, 2, 3,  0, 3, 4, 5,   0, 5, 6, 1,
               1, 6, 7, 2,  7, 4, 3, 2,   4, 7, 6, 5]
    faces_n = [0, 0, 0, 0,  1, 1, 1, 1,   2, 2, 2, 2,
               3, 3, 3, 3,  4, 4, 4, 4,   5, 5, 5, 5]
    faces_t = [0, 1, 2, 3,  0, 1, 2, 3,   0, 1, 2, 3,
               3, 2, 1, 0,  0, 1, 2, 3,   0, 1, 2, 3]

    vertices = np.zeros(24, vtype)
    vertices['a_position'] = p[faces_p]
    vertices['a_normal']   = n[faces_n]
    vertices['a_color']    = c[faces_c]
    vertices['a_texcoord'] = t[faces_t]

    filled = np.resize(
       np.array([0, 1, 2, 0, 2, 3], dtype=itype), 6 * (2 * 3))
    filled += np.repeat(4 * np.arange(6, dtype=itype), 6)

    outline = np.resize(
        np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=itype), 6 * (2 * 4))
    outline += np.repeat(4 * np.arange(6, dtype=itype), 8)

    vertices = vertices.view(gloo.VertexBuffer)
    filled   = filled.view(gloo.IndexBuffer)
    outline  = outline.view(gloo.IndexBuffer)

    return vertices, filled, outline

camerapos = (0, 0, 5)

V,I,O = gen_cube_light()
cube.bind(V)
cube['u_lightpos'] = camerapos
cube['u_lightint'] = 1, 1, 1
cube['u_camerapos'] = camerapos
cube['u_specular'] = 1, 1, 0
cube['u_shininess'] = 128

cube['u_linecolor'] = 1, 0, 0

def checkerboard(grid_num=8, grid_size=32):
    """ Checkerboard pattern """
    
    row_even = grid_num // 2 * [0, 1]
    row_odd = grid_num // 2 * [1, 0]
    Z = np.row_stack(grid_num // 2 * (row_even, row_odd)).astype(np.uint8)
    return 255 * Z.repeat(grid_size, axis=0).repeat(grid_size, axis=1)

model = np.eye(4, dtype=np.float32)
projection = np.eye(4, dtype=np.float32)

view = np.eye(4, dtype=np.float32)
glm.translate(view, camerapos[0], camerapos[1], camerapos[2])
view = np.matrix(view).I

cube['u_model'] = model
cube['u_model_it'] = np.matrix(model).I.T
cube['u_view'] = view
cube['u_projection'] = projection
cube['u_texture'] = checkerboard()

phi, theta = 0, 0

@window.event
def on_resize(width, height):
    ratio = width / float(height)
    cube['u_projection'] = glm.perspective(45, ratio, 2, 100)

frame_index = 0

@window.event
def on_draw(dt):
    global phi, theta, frame_index
    window.clear()

    cube['u_outline'] = 0.0;
    cube.draw(gl.GL_TRIANGLES, I)
    cube['u_outline'] = 1.0;
    cube.draw(gl.GL_LINES, O)

    theta += 0.5
    phi += 0.5

    model = np.eye(4, dtype=np.float32)
    glm.rotate(model, theta, 0, 0, 1)
    glm.rotate(model, phi, 0, 1, 0)

    cube['u_model'] = model
    cube['u_model_it'] = np.matrix(model).I.T

    frame_index += 1
    print('current frame: ' + str(frame_index))
    if frame_index == 100:
        print('changing checkboard dimension to 16')
        cube['u_texture'] = checkerboard(16).view(gloo.Texture2D)
    elif frame_index == 200:
        print('changing back checkboard dimension to 8')
        cube['u_texture'] = checkerboard().view(gloo.Texture2D)


@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glLineWidth(4.0)
    gl.glDisable(gl.GL_CULL_FACE)
    # gl.glCullFace(gl.GL_FRONT_AND_BACK)

app.run()
