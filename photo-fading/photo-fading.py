from glumpy import app, gl, gloo, data, glm
import numpy as np
# from glumpy.transforms import PanZoom, Translate
import cv2
import os

wnd_w = 800
wnd_h = 800

window = app.Window(wnd_w, wnd_h)

vertex_shader = """
    attribute vec2 position;
    attribute vec2 texcoord;

    varying vec2 v_texcoord;

    uniform mat4 u_model;
    uniform mat4 u_view;
    uniform mat4 u_projection;

    void main()
    {
        gl_Position = u_projection * u_view * u_model * vec4(position, 0, 1);
        v_texcoord = texcoord;
    }
"""

fragment_shader = """
    varying vec2 v_texcoord;

    uniform sampler2D tex;
    uniform float alpha;

    uniform vec3 u_linecolor;
    uniform float u_lineflag;

    void main()
    {
        vec3 texel = texture2D(tex, v_texcoord).rgb;
        gl_FragColor = vec4(mix(texel, u_linecolor, u_lineflag), alpha);
    }
"""

line_fragment_shader = """
    varying vec2 v_texcoord;
    uniform vec4 u_color;
    void main()
    {
        gl_FragColor = u_color;
    }
"""

V = np.zeros(4, [('position', np.float32, 2), ("texcoord", np.float32, 2)])
V['position'] = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
V['texcoord'] = [(0, 1), (0, 0), (1, 1), (1, 0)]

V = V.view(gloo.VertexBuffer)

program = gloo.Program(vertex_shader, fragment_shader)
program.bind(V)

program['u_model'] = np.eye(4, dtype=np.float32)

view = np.eye(4, dtype=np.float32)
view = glm.translate(view, 0, 0, -3)
program['u_view'] = view
program['u_projection'] = np.eye(4, dtype=np.float32)

line_program = gloo.Program(vertex_shader, line_fragment_shader)
program.bind(V)

# cv face detection.
face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')

tar_pos = [0, 0]

tar_scale = 2
t = 0
duration = 5
start_tm = 0
last_rect = None
curr_rect = None

last_img = None
curr_img = None
last_asp = []
curr_asp = []

# get all files in folder.
img_folder = './images'
images = os.listdir(img_folder)
imge_list = []
for f in images:
    imge_list.append(os.path.join(img_folder, f))

# map pos to [-1, 1] space, first map longer edge to [-1, 1]
def normalized_point(pt, img_size):
    half_long_edge = max(img_size[0], img_size[1]) * 0.5
    norm_pt = (pt - np.asarray([img_size[1], img_size[0]], dtype=np.float32) * 0.5) / half_long_edge
    return norm_pt

def normalized_image_vertices(img_size):
    asp = img_size[1] / float(img_size[0])
    half_w = 1.0 if asp >= 1 else asp
    half_h = 1.0 if asp < 1 else 1.0 / asp

    return np.asarray([[-half_w, -half_h], [-half_w, half_h], [half_w, -half_h], [half_w, half_h]])

def map_coord(pos, img_size):
    ndc = pos[:]
    hw = img_size[0] / 2
    hh = img_size[1] / 2
    ndc[0] = (pos[0] - hw) / hw
    ndc[1] = (pos[1] - hh) / hh

    asp = img_size[0] / float(img_size[1])
    if asp > 1:
        ndc[0] *= asp
    else:
        ndc[1] /= asp

    return ndc

def detect_face(image):
    global face_cascade
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5)
    )

    if faces is not None and len(faces) > 0:
        # return list(map(lambda x: normalized_point(x, image.shape), faces[0]))
        rect = faces[0]
        print('detected face rect: {}'.format(rect))
        rect_vt = [[rect[0], rect[1]],
                   [rect[0], rect[1] + rect[3]],
                   [rect[0] + rect[2], rect[1] + rect[3]],
                   [rect[0] + rect[2], rect[1]]]
        return np.asarray(list(map(lambda x: normalized_point(x, image.shape), rect_vt)))
    else:
        return None

def load_image2(image_file_path):
    image = data.load(image_file_path)
    if image is None:
        print('load image failed, path: ' + image_file_path)
        return None

    print('image: {}, shape: {}'.format(image_file_path, image.shape))

    img_info = {}
    img_info['image'] = image
    img_info['image_rect'] = normalized_image_vertices(image.shape)    
    img_info['face_rect'] = detect_face(image)

    return img_info

image_info = []
image_info.append(load_image2(imge_list[0]))
image_info.append(load_image2(imge_list[1]))

def load_image(image_file_path):
    global last_rect, curr_rect, program, face_cascade, tar_pos, tar_scale, t, last_img, curr_img, last_asp, curr_asp

    last_img = curr_img

    curr_img = data.load(image_file_path)
    asp = curr_img.shape[1] / curr_img.shape[0]
 
    last_asp = curr_asp
    print('image asp: {}'.format(asp))
    if asp > 1:
        curr_asp = (asp, 1)
    else:
        curr_asp = (1, 1 / asp)

    gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5,5)
    )

    last_rect = curr_rect
    if len(faces) > 0:
        curr_rect = faces[0]

        if last_rect is not None:
            cx = curr_rect[0] + curr_rect[2] / 2
            cy = curr_rect[1] + curr_rect[3] / 2

            tar_pos = map_coord([cx, cy], curr_img.shape[1::-1])
            tar_scale = curr_rect[2] / last_rect[2]

    # rest t.
    t = 0

current_img_idx = 0

# load first image.
if len(imge_list) > 0:
    load_image(imge_list[current_img_idx])
    current_img_idx += 1

    program['tex'] = curr_img

@window.event
def on_resize(width, height):
    # half_w, half_h = width * 0.5, height * 0.5
    hw = width / float(height)
    hh = 1.0

    if hw > 1.0:
        hh = 1.0 / hw
        hw = 1.0

    print('frustum boundary, horizontal: {}, vertical boundary: {}'.format(hw, hh))
    program['u_projection'] = glm.ortho(-hw, hw, -hh, hh, -100, 100)
    # program['u_projection'] = glm.perspective(45.0, ratiow_half * 2.0, 1.0, 100.0)

alpha = 0.0

def mix(x, y, f):
    return (1.0 - f) * x + f * y

def face_rect_size(face_rect):
    return face_rect[1][1] - face_rect[0][1]

def face_rect_center(face_rect):
    p1, p2 = np.asarray(face_rect[0]), np.asarray(face_rect[2])
    return (p1 + p2) * 0.5

@window.event
def on_draw(dt):
    global alpha

    window.clear()

    gl.glEnable(gl.GL_BLEND)
    gl.glDisable(gl.GL_DEPTH_TEST);

    for i in range(len(image_info)):
        img = image_info[i]
        if img is not None:
            modelmat = np.eye(4, dtype=np.float32)

            if i > 0:
                alpha = min(1.0, alpha + dt * 0.4)
                print('current alpha: {}'.format(alpha))
                program['alpha'] = alpha # min(0.6, alpha)
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

                # target position.
                front_center = face_rect_center(img['face_rect'])
                back_center = face_rect_center(image_info[i-1]['face_rect'])
                translate = back_center - front_center
                # print('translation: {}, front center: {}, back center: {}'.format(translate, front_center, back_center))
                translate = mix(np.zeros(2, dtype=np.float32), translate, alpha)
                T2 = glm.translation(translate[0], translate[1], 0.0)
                T1 = np.matrix(glm.translation(front_center[0], front_center[1], 0.0))

                # target scale.
                front_sz = face_rect_size(img['face_rect'])
                back_sz = face_rect_size(image_info[i-1]['face_rect'])
                scale = mix(1.0, back_sz / front_sz, alpha)
                S = np.eye(4, dtype=np.float32)
                S = glm.scale(S, scale, scale, 1.0)

                modelmat = T1.I * S * T1 * T2
            else:
                program['alpha'] = 1.0 - alpha

            program['u_model'] = modelmat
            program['position'] = img['image_rect']
            program['tex'] = img['image'].view(gloo.Texture2D)
            program['u_lineflag'] = 0.0
            program.draw(gl.GL_TRIANGLE_STRIP)

            program['position'] = img['face_rect']
            program['u_lineflag'] = 1.0
            program['u_linecolor'] = [1.0, 0.0, 0.0]
            program.draw(gl.GL_LINE_LOOP)

@window.event
def on_init():
    gl.glLineWidth(2.0)

app.run()
