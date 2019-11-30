from glumpy import app, gl, gloo, data
from glumpy.transforms import PanZoom, Translate
import cv2
import os

wnd_w = 800
wnd_h = 800

window = app.Window(wnd_w, wnd_h)

vertex_shader = """
    attribute vec2 position;
    attribute vec3 color;
    attribute vec2 texcoord;
    varying vec4 v_color;
    varying vec2 v_texcoord;
    uniform float t;
    uniform vec2 aspect;
    uniform vec2 translate;
    uniform float scale;
    void main() {
        // float theta = t * 10.0 / 3.14159;
        // vec2 cossin = vec2(cos(theta), sin(theta));
        // vec2 rot_pos = vec2(cossin.x * position.x - cossin.y * position.y, cossin.y * position.x + cossin.x * position.y);
        gl_Position = <transform(vec4(position * aspect * scale + translate, 0, 1))>;
        v_color = vec4(color, 1);
        v_texcoord = texcoord;
    }
"""

fragment_shader = """
    varying vec4 v_color;
    varying vec2 v_texcoord;
    uniform sampler2D tex;
    uniform float alpha;
    void main() {
        // gl_FragColor = v_color * texture2D(tex, v_texcoord);
        gl_FragColor = texture2D(tex, v_texcoord);
        gl_FragColor.a = alpha;
    }
"""

program = gloo.Program(vertex_shader, fragment_shader)
program["position"] = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
program["color"] = [(0, 0, 0), (0.5, 0, 0), (0.5, 0.5, 0), (0.5, 0.5, 0.5)]
program["texcoord"] = [(0, 1), (0, 0), (1, 1), (1, 0)]

transform = PanZoom(aspect=(wnd_w / wnd_w))
program['transform'] = transform
window.attach(program["transform"])

print(program.all_attributes)

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

# map pos to [-1, 1] space
def map_coord(pos, img_size):
    ndc = pos[:]
    hw = img_size[0] / 2
    hh = img_size[1] / 2
    ndc[0] = (pos[0] - hw) / hw
    ndc[1] = (pos[1] - hh) / hh

    asp = img_size[0] / img_size[1]
    if asp > 1:
        ndc[0] *= asp
    else:
        ndc[1] /= asp

    return ndc

def load_image(image_file_path):
    global last_rect, curr_rect, program, face_cascade, tar_pos, tar_scale, t, last_img, curr_img, last_asp, curr_asp

    last_img = curr_img

    curr_img = data.load(image_file_path)
    # program['tex'] = curr_img
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

@window.event
def on_draw(dt):
    global t, current_img_idx, imge_list

    window.clear()
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    
    t += dt
    if t > duration and current_img_idx < len(imge_list):
        load_image(imge_list[current_img_idx])
        current_img_idx += 1

    gl.glDisable(gl.GL_BLEND)

    # draw current.
    program['tex'] = curr_img
    program['aspect'] = curr_asp
    program['translate'] = (0, 0)
    program['scale'] = 1
    program['alpha'] = 1

    program.draw(gl.GL_TRIANGLE_STRIP)

    f = min(t, duration) / duration

    if last_img != None:
        gl.glEnable(gl.GL_BLEND)

        program['tex'] = last_img
        program['aspect'] = last_asp

        program['translate'] = (tar_pos[0] * f, tar_pos[1] * f)
        program['scale'] = 1 + (tar_scale - 1) * f
        program['alpha'] = max(f, 0.2)

        program.draw(gl.GL_TRIANGLE_STRIP)

app.run()