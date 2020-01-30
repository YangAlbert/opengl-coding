from glumpy import gloo, app, gl, glm
import numpy as np

vertex_shader = """
    attribute vec2 position;
    void main()
    {
        gl_Position = vec4(position, 0, 1);
    }
"""

fragment_shader = """
    uniform vec4 u_color;
    void main()
    {
        gl_FragColor = u_color;
    }
"""

program = gloo.Program(vertex_shader, fragment_shader)

V = np.zeros(4, [('position', np.float32, 2)])

# CAUTION! attribute array dimension must match attribute vector dimension!
V['position'] = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
V = V.view(gloo.VertexBuffer)

program['u_color'] = [1.0, 1.0, 0.0, 1.0]

program.bind(V)

window = app.Window(1024, 768)

@window.event
def on_draw(dt):
    window.clear()

    program.draw(gl.GL_TRIANGLE_STRIP)

@window.event
def on_init():
    gl.glFrontFace(gl.GL_CW)
    gl.glDisable(gl.GL_CULL_FACE)
    gl.glDisable(gl.GL_DEPTH_TEST)

app.run()