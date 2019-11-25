#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Circle.h"

using namespace std;
using namespace render3d;

GLuint g_program = 0;
GLuint g_vbo = 0;
GLuint g_vao = 0;

const int kWindowWidth = 800;
const int kWindowHeight = 600;

std::vector<Circle> g_Circles;
pointf g_CurrentMousePos;
pointf g_LastMousePos;

auto g_CurrentTm = std::chrono::system_clock::now();

bool createGLWindow(int width, int height, GLFWwindow** pp_window)
{
    int ret = GLFW_TRUE;
    ret = glfwInit();
    if (GLFW_TRUE != ret)
    {
        std::cerr << "init GLFW failed.\n";
        return false;
    }

#ifdef __APPLE__
    /* We need to explicitly ask for a 3.2 context on OS X */
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#endif

    GLFWwindow* window = glfwCreateWindow(width, height, "SAMPLE_GEN_AVATAR", NULL, NULL);
    if (!window)
    {
        std::cerr << "Create window failed.\n";
        return false;
    }

    glfwMakeContextCurrent(window);

    glewInit();

    *pp_window = window;
    return true;
}

static bool init();
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
static void display();

int main(int argc, char** argv)
{
    GLFWwindow* p_win = nullptr;
    if (!createGLWindow(kWindowWidth, kWindowHeight, &p_win))
    {
        cerr << "create window failed.\n";
        return -1;
    }

    if (!init())
    {
        cerr << "init failed.\n";
        return -1;
    }

    glfwSetKeyCallback(p_win, key_callback);
    glfwSetCursorPosCallback(p_win, cursor_position_callback);

    while (!glfwWindowShouldClose(p_win))
    {
        display();

        glfwSwapBuffers(p_win);
        glfwPollEvents();
    }
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    // cout << "current mouse position, x: " << x << ", y: " << y << endl;

    // viewport to world.
    pointf pt(xpos, kWindowHeight - ypos);
    pt.x = (pt.x / kWindowWidth) * 2.f - 1.f;
    pt.y = (pt.y / kWindowHeight) * 2.f - 1.f;

    g_CurrentMousePos = pt;
}

GLuint CreateShader(GLenum shader_type, const char* p_shader_src)
{
    GLuint id = glCreateShader(shader_type);
    glShaderSource(id, 1, &p_shader_src, nullptr);
    glCompileShader(id);

    GLint status = GL_TRUE;
    glGetShaderiv(id, GL_COMPILE_STATUS, &status);
    if (GL_TRUE != status)
    {
        GLint len = 0;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &len);

        std::unique_ptr<char[]> p_info_buf(new char[len]);
        glGetShaderInfoLog(id, len, &len, p_info_buf.get());

        cerr << "shader compile failed, log:\n" << p_info_buf.get() << endl;

        glDeleteShader(id);

        return 0;
    }

    return id;
}

void UpdateCircles(const pointf& start, const pointf& end, float delta_sec)
{
    // remove dead Circles.
    auto itEnd = g_Circles.begin();
    for (; itEnd != g_Circles.end() && itEnd->life <= FLT_EPSILON; ++itEnd);
    if (itEnd != g_Circles.begin())
    {
        g_Circles.erase(g_Circles.begin(), itEnd);
    }

    // generate new Circles.
    const float kCircleRate = 200.f;

    pointf vec = end - start;
    float len = vec.length();
    vec /= len;

    int num = len * kCircleRate;
    if (num > 0)
    {
        cout << "generated Circle count: " << num << endl;
    }
    for (int i = 0; i < num; ++i)
    {
        pointf pt = start + vec * i;
        // generate a Circle.
        g_Circles.emplace_back(pt);
    }

    // update Circle state.
    for (auto& c : g_Circles)
    {
        c.life -= delta_sec * 0.5f;
        //c.color[3] = c.normalizedLife();
        //c.radius = Circle::kStartRad + (Circle::kEndRad - Circle::kStartRad) * (1.f - c.normalizedLife());
    }

    // update vbo.
    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, g_Circles.size() * sizeof(Circle), g_Circles.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

bool CreateShaderProgram()
{
    // create program.
    const char* p_vertex_sh = R"glsl(
        #version 330
        layout(location=0) in vec4 position;
        layout(location=1) in float life;
        uniform vec4 start_color;
        uniform vec4 end_color;
        uniform vec2 radius_range;
        out vec4 vcolor;
        out float vradius;
        void main() {
           gl_Position = position;
           vcolor = mix(end_color, start_color, life);
           vradius = mix(radius_range.y, radius_range.x, life);
        }
    )glsl";

    GLuint vs = CreateShader(GL_VERTEX_SHADER, p_vertex_sh);
    if (0 == vs)
    {
        cerr << "create vertex shader failed.\n";
        return false;
    }

    const char* p_fragment_sh =
        "#version 330\n"
        "in vec4 gcolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "   fcolor = gcolor;\n"
        "}";

    GLuint fs = CreateShader(GL_FRAGMENT_SHADER, p_fragment_sh);
    if (0 == fs)
    {
        cerr << "create fragment shader failed.\n";
        return false;
    }

    const char* p_geometry_sh = R"glsl(
        #version 330
        layout(points) in;
        layout(line_strip, max_vertices=64) out;
        in vec4 vcolor[];
        in float vradius[];
        uniform float aspect;
        out vec4 gcolor;
        const float PI = 3.14159;
        void main() {
            gcolor = vcolor[0];

            for (int i=0; i<=32; ++i) {
                float ang = PI * 2.0 / 32.0 * i;
                vec4 offset = vec4(cos(ang) * vradius[0], sin(ang) * vradius[0] * aspect, 0, 0);
                gl_Position = gl_in[0].gl_Position + offset;
                EmitVertex();
            }
            EndPrimitive();
        }
    )glsl";

    GLuint gs = CreateShader(GL_GEOMETRY_SHADER, p_geometry_sh);
    if (0 == gs)
    {
        cerr << "create opengl geometry shader failed.\n";
        return false;
    }

    g_program = glCreateProgram();

    glAttachShader(g_program, vs);
    glAttachShader(g_program, fs);
    glAttachShader(g_program, gs);

    glLinkProgram(g_program);

    GLint status = GL_TRUE;
    glGetProgramiv(g_program, GL_LINK_STATUS, &status);

    if (GL_TRUE != status)
    {
        int len = 0;
        glGetProgramiv(g_program, GL_INFO_LOG_LENGTH, &len);

        std::unique_ptr<char[]> p_info_buf(new char[len]);
        glGetProgramInfoLog(g_program, len, &len, p_info_buf.get());

        cerr << "link openg program failed:\n" << p_info_buf.get() << endl;

        glDeleteProgram(g_program);
        glDeleteShader(vs);
        glDeleteShader(fs);

        return false;
    }

    // bind uniform value.
    glUseProgram(g_program);
    GLint unif_loc = glGetUniformLocation(g_program, "start_color");
    glUniform4fv(unif_loc, 1, Circle::kStartColor);

    unif_loc = glGetUniformLocation(g_program, "end_color");
    glUniform4fv(unif_loc, 1, Circle::kEndColor);
    unif_loc = glGetUniformLocation(g_program, "radius_range");
    glUniform2f(unif_loc, Circle::kStartRad, Circle::kEndRad);

    unif_loc = glGetUniformLocation(g_program, "aspect");
    glUniform1f(unif_loc, kWindowWidth * 1.0f / kWindowHeight);

    return true;
}

bool CreateVAO()
{
    // create vbo.
    glGenBuffers(1, &g_vbo);

    if (0 == g_vbo)
    {
        return false;
    }

    float vbo_data[] = {
        -0.45f,  0.45f, 1.f, 0.f, 0.f, 4.f,
         0.45f,  0.45f, 0.f, 1.f, 0.f, 8.f,
         0.45f, -0.45f, 0.f, 0.f, 1.f, 16.f,
        -0.45f, -0.45f, 1.f, 1.f, 1.f, 32.f,
    };

    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vbo_data), vbo_data, GL_STATIC_DRAW);

    glGenVertexArrays(1, &g_vao);

    //const int kElemCount = 6;
    const int kStride = sizeof(Circle);

    glBindVertexArray(g_vao);

    glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
    
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, kStride, (const void*)Circle::kPositionOffset);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, kStride, (const void*)Circle::kLifeOffset);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    return true;
}

static bool init()
{
    glewInit();
    cout << "OpenGL version: " << (const char*)glGetString(GL_VERSION) << endl;

    if (!CreateShaderProgram())
    {
        cerr << "create opengl shader program failed.\n";
        return false;
    }

    if (!CreateVAO())
    {
        cerr << "create opengl vao failed.\n";
        return false;
    }

    // enable blending.
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    glEnable(GL_MULTISAMPLE);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    return true;
}

static void display()
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // update cirlces.
    auto tm = chrono::system_clock::now();
    float delta_sec = chrono::duration_cast<chrono::milliseconds>(tm - g_CurrentTm).count() / 1000.f;
    g_CurrentTm = tm;

    //cout << "data sec: " << delta_sec << endl;
    UpdateCircles(g_LastMousePos, g_CurrentMousePos, delta_sec);
    g_LastMousePos = g_CurrentMousePos;

    if (!g_Circles.empty())
    {
        glUseProgram(g_program);
        glBindVertexArray(g_vao);

        glPointSize(10.f);
        glDrawArrays(GL_POINTS, 0, g_Circles.size());
    }
}
