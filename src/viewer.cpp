#define GLFW_INCLUDE_GLCOREARB
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <madrona/mw_cpu.hpp>

#include "sim.hpp"
#include "consts.hpp"

using namespace madrona;
using namespace madSameBoy;

static bool readFile(const char *path, std::vector<uint8_t> &out)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        return false;
    }

    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);

    if (size <= 0) {
        return false;
    }

    out.resize(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char *>(out.data()), size)) {
        return false;
    }

    return true;
}

static size_t roundedRomSize(size_t size)
{
    size = (size + 0x3FFF) & ~size_t(0x3FFF);
    while (size & (size - 1)) {
        size |= size >> 1;
        size++;
    }
    if (size < 0x8000) {
        size = 0x8000;
    }
    return size;
}

static GLuint compileShader(GLenum type, const char *src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        GLint log_len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_len);
        std::string log(static_cast<size_t>(log_len), '\0');
        glGetShaderInfoLog(shader, log_len, nullptr, log.data());
        fprintf(stderr, "Shader compile error: %s\n", log.c_str());
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

static GLuint createProgram(const char *vs_src, const char *fs_src)
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vs_src);
    if (!vs) return 0;
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fs_src);
    if (!fs) {
        glDeleteShader(vs);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint status = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        GLint log_len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_len);
        std::string log(static_cast<size_t>(log_len), '\0');
        glGetProgramInfoLog(program, log_len, nullptr, log.data());
        fprintf(stderr, "Program link error: %s\n", log.c_str());
        glDeleteProgram(program);
        return 0;
    }

    return program;
}

static uint8_t sampleInput(GLFWwindow *window)
{
    auto keyPressed = [window](int key) {
        return glfwGetKey(window, key) == GLFW_PRESS;
    };

    uint8_t buttons = 0;
    if (keyPressed(GLFW_KEY_RIGHT)) buttons |= 1 << 0;
    if (keyPressed(GLFW_KEY_LEFT)) buttons |= 1 << 1;
    if (keyPressed(GLFW_KEY_UP)) buttons |= 1 << 2;
    if (keyPressed(GLFW_KEY_DOWN)) buttons |= 1 << 3;
    if (keyPressed(GLFW_KEY_Z)) buttons |= 1 << 4; // A
    if (keyPressed(GLFW_KEY_X)) buttons |= 1 << 5; // B
    if (keyPressed(GLFW_KEY_RIGHT_SHIFT)) buttons |= 1 << 6; // Select
    if (keyPressed(GLFW_KEY_ENTER)) buttons |= 1 << 7; // Start
    return buttons;
}

int main(int argc, char **argv)
{
    const char *rom_path = DEFAULT_ROM_PATH;
    if (argc >= 2) {
        rom_path = argv[1];
    }

    std::vector<uint8_t> rom_data;
    if (!readFile(rom_path, rom_data)) {
        fprintf(stderr, "Failed to read ROM: %s\n", rom_path);
        return 1;
    }

    size_t padded_size = roundedRomSize(rom_data.size());
    std::vector<uint8_t> rom_padded(padded_size, 0xFF);
    std::memcpy(rom_padded.data(), rom_data.data(), rom_data.size());

    Sim::Config sim_cfg {};
    sim_cfg.romData = rom_padded.data();
    sim_cfg.romSize = rom_padded.size();

    constexpr uint32_t num_worlds = 1;
    std::vector<Sim::WorldInit> world_inits(num_worlds);

    TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit> exec({
        .numWorlds = num_worlds,
        .numExportedBuffers = (uint32_t)ExportID::NumExports,
        .numWorkers = 0,
    }, sim_cfg, world_inits.data(), (CountT)TaskGraphID::NumTaskGraphs);

    auto *obs = static_cast<GBObs *>(
        exec.getExported((uint32_t)ExportID::Observation));
    auto *input = static_cast<GBInput *>(
        exec.getExported((uint32_t)ExportID::Input));

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    const int scale = 4;
    GLFWwindow *window = glfwCreateWindow(
        (int)consts::screenWidth * scale,
        (int)consts::screenHeight * scale,
        "Madrona SameBoy",
        nullptr, nullptr);

    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    const char *vs_src =
        "#version 330 core\n"
        "layout(location = 0) in vec2 inPos;\n"
        "layout(location = 1) in vec2 inUV;\n"
        "out vec2 vUV;\n"
        "void main() {\n"
        "    vUV = inUV;\n"
        "    gl_Position = vec4(inPos, 0.0, 1.0);\n"
        "}\n";

    const char *fs_src =
        "#version 330 core\n"
        "in vec2 vUV;\n"
        "out vec4 FragColor;\n"
        "uniform sampler2D uTex;\n"
        "void main() {\n"
        "    vec2 uv = vec2(vUV.x, 1.0 - vUV.y);\n"
        "    float v = texture(uTex, uv).r * 255.0;\n"
        "    float idx = floor(v + 0.5);\n"
        "    float shade = 1.0 - (idx / 3.0);\n"
        "    FragColor = vec4(vec3(shade), 1.0);\n"
        "}\n";

    GLuint program = createProgram(vs_src, fs_src);
    if (!program) {
        glfwTerminate();
        return 1;
    }

    float vertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
    };

    GLuint vao = 0;
    GLuint vbo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8,
                 consts::screenWidth, consts::screenHeight,
                 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);

    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "uTex"), 0);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        input[0].buttons = sampleInput(window);

        exec.run();

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        consts::screenWidth, consts::screenHeight,
                        GL_RED, GL_UNSIGNED_BYTE, obs[0].pixels);

        int fb_w = 0;
        int fb_h = 0;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        glViewport(0, 0, fb_w, fb_h);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glfwSwapBuffers(window);
    }

    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
