/*
MIT License

Copyright (c) 2025 Marcin Hajder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "wave_app.hpp"
#include "wave_compute_layer.hpp"
#include "wave_foam_compute_layer.hpp"

#include <iomanip>

////////////////////////////////////////////////////////////////////////////////
void WaveApp::run() {
  // create different models based on CLI options
  if (opts.foam_technique==0)
    _model = std::make_unique<WaveOpenCLLayer>(opts);
  else
    _model = std::make_unique<WaveOpenCLFoamLayer>(opts);

    initWindow();

  _model->init(window);

  mainLoop();

  cleanup();
}

////////////////////////////////////////////////////////////////////////////////
void WaveApp::initWindow()
{
  if (!glfwInit()) {
    throw std::runtime_error("failed to initialize glfw!");
  }

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  window = glfwCreateWindow((int)opts.window_width, (int)opts.window_height,
                            "Ocean surface simulation with OpenCL and Vulkan",
                            nullptr, nullptr);
  glfwSetWindowUserPointer(window, this);
}

////////////////////////////////////////////////////////////////////////////////
void WaveApp::mainLoop()
{
  glfwSetKeyCallback(window, keyboard);
  glfwSetMouseButtonCallback(window, mouse_event);
  glfwSetCursorPosCallback(window, mouse_pos);
  glfwSetScrollCallback(window, mouse_roll);

  while (!glfwWindowShouldClose(window)) {
    show_fps_window_title();
    _model->drawFrame();
    glfwPollEvents();
  }

  _model->wait();
}

////////////////////////////////////////////////////////////////////////////////
void WaveApp::keyboard(int key, int scancode, int action, int mods)
{
  if (action == GLFW_PRESS || action == GLFW_REPEAT) {
    switch (key) {
    case GLFW_KEY_ESCAPE:
      glfwSetWindowShouldClose(window, GLFW_TRUE);
      break;
    case GLFW_KEY_SPACE:
      opts.animate = !opts.animate;
      printf("animation is %s\n", opts.animate ? "ON" : "OFF");
      break;

    case GLFW_KEY_A:
      opts.wind_magnitude += 1.f;
      opts.changed = true;
      break;
    case GLFW_KEY_Z:
      opts.wind_magnitude -= 1.f;
      opts.changed = true;
      break;

    case GLFW_KEY_S:
      opts.wind_angle += 1.f;
      opts.changed = true;
      break;
    case GLFW_KEY_X:
      opts.wind_angle -= 1.f;
      opts.changed = true;
      break;

    case GLFW_KEY_D:
      opts.amplitude += 0.5f;
      opts.changed = true;
      break;
    case GLFW_KEY_C:
      opts.amplitude -= 0.5f;
      opts.changed = true;
      break;

    case GLFW_KEY_F:
      opts.choppiness += 0.5f;
      break;
    case GLFW_KEY_V:
      opts.choppiness -= 0.5f;
      break;

    case GLFW_KEY_G:
      opts.alt_scale += 0.5f;
      break;
    case GLFW_KEY_B:
      opts.alt_scale -= 0.5f;
      break;

    case GLFW_KEY_W:
      opts.wireframe_mode = !opts.wireframe_mode;
      _model->createCommandBuffers();
      break;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
void WaveApp::mouse_event(int button, int action, int mods)
{
  double x, y;
  glfwGetCursorPos(window, &x, &y);
  switch (action) {
  case 0:
    // Button Up
    opts.camera.drag = false;
    break;
  case 1:
    // Button Down
    opts.camera.drag = true;
    opts.camera.begin = glm::vec2(x, y);
    break;
  default:
    break;
  }
}

////////////////////////////////////////////////////////////////////////////////
void WaveApp::mouse_pos(double pX, double pY)
{
  if (!opts.camera.drag)
    return;

  glm::vec2 off = opts.camera.begin - glm::vec2(pX, pY);
  opts.camera.begin = glm::vec2(pX, pY);

  opts.camera.yaw -= off.x * DRAG_SPEED_FAC;
  opts.camera.pitch += off.y * DRAG_SPEED_FAC;

  glm::quat yaw(glm::cos(glm::radians(opts.camera.yaw / 2)),
                glm::vec3(0, 0, 1) *
                    glm::sin(glm::radians(opts.camera.yaw / 2)));
  glm::quat pitch(glm::cos(glm::radians(opts.camera.pitch / 2)),
                  glm::vec3(1, 0, 0) *
                      glm::sin(glm::radians(opts.camera.pitch / 2)));
  glm::mat3 rot_mat(yaw * pitch);
  glm::vec3 dir = rot_mat * glm::vec3(0, 0, -1);

  opts.camera.dir = glm::normalize(dir);
  opts.camera.rvec =
      glm::normalize(glm::cross(opts.camera.dir, glm::vec3(0, 0, 1)));
  opts.camera.up =
      glm::normalize(glm::cross(opts.camera.rvec, opts.camera.dir));
}

////////////////////////////////////////////////////////////////////////////////
void WaveApp::mouse_roll(double offset_x, double offset_y)
{
  opts.camera.eye += opts.camera.dir * (float)offset_y * ROLL_SPEED_FAC;
}

////////////////////////////////////////////////////////////////////////////////
void WaveApp::cleanup()
{
  _model->cleanup();
  glfwDestroyWindow(window);
  glfwTerminate();
}

////////////////////////////////////////////////////////////////////////////////
void WaveApp::keyboard(GLFWwindow *window, int key, int scancode, int action,
                       int mods)
{
  auto app = (WaveApp *)glfwGetWindowUserPointer(window);
  app->keyboard(key, scancode, action, mods);
}

////////////////////////////////////////////////////////////////////////////////
void WaveApp::mouse_event(GLFWwindow *window, int button, int action,
                          int mods)
{
  auto app = (WaveApp *)glfwGetWindowUserPointer(window);
  app->mouse_event(button, action, mods);
}

////////////////////////////////////////////////////////////////////////////////
void WaveApp::mouse_pos(GLFWwindow *window, double pX, double pY)
{
  auto app = (WaveApp *)glfwGetWindowUserPointer(window);
  app->mouse_pos(pX, pY);
}

////////////////////////////////////////////////////////////////////////////////
void WaveApp::mouse_roll(GLFWwindow *window, double oX, double oY)
{
  auto app = (WaveApp *)glfwGetWindowUserPointer(window);
  app->mouse_roll(oX, oY);
}
////////////////////////////////////////////////////////////////////////////////
void WaveApp::show_fps_window_title()
{
    auto app = (WaveApp *)glfwGetWindowUserPointer(window);
    if (app->opts.show_fps)
    {
        auto fps_now = std::chrono::system_clock::now();

        std::chrono::duration<float> elapsed = fps_now - fps_last_time;
        float delta = elapsed.count();

        const float elapsed_tres = 1.f;

        delta_frames++;
        if (window && delta >= 1.f)
        {
            double fps = double(delta_frames) / delta;

            std::stringstream ss;
            ss << "Water sim app, [FPS:" << std::fixed << std::setprecision(2)
               << fps << "]";

            glfwSetWindowTitle(window, ss.str().c_str());

            delta_frames = 0;
            fps_last_time = fps_now;
        }
    }
    else
    {
        fps_last_time = std::chrono::system_clock::now();
        delta_frames = 0;
    }
}
