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
#ifndef _WAVE_APP_HPP_
#define _WAVE_APP_HPP_

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <memory>

// GLM includes
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "wave_render_layer.hpp"

class WaveApp {

public:
  void run();

public:
  SharedOptions opts;

  std::chrono::system_clock::time_point fps_last_time = std::chrono::system_clock::now();

  unsigned int delta_frames=0;
private:

  std::unique_ptr<WaveVulkanLayer> _model;

  GLFWwindow *window;

  void initWindow();

  void mainLoop();

  void keyboard(int key, int scancode, int action, int mods);

  void mouse_event(int button, int action, int mods);

  void mouse_pos(double pX, double pY);

  void mouse_roll(double offset_x, double offset_y);

  void cleanup();

  void show_fps_window_title();

  static void keyboard(GLFWwindow *window, int key, int scancode, int action,
                       int mods);

  static void mouse_event(GLFWwindow *window, int button, int action, int mods);

  static void mouse_pos(GLFWwindow *window, double pX, double pY);

  static void mouse_roll(GLFWwindow *window, double oX, double oY);
};
#endif //_WAVE_APP_HPP_
