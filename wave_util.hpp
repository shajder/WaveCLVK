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
#ifndef OCEANCLVK_UTIL_HPP
#define OCEANCLVK_UTIL_HPP

#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/cl_ext.h>
#include <CL/opencl.hpp>
#if !defined(cl_khr_external_memory)
#error cl_khr_external_memory not found, please update your OpenCL headers!
#endif
#if !defined(cl_khr_external_semaphore)
#error cl_khr_external_semaphore not found, please update your OpenCL headers!
#endif

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <array>
#include <sstream>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

const float DRAG_SPEED_FAC = 0.2f;
const float ROLL_SPEED_FAC = 8.f;
const int MAX_FRAMES_IN_FLIGHT = 2;

static const char *IGetErrorString(int clErrorCode) {
  switch (clErrorCode) {
  case CL_SUCCESS:
    return "CL_SUCCESS";
  case CL_DEVICE_NOT_FOUND:
    return "CL_DEVICE_NOT_FOUND";
  case CL_DEVICE_NOT_AVAILABLE:
    return "CL_DEVICE_NOT_AVAILABLE";
  case CL_COMPILER_NOT_AVAILABLE:
    return "CL_COMPILER_NOT_AVAILABLE";
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case CL_OUT_OF_RESOURCES:
    return "CL_OUT_OF_RESOURCES";
  case CL_OUT_OF_HOST_MEMORY:
    return "CL_OUT_OF_HOST_MEMORY";
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case CL_MEM_COPY_OVERLAP:
    return "CL_MEM_COPY_OVERLAP";
  case CL_IMAGE_FORMAT_MISMATCH:
    return "CL_IMAGE_FORMAT_MISMATCH";
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case CL_BUILD_PROGRAM_FAILURE:
    return "CL_BUILD_PROGRAM_FAILURE";
  case CL_MAP_FAILURE:
    return "CL_MAP_FAILURE";
  case CL_MISALIGNED_SUB_BUFFER_OFFSET:
    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case CL_COMPILE_PROGRAM_FAILURE:
    return "CL_COMPILE_PROGRAM_FAILURE";
  case CL_LINKER_NOT_AVAILABLE:
    return "CL_LINKER_NOT_AVAILABLE";
  case CL_LINK_PROGRAM_FAILURE:
    return "CL_LINK_PROGRAM_FAILURE";
  case CL_DEVICE_PARTITION_FAILED:
    return "CL_DEVICE_PARTITION_FAILED";
  case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
    return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case CL_INVALID_VALUE:
    return "CL_INVALID_VALUE";
  case CL_INVALID_DEVICE_TYPE:
    return "CL_INVALID_DEVICE_TYPE";
  case CL_INVALID_DEVICE:
    return "CL_INVALID_DEVICE";
  case CL_INVALID_CONTEXT:
    return "CL_INVALID_CONTEXT";
  case CL_INVALID_QUEUE_PROPERTIES:
    return "CL_INVALID_QUEUE_PROPERTIES";
  case CL_INVALID_COMMAND_QUEUE:
    return "CL_INVALID_COMMAND_QUEUE";
  case CL_INVALID_HOST_PTR:
    return "CL_INVALID_HOST_PTR";
  case CL_INVALID_MEM_OBJECT:
    return "CL_INVALID_MEM_OBJECT";
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case CL_INVALID_IMAGE_SIZE:
    return "CL_INVALID_IMAGE_SIZE";
  case CL_INVALID_SAMPLER:
    return "CL_INVALID_SAMPLER";
  case CL_INVALID_BINARY:
    return "CL_INVALID_BINARY";
  case CL_INVALID_BUILD_OPTIONS:
    return "CL_INVALID_BUILD_OPTIONS";
  case CL_INVALID_PLATFORM:
    return "CL_INVALID_PLATFORM";
  case CL_INVALID_PROGRAM:
    return "CL_INVALID_PROGRAM";
  case CL_INVALID_PROGRAM_EXECUTABLE:
    return "CL_INVALID_PROGRAM_EXECUTABLE";
  case CL_INVALID_KERNEL_NAME:
    return "CL_INVALID_KERNEL_NAME";
  case CL_INVALID_KERNEL_DEFINITION:
    return "CL_INVALID_KERNEL_DEFINITION";
  case CL_INVALID_KERNEL:
    return "CL_INVALID_KERNEL";
  case CL_INVALID_ARG_INDEX:
    return "CL_INVALID_ARG_INDEX";
  case CL_INVALID_ARG_VALUE:
    return "CL_INVALID_ARG_VALUE";
  case CL_INVALID_ARG_SIZE:
    return "CL_INVALID_ARG_SIZE";
  case CL_INVALID_KERNEL_ARGS:
    return "CL_INVALID_KERNEL_ARGS";
  case CL_INVALID_WORK_DIMENSION:
    return "CL_INVALID_WORK_DIMENSION";
  case CL_INVALID_WORK_GROUP_SIZE:
    return "CL_INVALID_WORK_GROUP_SIZE";
  case CL_INVALID_WORK_ITEM_SIZE:
    return "CL_INVALID_WORK_ITEM_SIZE";
  case CL_INVALID_GLOBAL_OFFSET:
    return "CL_INVALID_GLOBAL_OFFSET";
  case CL_INVALID_EVENT_WAIT_LIST:
    return "CL_INVALID_EVENT_WAIT_LIST";
  case CL_INVALID_EVENT:
    return "CL_INVALID_EVENT";
  case CL_INVALID_OPERATION:
    return "CL_INVALID_OPERATION";
  case CL_INVALID_GL_OBJECT:
    return "CL_INVALID_GL_OBJECT";
  case CL_INVALID_BUFFER_SIZE:
    return "CL_INVALID_BUFFER_SIZE";
  case CL_INVALID_MIP_LEVEL:
    return "CL_INVALID_MIP_LEVEL";
  case CL_INVALID_GLOBAL_WORK_SIZE:
    return "CL_INVALID_GLOBAL_WORK_SIZE";
  case CL_INVALID_PROPERTY:
    return "CL_INVALID_PROPERTY";
  case CL_INVALID_IMAGE_DESCRIPTOR:
    return "CL_INVALID_IMAGE_DESCRIPTOR";
  case CL_INVALID_COMPILER_OPTIONS:
    return "CL_INVALID_COMPILER_OPTIONS";
  case CL_INVALID_LINKER_OPTIONS:
    return "CL_INVALID_LINKER_OPTIONS";
  case CL_INVALID_DEVICE_PARTITION_COUNT:
    return "CL_INVALID_DEVICE_PARTITION_COUNT";
  case CL_INVALID_PIPE_SIZE:
    return "CL_INVALID_PIPE_SIZE";
  case CL_INVALID_DEVICE_QUEUE:
    return "CL_INVALID_DEVICE_QUEUE";
  case CL_INVALID_SPEC_ID:
    return "CL_INVALID_SPEC_ID";
  case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
    return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
  default:
    return "(unknown)";
  }
}

#define print_error(errCode, msg)                                              \
  printf("ERROR: %s! (%s from %s:%d)\n", msg, IGetErrorString(errCode),        \
         __FILE__, __LINE__);

#define test_error(errCode, msg)                                               \
  {                                                                            \
    auto errCodeResult = errCode;                                              \
    if (errCodeResult != CL_SUCCESS) {                                         \
      print_error(errCodeResult, msg);                                         \
      return errCode;                                                          \
    }                                                                          \
  }

/* Determines if an extension is supported by a device. */
static int isExtensionSupported(cl_device_id device,
                                const char *extensionName) {
  size_t size = 0;
  int err = clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &size);
  if (err != CL_SUCCESS || size == 0)
    throw std::runtime_error("clGetDeviceInfo failed\n");

  std::vector<char> info(size);

  err =
      clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, size, info.data(), nullptr);
  if (err != CL_SUCCESS)
    throw std::runtime_error("clGetDeviceInfo failed\n");

  /* The returned string does not include the null terminator. */
  std::string extString = std::string(info.data(), size - 1);

  std::istringstream ss(extString);

  while (ss) {
    std::string found;
    ss >> found;
    if (found == extensionName)
      return true;
  }
  return false;
}

static uint32_t reverse_bits(uint32_t n, uint32_t log_2_N) {
  uint32_t r = 0;
  for (int j = 0; j < log_2_N; j++) {
    r = (r << 1) + (n & 1);
    n >>= 1;
  }
  return r;
}

const std::vector<const char *> gValidationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

const std::vector<const char *> gDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

#ifdef NDEBUG
const bool gEnableValidationLayers = false;
#else
const bool gEnableValidationLayers = true;
#endif

static VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
    const VkAllocationCallbacks *pAllocator,
    VkDebugUtilsMessengerEXT *pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

static void
DestroyDebugUtilsMessengerEXT(VkInstance instance,
                              VkDebugUtilsMessengerEXT debugMessenger,
                              const VkAllocationCallbacks *pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

struct QueueFamilyIndices {
  uint32_t graphicsFamily;
  uint32_t presentFamily;

  QueueFamilyIndices() : graphicsFamily(~0), presentFamily(~0) {}

  bool isComplete() { return graphicsFamily != ~0 && presentFamily != ~0; }
};

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

struct UniformBufferObject {
  alignas(4) glm::mat4 view_mat;
  alignas(4) glm::mat4 proj_mat;
  alignas(4) glm::vec3 sun_dir = glm::normalize(glm::vec3(0.f, 1.f, 1.f));
  alignas(4) std::float_t z_range_min = 0.f;
  alignas(4) std::float_t z_range_max = 2.f;
  alignas(4) std::float_t choppiness = 1.f;
  alignas(4) std::float_t alt_scale = 1.f;
};

struct Vertex {

  glm::vec3 pos;
  glm::vec2 tc;

  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription{};

    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    return bindingDescription;
  }

  static std::array<VkVertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, tc);

    return attributeDescriptions;
  }
};

struct Camera {
  glm::vec3 eye = glm::vec3(0.0f, 0.0f, 20.0f);
  glm::vec3 dir = glm::vec3(-0.57359f, 0.73945f, -0.35241f);
  glm::vec3 up = glm::vec3(-0.2159f, 0.27846f, 0.93584f);
  glm::vec3 rvec = glm::vec3(1.0f, 0.0f, 0.0f);
  glm::vec2 begin = glm::vec2(-1.0f, -1.0f);
  float yaw = 37.8f;
  float pitch = 69.3649f;
  bool drag = false;
};

struct CliOptions {
  size_t window_width = 1024;
  size_t window_height = 1024;

  unsigned short plat_index = 0;
  unsigned short dev_index = 0;
  unsigned short technique = 0;
  unsigned short foam_technique = 0;

  bool immediate = false;

  bool linearImages = false;
  bool deviceLocalImages = true;

  bool useExternalMemory = true;
};

struct SharedOptions : public CliOptions {
  Camera camera;

  // ocean texture size - assume uniform x/y
  size_t ocean_tex_size = 512;

  size_t group_size =
      16; // must be aligned with local memory array in normal kernel

  // mesh patch size - assume uniform x/y
  size_t ocean_grid_size = 256;

  // mesh patch spacing
  float mesh_spacing = 2.f;

  bool animate = true;

  bool show_fps = true;

  // foam simulation range multiplier
  unsigned short foam_scope_mult = 2;

  // ocean parameters changed - rebuild initial spectrum resources
  bool changed = true;
  bool twiddle_factors_init = true;

  // ocean in-factors
  float wind_magnitude = 30.f;
  float wind_angle = 45.f;
  float choppiness = 10.f;
  float alt_scale = 20.f;

  float amplitude = 80.f;
  float supress_factor = 0.1f;

  // env factors
  int sun_elevation = 0;
  int sun_azimuth = 90;
  bool wireframe_mode = false;
};

#endif // OCEANCLVK_UTIL_HPP
