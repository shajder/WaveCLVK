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
#ifndef _WAVE_MODEL_BASE_HPP_
#define _WAVE_MODEL_BASE_HPP_

#include "wave_util.hpp"

#include <chrono>

// GLM includes
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class WaveVulkanLayer {

protected:
  SharedOptions &_opts;

public:
  WaveVulkanLayer(SharedOptions &opts) : _opts(opts) {}

  void init(GLFWwindow *window);
  void drawFrame();
  void wait();
  void createCommandBuffers();

  enum InteropTexType { IOPT_DISPLACEMENT = 0, IOPT_NORMAL_MAP, IOPT_COUNT };

protected:

#ifdef _WIN32
    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR = NULL;
    PFN_vkGetSemaphoreWin32HandleKHR vkGetSemaphoreWin32HandleKHR = NULL;
#elif defined(__linux__)
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = NULL;
    PFN_vkGetSemaphoreFdKHR vkGetSemaphoreFdKHR = NULL;
#endif

  glm::vec2 z_range = glm::vec2(0, 0);

  std::chrono::system_clock::time_point start =
      std::chrono::system_clock::now();

  struct VkIface {
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    VkRenderPass renderPass;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkPipeline wireframePipeline;

    VkCommandPool commandPool;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    struct TextureInterop {
      std::vector<VkImage> images;
      std::vector<VkDeviceMemory> imageMemories;
      std::vector<VkImageView> imageViews;
    };

    // vulkan-opencl interop resources
    std::array<TextureInterop, IOPT_COUNT> textureImages;

    // Ocean grid vertices and related buffers
    std::vector<Vertex> verts;
    std::vector<VkBuffer> vertexBuffers;
    std::vector<VkDeviceMemory> vertexBufferMemories;

    std::vector<std::uint32_t> inds;
    struct IndexBuffer {
      std::vector<VkBuffer> buffers;
      std::vector<VkDeviceMemory> bufferMemories;
    };
    std::vector<IndexBuffer> indexBuffers;

    std::array<VkSampler, IOPT_COUNT> textureSampler;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkSemaphore> openclFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
  };

  VkIface _vulkan;

  size_t _currentFrame = 0;

  struct PerFrameData {
    UniformBufferObject data;
    void *buffer_memory;
  };

  std::vector<PerFrameData> _perFrame;

public:
  virtual void cleanup();

  virtual void pickPhysicalDevice();

  virtual void initCompute() = 0;

  virtual void initComputeResources() = 0;

  virtual void updateSolver(uint32_t currentImage) = 0;

  virtual bool useExternalMemoryType() = 0;

protected:
  void initVulkan(GLFWwindow *window);

  void createInstance();

  void populateDebugMessengerCreateInfo(
      VkDebugUtilsMessengerCreateInfoEXT &createInfo);

  void setupDebugMessenger();

  void createSurface(GLFWwindow *window);

  void createLogicalDevice();

  void createSwapChain();

  void createImageViews();

  void createRenderPass();

  void createUniformBuffer();

  void createDescriptorSetLayout();

  void createGraphicsPipeline();

  void createFramebuffers();

  void createCommandPool();

  void createVertexBuffers();

  void createIndexBuffers();

  void createTextureImages();

  void createTextureImageViews();

  void createTextureSampler();

  bool isDeviceSuitable(VkPhysicalDevice device);

  bool checkDeviceExtensionSupport(VkPhysicalDevice device);

  VkImageView createImageView(VkImage image, VkFormat format,
                              VkImageAspectFlags aspectFlags);

  void createShareableImage(uint32_t width, uint32_t height, VkFormat format,
                            VkImageTiling tiling, VkImageUsageFlags usage,
                            VkMemoryPropertyFlags properties, VkImage &image,
                            VkDeviceMemory &imageMemory,
                            VkImageType type = VK_IMAGE_TYPE_2D);

  void createImage(uint32_t width, uint32_t height, VkFormat format,
                   VkImageTiling tiling, VkImageUsageFlags usage,
                   VkMemoryPropertyFlags properties, VkImage &image,
                   VkDeviceMemory &imageMemory);

  VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features);

  VkFormat findDepthFormat();

  bool hasStencilComponent(VkFormat format);

  void createDepthResources();

  void transitionImageLayout(VkImage image, VkFormat format,
                             VkImageLayout oldLayout, VkImageLayout newLayout,
                             uint32_t layers = 1);

  void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width,
                         uint32_t height);

  void transitionUniformLayout(VkBuffer buffer, VkAccessFlagBits src,
                               VkAccessFlagBits dst);

  void createDescriptorPool();

  void createDescriptorSets();

  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer &buffer,
                    VkDeviceMemory &bufferMemory);

  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties);

  VkCommandBuffer beginSingleTimeCommands();

  void endSingleTimeCommands(VkCommandBuffer commandBuffer);

  void createSyncObjects();

  void updateUniforms(uint32_t currentImage);

  VkShaderModule createShaderModule(const std::vector<char> &code);

  VkSurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR> &availableFormats);

  VkPresentModeKHR chooseSwapPresentMode(
      const std::vector<VkPresentModeKHR> &availablePresentModes);

  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities);

  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

  std::vector<const char *> getRequiredExtensions();

  std::vector<const char *> getRequiredDeviceExtensions();

  bool checkValidationLayerSupport();

  static std::vector<char> readFile(const std::string &filename);

  static VKAPI_ATTR VkBool32 VKAPI_CALL
  debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                VkDebugUtilsMessageTypeFlagsEXT messageType,
                const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                void *pUserData);
};

#endif //_WAVE_MODEL_BASE_HPP_
