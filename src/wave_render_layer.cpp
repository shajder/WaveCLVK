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

// The code in this sample was derived from several samples in the Vulkan
// Tutorial: https://vulkan-tutorial.com
//
// The code samples in the Vulkan Tutorial are licensed as CC0 1.0 Universal.

#include "wave_render_layer.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::init(GLFWwindow *window) {
  initCompute();
  initVulkan(window);
  initComputeResources();
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::wait() { vkDeviceWaitIdle(_vulkan.device); }

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::initVulkan(GLFWwindow *window) {
  createInstance();
  setupDebugMessenger();
  createSurface(window);
  pickPhysicalDevice();
  createLogicalDevice();
  createSwapChain();
  createImageViews();
  createRenderPass();
  createUniformBuffer();
  createDescriptorSetLayout();
  createGraphicsPipeline();
  createCommandPool();

  createDepthResources();
  createVertexBuffers();
  createIndexBuffers();

  createFramebuffers();
  createTextureImages();
  createTextureImageViews();
  createTextureSampler();
  createDescriptorPool();
  createDescriptorSets();
  createCommandBuffers();
  createSyncObjects();
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::cleanup() {
  vkDestroyImageView(_vulkan.device, _vulkan.depthImageView, nullptr);
  vkDestroyImage(_vulkan.device, _vulkan.depthImage, nullptr);
  vkFreeMemory(_vulkan.device, _vulkan.depthImageMemory, nullptr);

  for (auto framebuffer : _vulkan.swapChainFramebuffers) {
    vkDestroyFramebuffer(_vulkan.device, framebuffer, nullptr);
  }

  vkDestroyPipeline(_vulkan.device, _vulkan.graphicsPipeline, nullptr);
  vkDestroyPipeline(_vulkan.device, _vulkan.wireframePipeline, nullptr);
  vkDestroyPipelineLayout(_vulkan.device, _vulkan.pipelineLayout, nullptr);
  vkDestroyRenderPass(_vulkan.device, _vulkan.renderPass, nullptr);

  for (auto imageView : _vulkan.swapChainImageViews) {
    vkDestroyImageView(_vulkan.device, imageView, nullptr);
  }

  vkDestroySwapchainKHR(_vulkan.device, _vulkan.swapChain, nullptr);
  vkDestroyDescriptorPool(_vulkan.device, _vulkan.descriptorPool, nullptr);

  vkDestroyBuffer(_vulkan.device, _vulkan.stagingBuffer, nullptr);
  vkFreeMemory(_vulkan.device, _vulkan.stagingBufferMemory, nullptr);

  for (size_t img_num = 0; img_num < _vulkan.textureImages.size(); img_num++) {
    for (auto textureImageView : _vulkan.textureImages[img_num].imageViews) {
      vkDestroyImageView(_vulkan.device, textureImageView, nullptr);
    }
    for (auto textureImage : _vulkan.textureImages[img_num].images) {
      vkDestroyImage(_vulkan.device, textureImage, nullptr);
    }
    for (auto textureImageMemory :
         _vulkan.textureImages[img_num].imageMemories) {
      vkFreeMemory(_vulkan.device, textureImageMemory, nullptr);
    }
  }

  for (size_t sampler_num = 0; sampler_num < _vulkan.textureSampler.size();
       sampler_num++) {
    vkDestroySampler(_vulkan.device, _vulkan.textureSampler[sampler_num],
                     nullptr);
  }

  // cleanup vertices buffers
  for (auto buffer : _vulkan.vertexBuffers) {
    vkDestroyBuffer(_vulkan.device, buffer, nullptr);
  }

  for (auto bufferMemory : _vulkan.vertexBufferMemories) {
    vkFreeMemory(_vulkan.device, bufferMemory, nullptr);
  }

  // cleanup indices buffers
  for (auto & ind_buffer : _vulkan.indexBuffers) {
    for (auto & buffer : ind_buffer.buffers) {
      vkDestroyBuffer(_vulkan.device, buffer, nullptr);
    }
  }

  for (auto & ind_buffer : _vulkan.indexBuffers) {
    for (auto & bufferMemory : ind_buffer.bufferMemories) {
      vkFreeMemory(_vulkan.device, bufferMemory, nullptr);
    }
  }

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vkDestroySemaphore(_vulkan.device, _vulkan.renderFinishedSemaphores[i],
                       nullptr);
    vkDestroySemaphore(_vulkan.device, _vulkan.imageAvailableSemaphores[i],
                       nullptr);
    vkDestroyFence(_vulkan.device, _vulkan.inFlightFences[i], nullptr);
  }

  for (auto &unif_buffer : _vulkan.uniformBuffers) {
    vkDestroyBuffer(_vulkan.device, unif_buffer, nullptr);
  }

  for (auto &unif_buf_mem : _vulkan.uniformBuffersMemory) {
    vkFreeMemory(_vulkan.device, unif_buf_mem, nullptr);
  }

  vkDestroyDescriptorSetLayout(_vulkan.device, _vulkan.descriptorSetLayout,
                               nullptr);

  vkDestroyCommandPool(_vulkan.device, _vulkan.commandPool, nullptr);

  vkDestroyDevice(_vulkan.device, nullptr);

  if (gEnableValidationLayers) {
    DestroyDebugUtilsMessengerEXT(_vulkan.instance, _vulkan.debugMessenger,
                                  nullptr);
  }

  vkDestroySurfaceKHR(_vulkan.instance, _vulkan.surface, nullptr);
  vkDestroyInstance(_vulkan.instance, nullptr);
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createInstance() {
  if (gEnableValidationLayers && !checkValidationLayerSupport())
    throw std::runtime_error(
        "WaveVulkanLayer::createInstance: validation layers not available!");

  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "IFFT Waves";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "Custom";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  auto extensions = getRequiredExtensions();
  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames = extensions.data();

  VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
  if (gEnableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(gValidationLayers.size());
    createInfo.ppEnabledLayerNames = gValidationLayers.data();

    populateDebugMessengerCreateInfo(debugCreateInfo);
    createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
  } else {
    createInfo.enabledLayerCount = 0;

    createInfo.pNext = nullptr;
  }

  if (vkCreateInstance(&createInfo, nullptr, &_vulkan.instance) != VK_SUCCESS)
    throw std::runtime_error(
        "WaveVulkanLayer::createInstance: failed to create instance!");

#ifdef _WIN32
  if (app_opps.useExternalMemory) {
    vkGetMemoryWin32HandleKHR =
        (PFN_vkGetMemoryWin32HandleKHR)vkGetInstanceProcAddr(
            _vulkan.instance, "vkGetMemoryWin32HandleKHR");
    if (vkGetMemoryWin32HandleKHR == nullptr) {
      throw std::runtime_error("couldn't get function pointer for "
                               "vkGetMemoryWin32HandleKHR");
    }
  }
#elif defined(__linux__)
  if (_opts.useExternalMemory) {
    vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetInstanceProcAddr(
        _vulkan.instance, "vkGetMemoryFdKHR");
    if (vkGetMemoryFdKHR == nullptr)
      throw std::runtime_error("WaveVulkanLayer::createInstance: function "
                               "pointer for vkGetMemoryFdKHR not found");
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::populateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
  createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debugCallback;
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::setupDebugMessenger() {
  if (!gEnableValidationLayers)
    return;

  VkDebugUtilsMessengerCreateInfoEXT createInfo;
  populateDebugMessengerCreateInfo(createInfo);

  if (CreateDebugUtilsMessengerEXT(_vulkan.instance, &createInfo, nullptr,
                                   &_vulkan.debugMessenger) != VK_SUCCESS)
    throw std::runtime_error("WaveVulkanLayer::setupDebugMessenger: failed to "
                             "set up debug messenger!");
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createSurface(GLFWwindow *window) {
  if (glfwCreateWindowSurface(_vulkan.instance, window, nullptr,
                              &_vulkan.surface) != VK_SUCCESS)
    throw std::runtime_error("WaveVulkanLayer::createSurface: failed to create "
                             "window _vulkan.surface!");
}

////////////////////////////////////////////////////////////////////////////////

bool WaveVulkanLayer::checkDeviceExtensionSupport(VkPhysicalDevice device) {
  uint32_t extensionCount;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       nullptr);

  std::vector<VkExtensionProperties> availableExtensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                       availableExtensions.data());

  auto extensions = getRequiredDeviceExtensions();
  std::set<std::string> requiredExtensions(extensions.begin(),
                                           extensions.end());

  for (const auto &extension : availableExtensions)
    requiredExtensions.erase(extension.extensionName);

  return requiredExtensions.empty();
}

////////////////////////////////////////////////////////////////////////////////

bool WaveVulkanLayer::isDeviceSuitable(VkPhysicalDevice device) {
  QueueFamilyIndices indices = findQueueFamilies(device);

  bool extensionsSupported = checkDeviceExtensionSupport(device);

  bool swapChainAdequate = false;
  if (extensionsSupported) {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
    swapChainAdequate = !swapChainSupport.formats.empty() &&
                        !swapChainSupport.presentModes.empty();
  }

  return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::pickPhysicalDevice() {
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(_vulkan.instance, &deviceCount, nullptr);

  if (deviceCount == 0)
    throw std::runtime_error("WaveVulkanLayer::pickPhysicalDevice: failed to "
                             "find GPUs with Vulkan support!");

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(_vulkan.instance, &deviceCount, devices.data());

  for (const auto &device : devices) {
    if (isDeviceSuitable(device)) {
      _vulkan.physicalDevice = device;
      break;
    }
  }

  if (_vulkan.physicalDevice == VK_NULL_HANDLE)
    throw std::runtime_error(
        "WaveVulkanLayer::pickPhysicalDevice: failed to find a suitable GPU!");

  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(_vulkan.physicalDevice, &properties);

  printf("WaveVulkanLayer::pickPhysicalDevice: Running on Vulkan physical "
         "_vulkan.device: %s\n",
         properties.deviceName);
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createLogicalDevice() {
  QueueFamilyIndices indices = findQueueFamilies(_vulkan.physicalDevice);

  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily,
                                            indices.presentFamily};

  float queuePriority = 1.0f;
  for (uint32_t queueFamily : uniqueQueueFamilies) {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);
  }

  VkPhysicalDeviceFeatures deviceFeatures{};
  deviceFeatures.fillModeNonSolid = true;

  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

  createInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
  createInfo.pQueueCreateInfos = queueCreateInfos.data();

  createInfo.pEnabledFeatures = &deviceFeatures;

  auto extensions = getRequiredDeviceExtensions();
  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames = extensions.data();

  if (gEnableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(gValidationLayers.size());
    createInfo.ppEnabledLayerNames = gValidationLayers.data();
  } else {
    createInfo.enabledLayerCount = 0;
  }

  if (vkCreateDevice(_vulkan.physicalDevice, &createInfo, nullptr,
                     &_vulkan.device) != VK_SUCCESS)
    throw std::runtime_error("WaveVulkanLayer::createLogicalDevice: failed to "
                             "create logical _vulkan.device!");

  vkGetDeviceQueue(_vulkan.device, indices.graphicsFamily, 0,
                   &_vulkan.graphicsQueue);
  vkGetDeviceQueue(_vulkan.device, indices.presentFamily, 0,
                   &_vulkan.presentQueue);
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createSwapChain() {
  SwapChainSupportDetails swapChainSupport =
      querySwapChainSupport(_vulkan.physicalDevice);

  VkSurfaceFormatKHR surfaceFormat =
      chooseSwapSurfaceFormat(swapChainSupport.formats);
  VkPresentModeKHR presentMode =
      chooseSwapPresentMode(swapChainSupport.presentModes);
  VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
  if (swapChainSupport.capabilities.maxImageCount > 0 &&
      imageCount > swapChainSupport.capabilities.maxImageCount) {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = _vulkan.surface;

  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  QueueFamilyIndices indices = findQueueFamilies(_vulkan.physicalDevice);
  uint32_t queueFamilyIndices[] = {indices.graphicsFamily,
                                   indices.presentFamily};

  if (indices.graphicsFamily != indices.presentFamily) {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;

  createInfo.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(_vulkan.device, &createInfo, nullptr,
                           &_vulkan.swapChain) != VK_SUCCESS)
    throw std::runtime_error(
        "WaveVulkanLayer::createSwapChain: failed to create swap chain!");

  vkGetSwapchainImagesKHR(_vulkan.device, _vulkan.swapChain, &imageCount,
                          nullptr);
  _vulkan.swapChainImages.resize(imageCount);
  vkGetSwapchainImagesKHR(_vulkan.device, _vulkan.swapChain, &imageCount,
                          _vulkan.swapChainImages.data());

  _vulkan.swapChainImageFormat = surfaceFormat.format;
  _vulkan.swapChainExtent = extent;
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createImageViews() {
  _vulkan.swapChainImageViews.resize(_vulkan.swapChainImages.size());

  for (uint32_t i = 0; i < _vulkan.swapChainImages.size(); i++) {
    _vulkan.swapChainImageViews[i] = createImageView(
        _vulkan.swapChainImages[i], _vulkan.swapChainImageFormat,
        VK_IMAGE_ASPECT_COLOR_BIT);
  }
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createRenderPass() {
  VkAttachmentDescription colorAttachment{};
  colorAttachment.format = _vulkan.swapChainImageFormat;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentDescription depthAttachment{};
  depthAttachment.format = findDepthFormat();
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAttachment.finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference colorAttachmentRef{};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthAttachmentRef{};
  depthAttachmentRef.attachment = 1;
  depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;
  subpass.pDepthStencilAttachment = &depthAttachmentRef;

  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  std::array<VkAttachmentDescription, 2> attachments = {colorAttachment,
                                                        depthAttachment};
  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
  renderPassInfo.pAttachments = attachments.data();
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies = &dependency;

  if (vkCreateRenderPass(_vulkan.device, &renderPassInfo, nullptr,
                         &_vulkan.renderPass) != VK_SUCCESS)
    throw std::runtime_error(
        "WaveVulkanLayer::createRenderPass: failed to create render pass!");
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createUniformBuffer() {
  VkDeviceSize bufferSize = sizeof(UniformBufferObject);

  _vulkan.uniformBuffers.resize(_vulkan.swapChainImages.size());
  _vulkan.uniformBuffersMemory.resize(_vulkan.swapChainImages.size());

  _perFrame.resize(_vulkan.swapChainImages.size());

  for (size_t i = 0; i < _vulkan.uniformBuffers.size(); i++) {
    createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 _vulkan.uniformBuffers[i], _vulkan.uniformBuffersMemory[i]);

    vkMapMemory(_vulkan.device, _vulkan.uniformBuffersMemory[i], 0, bufferSize,
                0, &_perFrame[i].buffer_memory);
  }
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createDescriptorSetLayout() {
  VkDescriptorSetLayoutBinding sampler0LayoutBinding{};
  sampler0LayoutBinding.binding = 0;
  sampler0LayoutBinding.descriptorCount = 1;
  sampler0LayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  sampler0LayoutBinding.pImmutableSamplers = nullptr;
  sampler0LayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

  VkDescriptorSetLayoutBinding sampler1LayoutBinding{};
  sampler1LayoutBinding.binding = 1;
  sampler1LayoutBinding.descriptorCount = 1;
  sampler1LayoutBinding.descriptorType =
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  sampler1LayoutBinding.pImmutableSamplers = nullptr;
  sampler1LayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutBinding uniformLayoutBinding{};
  uniformLayoutBinding.binding = 2;
  uniformLayoutBinding.descriptorCount = 1;
  uniformLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  uniformLayoutBinding.pImmutableSamplers = nullptr;
  uniformLayoutBinding.stageFlags =
      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
      sampler0LayoutBinding, sampler1LayoutBinding, uniformLayoutBinding};

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutInfo.pBindings = bindings.data();

  if (vkCreateDescriptorSetLayout(_vulkan.device, &layoutInfo, nullptr,
                                  &_vulkan.descriptorSetLayout) != VK_SUCCESS)
    throw std::runtime_error("WaveVulkanLayer::createDescriptorSetLayout: "
                             "failed to create descriptor set layout!");
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createGraphicsPipeline() {
  auto vertShaderCode = readFile("shaders/ocean.vert.spv");
  auto fragShaderCode = readFile("shaders/ocean.frag.spv");

  VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
  VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

  VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
  vertShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertShaderModule;
  vertShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
  fragShaderStageInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragShaderModule;
  fragShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                    fragShaderStageInfo};

  // vertex info
  auto bindingDescription = Vertex::getBindingDescription();
  auto attributeDescriptions = Vertex::getAttributeDescriptions();

  VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
  vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attributeDescriptions.size());
  vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
  vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

  VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
  inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  inputAssembly.primitiveRestartEnable = VK_TRUE;

  VkViewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)_vulkan.swapChainExtent.width;
  viewport.height = (float)_vulkan.swapChainExtent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor{};
  scissor.offset = {0, 0};
  scissor.extent = _vulkan.swapChainExtent;

  VkPipelineViewportStateCreateInfo viewportState{};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer{};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling{};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineDepthStencilStateCreateInfo depthStencil{};
  depthStencil.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencil.depthTestEnable = VK_TRUE;
  depthStencil.depthWriteEnable = VK_TRUE;
  depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
  depthStencil.depthBoundsTestEnable = VK_FALSE;
  depthStencil.stencilTestEnable = VK_FALSE;

  VkPipelineColorBlendAttachmentState colorBlendAttachment{};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending{};
  colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &_vulkan.descriptorSetLayout;

  if (vkCreatePipelineLayout(_vulkan.device, &pipelineLayoutInfo, nullptr,
                             &_vulkan.pipelineLayout) != VK_SUCCESS)
    throw std::runtime_error("WaveVulkanLayer::createGraphicsPipeline: failed "
                             "to create pipeline layout!");

  VkGraphicsPipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pDepthStencilState = &depthStencil;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.layout = _vulkan.pipelineLayout;
  pipelineInfo.renderPass = _vulkan.renderPass;
  pipelineInfo.subpass = 0;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

  if (vkCreateGraphicsPipelines(_vulkan.device, VK_NULL_HANDLE, 1,
                                &pipelineInfo, nullptr,
                                &_vulkan.graphicsPipeline) != VK_SUCCESS)
    throw std::runtime_error("WaveVulkanLayer::createGraphicsPipeline: failed "
                             "to create graphics pipeline!");

  rasterizer.polygonMode = VK_POLYGON_MODE_LINE;
  if (vkCreateGraphicsPipelines(_vulkan.device, VK_NULL_HANDLE, 1,
                                &pipelineInfo, nullptr,
                                &_vulkan.wireframePipeline) != VK_SUCCESS)
    throw std::runtime_error("WaveVulkanLayer::createGraphicsPipeline: failed "
                             "to create graphics pipeline!");

  vkDestroyShaderModule(_vulkan.device, fragShaderModule, nullptr);
  vkDestroyShaderModule(_vulkan.device, vertShaderModule, nullptr);
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createFramebuffers() {
  _vulkan.swapChainFramebuffers.resize(_vulkan.swapChainImageViews.size());

  for (size_t i = 0; i < _vulkan.swapChainImageViews.size(); i++) {
    std::array<VkImageView, 2> attachments = {_vulkan.swapChainImageViews[i],
                                              _vulkan.depthImageView};

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = _vulkan.renderPass;
    framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    framebufferInfo.pAttachments = attachments.data();
    framebufferInfo.width = _vulkan.swapChainExtent.width;
    framebufferInfo.height = _vulkan.swapChainExtent.height;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(_vulkan.device, &framebufferInfo, nullptr,
                            &_vulkan.swapChainFramebuffers[i]) != VK_SUCCESS)
      throw std::runtime_error(
          "WaveVulkanLayer::createFramebuffers: failed to create framebuffer!");
  }
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createCommandPool() {
  QueueFamilyIndices queueFamilyIndices =
      findQueueFamilies(_vulkan.physicalDevice);

  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;

  if (vkCreateCommandPool(_vulkan.device, &poolInfo, nullptr,
                          &_vulkan.commandPool) != VK_SUCCESS)
    throw std::runtime_error(
        "WaveVulkanLayer::createCommandPool: failed to create command pool!");
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createVertexBuffers() {

  int iCXY = (_opts.ocean_grid_size + 1) * (_opts.ocean_grid_size + 1);
  _vulkan.verts.resize(iCXY);

  cl_float dfY = -0.5 * (_opts.ocean_grid_size * _opts.mesh_spacing),
           dfBaseX = -0.5 * (_opts.ocean_grid_size * _opts.mesh_spacing);
  cl_float tx = 0.f, ty = 0.f, dtx = 1.f / _opts.ocean_grid_size,
           dty = 1.f / _opts.ocean_grid_size;
  for (int iBase = 0, iY = 0; iY <= _opts.ocean_grid_size;
       iY++, iBase += _opts.ocean_grid_size + 1) {
    tx = 0.f;
    double dfX = dfBaseX;
    for (int iX = 0; iX <= _opts.ocean_grid_size; iX++) {
      _vulkan.verts[iBase + iX].pos = glm::vec3(dfX, dfY, 0.0);
      _vulkan.verts[iBase + iX].tc = glm::vec2(tx, ty);
      tx += dtx;
      dfX += _opts.mesh_spacing;
    }
    dfY += _opts.mesh_spacing;
    ty += dty;
  }

  _vulkan.vertexBuffers.resize(_vulkan.swapChainImages.size());
  _vulkan.vertexBufferMemories.resize(_vulkan.swapChainImages.size());

  VkDeviceSize bufferSize = sizeof(_vulkan.verts[0]) * _vulkan.verts.size();

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               stagingBuffer, stagingBufferMemory);

  void *data;
  vkMapMemory(_vulkan.device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, _vulkan.verts.data(), (size_t)bufferSize);
  vkUnmapMemory(_vulkan.device, stagingBufferMemory);

  for (size_t i = 0; i < _vulkan.swapChainImages.size(); i++) {

    // create local memory buffer
    createBuffer(bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _vulkan.vertexBuffers[i],
                 _vulkan.vertexBufferMemories[i]);

    copyBuffer(stagingBuffer, _vulkan.vertexBuffers[i], bufferSize);
  }

  vkDestroyBuffer(_vulkan.device, stagingBuffer, nullptr);
  vkFreeMemory(_vulkan.device, stagingBufferMemory, nullptr);
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createIndexBuffers() {
  size_t totalIndices =
      ((_opts.ocean_grid_size + 1) * 2 + 1) * _opts.ocean_grid_size;
  _vulkan.inds.resize(totalIndices);

  VkDeviceSize bufferSize = sizeof(_vulkan.inds[0]) * _vulkan.inds.size();

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               stagingBuffer, stagingBufferMemory);

  size_t indexCount = 0;
  for (size_t iY = 0; iY < _opts.ocean_grid_size; iY++) {
    size_t iBaseFrom = iY * (_opts.ocean_grid_size + 1);
    size_t iBaseTo = iBaseFrom + _opts.ocean_grid_size + 1;

    for (size_t iX = 0; iX <= _opts.ocean_grid_size; iX++) {
      _vulkan.inds[indexCount++] = static_cast<int>(iBaseFrom + iX);
      _vulkan.inds[indexCount++] = static_cast<int>(iBaseTo + iX);
    }
    _vulkan.inds[indexCount++] = -1;
  }

  void *data;
  vkMapMemory(_vulkan.device, stagingBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, _vulkan.inds.data(), (size_t)bufferSize);
  vkUnmapMemory(_vulkan.device, stagingBufferMemory);

  _vulkan.indexBuffers.resize(1);
  _vulkan.indexBuffers[0].buffers.resize(_vulkan.swapChainImages.size());
  _vulkan.indexBuffers[0].bufferMemories.resize(_vulkan.swapChainImages.size());

  for (size_t i = 0; i < _vulkan.swapChainImages.size(); i++) {
    createBuffer(
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _vulkan.indexBuffers[0].buffers[i],
        _vulkan.indexBuffers[0].bufferMemories[i]);

    copyBuffer(stagingBuffer, _vulkan.indexBuffers[0].buffers[i], bufferSize);
  }

  vkDestroyBuffer(_vulkan.device, stagingBuffer, nullptr);
  vkFreeMemory(_vulkan.device, stagingBufferMemory, nullptr);
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createTextureImages() {
  VkImageTiling tiling =
      _opts.linearImages ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
  VkMemoryPropertyFlags properties =
      _opts.deviceLocalImages ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : 0;

  uint32_t texWidth = static_cast<uint32_t>(_opts.ocean_tex_size);
  uint32_t texHeight = static_cast<uint32_t>(_opts.ocean_tex_size);

  VkDeviceSize imageSize = texWidth * texHeight * 4 * sizeof(float);

  createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               _vulkan.stagingBuffer, _vulkan.stagingBufferMemory);

  for (size_t target = 0; target < _vulkan.textureImages.size(); target++) {
    _vulkan.textureImages[target].images.resize(_vulkan.swapChainImages.size());
    _vulkan.textureImages[target].imageMemories.resize(
        _vulkan.swapChainImages.size());

    for (size_t i = 0; i < _vulkan.swapChainImages.size(); i++) {
      createShareableImage(
          texWidth, texHeight, VK_FORMAT_R32G32B32A32_SFLOAT, tiling,
          VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
          properties, _vulkan.textureImages[target].images[i],
          _vulkan.textureImages[target].imageMemories[i]);
      if (_opts.useExternalMemory)
        transitionImageLayout(_vulkan.textureImages[target].images[i],
                              VK_FORMAT_R32G32B32A32_SFLOAT,
                              VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createTextureImageViews() {
  for (size_t img_num = 0; img_num < _vulkan.textureImages.size(); img_num++) {
    _vulkan.textureImages[img_num].imageViews.resize(
        _vulkan.swapChainImages.size());

    for (size_t i = 0; i < _vulkan.swapChainImages.size(); i++) {
      _vulkan.textureImages[img_num].imageViews[i] = createImageView(
          _vulkan.textureImages[img_num].images[i],
          VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createTextureSampler() {
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

  for (size_t sampler_num = 0; sampler_num < _vulkan.textureSampler.size();
       sampler_num++) {
    if (vkCreateSampler(_vulkan.device, &samplerInfo, nullptr,
                        &_vulkan.textureSampler[sampler_num]) != VK_SUCCESS)
      throw std::runtime_error("WaveVulkanLayer::createTextureSampler: failed "
                               "to create texture sampler!");
  }
}

////////////////////////////////////////////////////////////////////////////////

VkImageView WaveVulkanLayer::createImageView(VkImage image, VkFormat format,
                                             VkImageAspectFlags aspectFlags) {
  VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  viewInfo.pNext = nullptr;
  viewInfo.image = image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1; // VK_REMAINING_MIP_LEVELS;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
  viewInfo.subresourceRange.aspectMask = aspectFlags;

  VkImageView imageView;
  if (vkCreateImageView(_vulkan.device, &viewInfo, nullptr, &imageView) !=
      VK_SUCCESS)
    throw std::runtime_error("WaveVulkanLayer::createImageView: failed to "
                             "create texture image view!");

  return imageView;
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createShareableImage(
    uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
    VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image,
    VkDeviceMemory &imageMemory, VkImageType type) {
  VkExternalMemoryImageCreateInfo externalMemCreateInfo{};
  externalMemCreateInfo.sType =
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;

#ifdef _WIN32
  externalMemCreateInfo.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#elif defined(__linux__)
  externalMemCreateInfo.handleTypes =
      useExternalMemoryType() ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
                              : VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
#endif

  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  if (_opts.useExternalMemory) {
    imageInfo.pNext = &externalMemCreateInfo;
  }

  imageInfo.imageType = type;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateImage(_vulkan.device, &imageInfo, nullptr, &image) != VK_SUCCESS)
    throw std::runtime_error(
        "WaveVulkanLayer::createShareableImage: failed to create image!");

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(_vulkan.device, image, &memRequirements);

  VkExportMemoryAllocateInfo exportMemoryAllocInfo{};
  exportMemoryAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
  exportMemoryAllocInfo.handleTypes = externalMemCreateInfo.handleTypes;

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  if (_opts.useExternalMemory) {
    allocInfo.pNext = &exportMemoryAllocInfo;
  }
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(_vulkan.device, &allocInfo, nullptr, &imageMemory) !=
      VK_SUCCESS)
    throw std::runtime_error("WaveVulkanLayer::createShareableImage: failed to "
                             "allocate image memory!");

  vkBindImageMemory(_vulkan.device, image, imageMemory, 0);
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createImage(uint32_t width, uint32_t height,
                                  VkFormat format, VkImageTiling tiling,
                                  VkImageUsageFlags usage,
                                  VkMemoryPropertyFlags properties,
                                  VkImage &image, VkDeviceMemory &imageMemory) {
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateImage(_vulkan.device, &imageInfo, nullptr, &image) != VK_SUCCESS)
    throw std::runtime_error(
        "WaveVulkanLayer::createImage: failed to create image!");

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(_vulkan.device, image, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(_vulkan.device, &allocInfo, nullptr, &imageMemory) !=
      VK_SUCCESS)
    throw std::runtime_error(
        "WaveVulkanLayer::createImage: failed to allocate image memory!");

  vkBindImageMemory(_vulkan.device, image, imageMemory, 0);
}

////////////////////////////////////////////////////////////////////////////////

VkFormat
WaveVulkanLayer::findSupportedFormat(const std::vector<VkFormat> &candidates,
                                     VkImageTiling tiling,
                                     VkFormatFeatureFlags features) {
  for (VkFormat format : candidates) {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(_vulkan.physicalDevice, format, &props);

    if (tiling == VK_IMAGE_TILING_LINEAR &&
        (props.linearTilingFeatures & features) == features) {
      return format;
    } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
               (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }

  throw std::runtime_error(
      "WaveVulkanLayer::findSupportedFormat: failed to find supported format!");
}

////////////////////////////////////////////////////////////////////////////////

VkFormat WaveVulkanLayer::findDepthFormat() {
  return findSupportedFormat(
      {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
       VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

////////////////////////////////////////////////////////////////////////////////

bool WaveVulkanLayer::hasStencilComponent(VkFormat format) {
  return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
         format == VK_FORMAT_D24_UNORM_S8_UINT;
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createDepthResources() {
  VkFormat depthFormat = findDepthFormat();

  createImage(_vulkan.swapChainExtent.width, _vulkan.swapChainExtent.height,
              depthFormat, VK_IMAGE_TILING_OPTIMAL,
              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _vulkan.depthImage,
              _vulkan.depthImageMemory);
  _vulkan.depthImageView = createImageView(_vulkan.depthImage, depthFormat,
                                           VK_IMAGE_ASPECT_DEPTH_BIT);
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::transitionImageLayout(VkImage image, VkFormat format,
                                            VkImageLayout oldLayout,
                                            VkImageLayout newLayout,
                                            uint32_t layers) {

  VkCommandBuffer commandBuffer = beginSingleTimeCommands();

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = layers;

  VkPipelineStageFlags sourceStage;
  VkPipelineStageFlags destinationStage;

  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
      newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT /*|
                       VK_PIPELINE_STAGE_VERTEX_SHADER_BIT*/
        ;
  } else
    throw std::invalid_argument("WaveVulkanLayer::transitionImageLayout: "
                                "unsupported layout transition!");

  vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0,
                       nullptr, 0, nullptr, 1, &barrier);

  endSingleTimeCommands(commandBuffer);
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::copyBufferToImage(VkBuffer buffer, VkImage image,
                                        uint32_t width, uint32_t height) {
  VkCommandBuffer commandBuffer = beginSingleTimeCommands();

  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {width, height, 1};

  vkCmdCopyBufferToImage(commandBuffer, buffer, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  endSingleTimeCommands(commandBuffer);
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::transitionUniformLayout(VkBuffer buffer,
                                              VkAccessFlagBits src,
                                              VkAccessFlagBits dst) {
  VkCommandBuffer commandBuffer = beginSingleTimeCommands();

  VkDeviceSize bufferSize = sizeof(UniformBufferObject);
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.pNext = nullptr;
  barrier.srcAccessMask = src;
  barrier.dstAccessMask = dst;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = buffer;
  barrier.offset = 0;
  barrier.size = bufferSize;

  VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

  if (src == VK_ACCESS_SHADER_READ_BIT) {
    sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  }

  vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0,
                       nullptr, 1, &barrier, 0, nullptr);

  endSingleTimeCommands(commandBuffer);
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createDescriptorPool() {
  std::array<VkDescriptorPoolSize, 2> poolSizes{};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  poolSizes[0].descriptorCount =
      static_cast<uint32_t>(_vulkan.swapChainImages.size());

  poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[1].descriptorCount =
      static_cast<uint32_t>(_vulkan.swapChainImages.size());

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = static_cast<uint32_t>(_vulkan.swapChainImages.size());

  if (vkCreateDescriptorPool(_vulkan.device, &poolInfo, nullptr,
                             &_vulkan.descriptorPool) != VK_SUCCESS)
    throw std::runtime_error("WaveVulkanLayer::createDescriptorPool: failed to "
                             "create descriptor pool!");
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createDescriptorSets() {
  std::vector<VkDescriptorSetLayout> layouts(_vulkan.swapChainImages.size(),
                                             _vulkan.descriptorSetLayout);
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = _vulkan.descriptorPool;
  allocInfo.descriptorSetCount =
      static_cast<uint32_t>(_vulkan.swapChainImages.size());
  allocInfo.pSetLayouts = layouts.data();

  _vulkan.descriptorSets.resize(_vulkan.swapChainImages.size());
  if (vkAllocateDescriptorSets(_vulkan.device, &allocInfo,
                               _vulkan.descriptorSets.data()) != VK_SUCCESS)
    throw std::runtime_error("WaveVulkanLayer::createDescriptorSets: failed to "
                             "allocate descriptor sets!");

  for (size_t i = 0; i < _vulkan.swapChainImages.size(); i++) {
    VkDescriptorImageInfo imageInfo[(size_t)InteropTexType::IOPT_COUNT] = {0};

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = _vulkan.uniformBuffers[i];
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(UniformBufferObject);

    std::array<VkWriteDescriptorSet, IOPT_COUNT + 1> descriptorWrites{};

    for (cl_int target = 0; target < IOPT_COUNT; target++) {
      imageInfo[target].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo[target].imageView = _vulkan.textureImages[target].imageViews[i];
      imageInfo[target].sampler = _vulkan.textureSampler[target];

      descriptorWrites[target].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrites[target].dstSet = _vulkan.descriptorSets[i];
      descriptorWrites[target].dstBinding = target;
      descriptorWrites[target].dstArrayElement = 0;
      descriptorWrites[target].descriptorType =
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      descriptorWrites[target].descriptorCount = 1;
      descriptorWrites[target].pImageInfo = &imageInfo[target];
    }

    descriptorWrites[IOPT_COUNT].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[IOPT_COUNT].dstSet = _vulkan.descriptorSets[i];
    descriptorWrites[IOPT_COUNT].dstBinding = IOPT_COUNT;
    descriptorWrites[IOPT_COUNT].dstArrayElement = 0;
    descriptorWrites[IOPT_COUNT].descriptorType =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrites[IOPT_COUNT].descriptorCount = 1;
    descriptorWrites[IOPT_COUNT].pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(_vulkan.device,
                           static_cast<uint32_t>(descriptorWrites.size()),
                           descriptorWrites.data(), 0, nullptr);
  }
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                   VkMemoryPropertyFlags properties,
                                   VkBuffer &buffer,
                                   VkDeviceMemory &bufferMemory) {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(_vulkan.device, &bufferInfo, nullptr, &buffer) !=
      VK_SUCCESS)
    throw std::runtime_error(
        "WaveVulkanLayer::createBuffer: failed to create buffer!");

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(_vulkan.device, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(_vulkan.device, &allocInfo, nullptr, &bufferMemory) !=
      VK_SUCCESS)
    throw std::runtime_error(
        "WaveVulkanLayer::createBuffer: failed to allocate buffer memory!");

  vkBindBufferMemory(_vulkan.device, buffer, bufferMemory, 0);
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer,
                                 VkDeviceSize size) {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = _vulkan.commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(_vulkan.device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  VkBufferCopy copyRegion{};
  copyRegion.srcOffset = 0; // Optional
  copyRegion.dstOffset = 0; // Optional
  copyRegion.size = size;
  vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(_vulkan.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(_vulkan.graphicsQueue);

  vkFreeCommandBuffers(_vulkan.device, _vulkan.commandPool, 1, &commandBuffer);
}

////////////////////////////////////////////////////////////////////////////////

uint32_t WaveVulkanLayer::findMemoryType(uint32_t typeFilter,
                                         VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(_vulkan.physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    if ((typeFilter & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
      return i;

  throw std::runtime_error(
      "WaveVulkanLayer::findMemoryType: failed to find suitable memory type!");
}

////////////////////////////////////////////////////////////////////////////////

VkCommandBuffer WaveVulkanLayer::beginSingleTimeCommands() {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = _vulkan.commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(_vulkan.device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  return commandBuffer;
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(_vulkan.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(_vulkan.graphicsQueue);

  vkFreeCommandBuffers(_vulkan.device, _vulkan.commandPool, 1, &commandBuffer);
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createCommandBuffers() {
  _vulkan.commandBuffers.resize(_vulkan.swapChainFramebuffers.size());
  _perFrame.resize(_vulkan.swapChainFramebuffers.size());

  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = _vulkan.commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = (uint32_t)_vulkan.commandBuffers.size();

  if (vkAllocateCommandBuffers(_vulkan.device, &allocInfo,
                               _vulkan.commandBuffers.data()) != VK_SUCCESS)
    throw std::runtime_error("WaveVulkanLayer::createCommandBuffers: failed to "
                             "allocate command buffers!");

  for (size_t i = 0; i < _vulkan.commandBuffers.size(); i++) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(_vulkan.commandBuffers[i], &beginInfo) !=
        VK_SUCCESS)
      throw std::runtime_error("WaveVulkanLayer::createCommandBuffers: failed "
                               "to begin recording command buffer!");

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = _vulkan.renderPass;
    renderPassInfo.framebuffer = _vulkan.swapChainFramebuffers[i];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = _vulkan.swapChainExtent;

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(_vulkan.commandBuffers[i], &renderPassInfo,
                         VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(_vulkan.commandBuffers[i],
                      VK_PIPELINE_BIND_POINT_GRAPHICS,
                      _opts.wireframe_mode ? _vulkan.wireframePipeline
                                           : _vulkan.graphicsPipeline);

    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(_vulkan.commandBuffers[i], 0, 1,
                           &_vulkan.vertexBuffers[i], offsets);

    vkCmdBindDescriptorSets(
        _vulkan.commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
        _vulkan.pipelineLayout, 0, 1, &_vulkan.descriptorSets[i], 0, nullptr);

    for (auto ind_buffer : _vulkan.indexBuffers) {
      vkCmdBindIndexBuffer(_vulkan.commandBuffers[i], ind_buffer.buffers[i], 0,
                           VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(_vulkan.commandBuffers[i],
                       static_cast<uint32_t>(_vulkan.inds.size()), 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(_vulkan.commandBuffers[i]);

    if (vkEndCommandBuffer(_vulkan.commandBuffers[i]) != VK_SUCCESS)
      throw std::runtime_error("WaveVulkanLayer::createCommandBuffers: failed "
                               "to record command buffer!");
  }
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::createSyncObjects() {
  _vulkan.imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
  _vulkan.renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
  _vulkan.inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
  _vulkan.imagesInFlight.resize(_vulkan.swapChainImages.size(), VK_NULL_HANDLE);

  VkExportSemaphoreCreateInfo exportSemaphoreCreateInfo{};
  exportSemaphoreCreateInfo.sType =
      VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;

#ifdef _WIN32
  exportSemaphoreCreateInfo.handleTypes =
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#elif defined(__linux__)
  exportSemaphoreCreateInfo.handleTypes =
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT; // VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
#endif

  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    if (vkCreateSemaphore(_vulkan.device, &semaphoreInfo, nullptr,
                          &_vulkan.imageAvailableSemaphores[i]) != VK_SUCCESS ||
        vkCreateSemaphore(_vulkan.device, &semaphoreInfo, nullptr,
                          &_vulkan.renderFinishedSemaphores[i]) != VK_SUCCESS ||
        vkCreateFence(_vulkan.device, &fenceInfo, nullptr,
                      &_vulkan.inFlightFences[i]) != VK_SUCCESS) {
      throw std::runtime_error("WaveVulkanLayer::createSyncObjects: failed to "
                               "create synchronization objects for a frame!");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::updateUniforms(uint32_t currentImage) {
  UniformBufferObject ubo = _perFrame[currentImage].data;
  ubo.choppiness = _opts.choppiness;
  ubo.alt_scale = _opts.alt_scale;
  ubo.z_range_min = z_range.x;
  ubo.z_range_max = z_range.y;

  // update camera related uniform
  glm::mat4 view_matrix = glm::lookAt(
      _opts.camera.eye, _opts.camera.eye + _opts.camera.dir, _opts.camera.up);

  float fov = glm::radians(60.0);
  float aspect = (float)_opts.window_width / _opts.window_height;
  glm::mat4 proj_matrix = glm::perspective(
      fov, aspect, 1.f, 2.f * _opts.ocean_grid_size * _opts.mesh_spacing);
  proj_matrix[1][1] *= -1;

  ubo.view_mat = view_matrix;
  ubo.proj_mat = proj_matrix;

  memcpy(_perFrame[currentImage].buffer_memory, &ubo,
         sizeof(UniformBufferObject));
}

////////////////////////////////////////////////////////////////////////////////

void WaveVulkanLayer::drawFrame() {
  vkWaitForFences(_vulkan.device, 1, &_vulkan.inFlightFences[_currentFrame],
                  VK_TRUE, UINT64_MAX);

  uint32_t imageIndex;
  vkAcquireNextImageKHR(_vulkan.device, _vulkan.swapChain, UINT64_MAX,
                        _vulkan.imageAvailableSemaphores[_currentFrame],
                        VK_NULL_HANDLE, &imageIndex);

  updateSolver(imageIndex);

  if (_vulkan.imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
    vkWaitForFences(_vulkan.device, 1, &_vulkan.imagesInFlight[imageIndex],
                    VK_TRUE, UINT64_MAX);
  }
  _vulkan.imagesInFlight[imageIndex] = _vulkan.inFlightFences[_currentFrame];

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  std::vector<VkSemaphore> waitSemaphores;
  std::vector<VkPipelineStageFlags> waitStages;
  waitSemaphores.push_back(_vulkan.imageAvailableSemaphores[_currentFrame]);
  waitStages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
  submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
  submitInfo.pWaitSemaphores = waitSemaphores.data();
  submitInfo.pWaitDstStageMask = waitStages.data();

  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &_vulkan.commandBuffers[imageIndex];

  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores =
      &_vulkan.renderFinishedSemaphores[_currentFrame];

  vkResetFences(_vulkan.device, 1, &_vulkan.inFlightFences[_currentFrame]);

  if (vkQueueSubmit(_vulkan.graphicsQueue, 1, &submitInfo,
                    _vulkan.inFlightFences[_currentFrame]) != VK_SUCCESS)
    throw std::runtime_error(
        "WaveVulkanLayer::drawFrame: failed to submit draw command buffer!");

  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores =
      &_vulkan.renderFinishedSemaphores[_currentFrame];

  VkSwapchainKHR swapChains[] = {_vulkan.swapChain};
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = swapChains;

  presentInfo.pImageIndices = &imageIndex;

  vkQueuePresentKHR(_vulkan.presentQueue, &presentInfo);

  _currentFrame = (_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

////////////////////////////////////////////////////////////////////////////////

VkShaderModule
WaveVulkanLayer::createShaderModule(const std::vector<char> &code) {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(_vulkan.device, &createInfo, nullptr,
                           &shaderModule) != VK_SUCCESS)
    throw std::runtime_error(
        "WaveVulkanLayer::createShaderModule: failed to create shader module!");

  return shaderModule;
}

////////////////////////////////////////////////////////////////////////////////

VkSurfaceFormatKHR WaveVulkanLayer::chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &availableFormats) {
  for (const auto &availableFormat : availableFormats)
    if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM)
      return availableFormat;

  return availableFormats[0];
}

////////////////////////////////////////////////////////////////////////////////

VkPresentModeKHR WaveVulkanLayer::chooseSwapPresentMode(
    const std::vector<VkPresentModeKHR> &availablePresentModes) {
  for (const auto &availablePresentMode : availablePresentModes) {
    if (!_opts.immediate) {
      if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
        return availablePresentMode;
    } else {
      if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
        return availablePresentMode;
    }
  }

  return VK_PRESENT_MODE_FIFO_KHR;
}

////////////////////////////////////////////////////////////////////////////////

VkExtent2D WaveVulkanLayer::chooseSwapExtent(
    const VkSurfaceCapabilitiesKHR &capabilities) {
  if (capabilities.currentExtent.width != UINT32_MAX) {
    return capabilities.currentExtent;
  } else {
    VkExtent2D actualExtent = {static_cast<uint32_t>(_opts.window_width),
                               static_cast<uint32_t>(_opts.window_height)};

    actualExtent.width = std::max(
        capabilities.minImageExtent.width,
        std::min(actualExtent.width, capabilities.maxImageExtent.width));
    actualExtent.height = std::max(
        capabilities.minImageExtent.height,
        std::min(actualExtent.height, capabilities.maxImageExtent.height));

    return actualExtent;
  }
}

////////////////////////////////////////////////////////////////////////////////

SwapChainSupportDetails
WaveVulkanLayer::querySwapChainSupport(VkPhysicalDevice device) {
  SwapChainSupportDetails details;

  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, _vulkan.surface,
                                            &details.capabilities);

  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, _vulkan.surface, &formatCount,
                                       nullptr);

  if (formatCount != 0) {
    details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, _vulkan.surface, &formatCount,
                                         details.formats.data());
  }

  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, _vulkan.surface,
                                            &presentModeCount, nullptr);

  if (presentModeCount != 0) {
    details.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, _vulkan.surface,
                                              &presentModeCount,
                                              details.presentModes.data());
  }

  return details;
}

QueueFamilyIndices WaveVulkanLayer::findQueueFamilies(VkPhysicalDevice device) {
  QueueFamilyIndices indices;

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilies.data());

  int i = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
      indices.graphicsFamily = i;

    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _vulkan.surface,
                                         &presentSupport);

    if (presentSupport)
      indices.presentFamily = i;

    if (indices.isComplete())
      break;

    i++;
  }

  return indices;
}

////////////////////////////////////////////////////////////////////////////////

std::vector<const char *> WaveVulkanLayer::getRequiredExtensions() {
  uint32_t glfwExtensionCount = 0;
  const char **glfwExtensions;
  glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  std::vector<const char *> extensions(glfwExtensions,
                                       glfwExtensions + glfwExtensionCount);

  if (_opts.useExternalMemory) {
    extensions.push_back(
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  }
  if (_opts.useExternalMemory) {
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  }
  if (gEnableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

////////////////////////////////////////////////////////////////////////////////

std::vector<const char *> WaveVulkanLayer::getRequiredDeviceExtensions() {
  std::vector<const char *> extensions(gDeviceExtensions);

  if (_opts.useExternalMemory) {
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
#ifdef _WIN32
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#elif defined(__linux__)
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
  }

  return extensions;
}

////////////////////////////////////////////////////////////////////////////////

bool WaveVulkanLayer::checkValidationLayerSupport() {
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const char *layerName : gValidationLayers) {
    bool layerFound = false;

    for (const auto &layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound)
      return false;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////

std::vector<char> WaveVulkanLayer::readFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open())
    throw std::runtime_error("WaveVulkanLayer::readFile: failed to open file!");

  size_t fileSize = (size_t)file.tellg();
  if (filename.find(".spv") == std::string::npos)
    fileSize += 1;
  std::vector<char> buffer(fileSize, '\0');

  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();
  return buffer;
}

////////////////////////////////////////////////////////////////////////////////

VKAPI_ATTR VkBool32 VKAPI_CALL WaveVulkanLayer::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData) {
  fprintf(stderr, "WaveVulkanLayer::debugCallback: validation layer: %s\n",
          pCallbackData->pMessage);

  return VK_FALSE;
}
