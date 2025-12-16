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
#include "wave_compute_layer.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

////////////////////////////////////////////////////////////////////////////////
void WaveOpenCLLayer::initCompute() {
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  printf("Running on platform: %s\n",
         platforms[_opts.plat_index].getInfo<CL_PLATFORM_NAME>().c_str());
  std::vector<cl::Device> devices;
  platforms[_opts.plat_index].getDevices(CL_DEVICE_TYPE_ALL, &devices);

  printf("Running on device: %s\n",
         devices[_opts.dev_index].getInfo<CL_DEVICE_NAME>().c_str());

  checkOpenCLExternalMemorySupport(devices[_opts.dev_index]);

  int error = CL_SUCCESS;
  error |=
      clGetDeviceInfo(devices[_opts.dev_index](), CL_DEVICE_IMAGE2D_MAX_WIDTH,
                      sizeof(ocl_max_img2d_width), &ocl_max_img2d_width, NULL);
  error |=
      clGetDeviceInfo(devices[_opts.dev_index](), CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                      sizeof(ocl_max_alloc_size), &ocl_max_alloc_size, NULL);
  error |=
      clGetDeviceInfo(devices[_opts.dev_index](), CL_DEVICE_GLOBAL_MEM_SIZE,
                      sizeof(ocl_mem_size), &ocl_mem_size, NULL);

  if (error != CL_SUCCESS)
    printf("WaveOpenCLLayer::initCompute: clGetDeviceInfo error: %d\n", error);

  cl_device = devices[_opts.dev_index];
  context = cl::Context{devices[_opts.dev_index]};
  commandQueue = cl::CommandQueue{context, devices[_opts.dev_index]};

  if (_opts.technique == 0) {
    _opts.alt_scale /= 2;
  }

  auto build_kernel = [&](const char *src_file, cl::Kernel &kernel,
                          const char *name) {
    try {
      std::string kernel_code = readFile(src_file).data();
      cl::Program program{context, kernel_code};
      program.build();
      kernel = cl::Kernel{program, name};
    } catch (const cl::BuildError &e) {
      auto bl = e.getBuildLog();
      std::cout << "Build OpenCL " << name << " kernel error: " << std::endl;
      for (auto &elem : bl)
        std::cout << elem.second << std::endl;
      exit(1);
    }
  };

  const char *init_spectrum = _opts.technique == 0 ? "init_spectrum_phillips.cl"
                                                   : "init_spectrum_jonswap.cl";

  build_kernel(init_spectrum, init_spectrum_kernel, "init_spectrum");

  build_kernel("kernels/twiddle.cl", twiddle_kernel, "generate");
  build_kernel("kernels/time_spectrum.cl", time_spectrum_kernel, "spectrum");
  build_kernel("kernels/fft_kernel.cl", fft_kernel, "fft_1D");
  build_kernel("kernels/inversion.cl", inversion_kernel, "inversion");
  build_kernel("kernels/normals.cl", normals_kernel, "normals");

  build_kernel("kernels/reduce_ranges.cl", z_ranges_kernel, "reduce_ranges");

  setupFoamSolver("kernels/foam.cl");
}

////////////////////////////////////////////////////////////////////////////////

void WaveOpenCLLayer::setupFoamSolver(const std::string &filename) {
  try {
    std::string kernel_code = readFile(filename).data();
    cl::Program program{context, kernel_code};
    program.build();
    foam_kernel = cl::Kernel{program, "update_foam"};
  } catch (const cl::BuildError &e) {
    auto bl = e.getBuildLog();
    std::cout << "WaveOpenCLLayer::setupFoamSolver: Build OpenCL update_foam "
                 "kernel error: "
              << std::endl;
    for (auto &elem : bl)
      std::cout << elem.second << std::endl;
    exit(1);
  }
}

////////////////////////////////////////////////////////////////////////////////

void WaveOpenCLLayer::initComputeResources() {
  // init intermediate opencl resources
  try {
    {
      std::vector<cl_float4> phase_array(_opts.ocean_tex_size *
                                         _opts.ocean_tex_size);
      std::random_device dev;
      std::mt19937 rng(dev());
      std::uniform_real_distribution<float> dist(0.f, 1.f);

      for (size_t i = 0; i < phase_array.size(); ++i)
        phase_array[i] = {dist(rng), dist(rng), dist(rng), dist(rng)};

      noise_mem = std::make_unique<cl::Image2D>(
          context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
          cl::ImageFormat(CL_RGBA, CL_FLOAT), _opts.ocean_tex_size,
          _opts.ocean_tex_size, 0, phase_array.data());
    }

    hkt_pong_mem = std::make_unique<cl::Image2D>(
        context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
        _opts.ocean_tex_size, _opts.ocean_tex_size);

    dxyz_coef_mem[0] = std::make_unique<cl::Image2D>(
        context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
        _opts.ocean_tex_size, _opts.ocean_tex_size);

    dxyz_coef_mem[1] = std::make_unique<cl::Image2D>(
        context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
        _opts.ocean_tex_size, _opts.ocean_tex_size);

    dxyz_coef_mem[2] = std::make_unique<cl::Image2D>(
        context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
        _opts.ocean_tex_size, _opts.ocean_tex_size);

    h0k_mem = std::make_unique<cl::Image2D>(
        context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT),
        _opts.ocean_tex_size, _opts.ocean_tex_size);

    z_ranges_mem[0] = std::make_unique<cl::Image2D>(
        context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
        _opts.ocean_tex_size, _opts.ocean_tex_size);

    z_ranges_mem[1] = std::make_unique<cl::Image2D>(
        context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
        _opts.ocean_tex_size / 2, _opts.ocean_tex_size / 2);

    size_t log_2_N =
        (size_t)((log((float)_opts.ocean_tex_size) / log(2.f)) - 1);
    twiddle_factors_mem = std::make_unique<cl::Image2D>(
        context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT), log_2_N,
        _opts.ocean_tex_size);

    for (size_t target = 0; target < IOPT_COUNT; target++) {
      mems[target].resize(_vulkan.swapChainImages.size());

      for (size_t i = 0; i < _vulkan.swapChainImages.size(); i++) {
        if (_opts.useExternalMemory) {
#ifdef _WIN32
          HANDLE handle = NULL;
          VkMemoryGetWin32HandleInfoKHR getWin32HandleInfo{};
          getWin32HandleInfo.sType =
              VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
          getWin32HandleInfo.memory =
              _vulkan.textureImages[target].imageMemories[i];
          getWin32HandleInfo.handleType =
              VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
          vkGetMemoryWin32HandleKHR(device, &getWin32HandleInfo, &handle);

          const cl_mem_properties props[] = {
              externalMemType,
              (cl_mem_properties)handle,
              0,
          };
#elif defined(__linux__)
          int fd = 0;
          VkMemoryGetFdInfoKHR getFdInfo{};
          getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
          getFdInfo.memory = _vulkan.textureImages[target]
                                 .imageMemories[i]; // textureImageMemories[i];
          getFdInfo.handleType =
              externalMemType == CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR
                  ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
                  : VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
          vkGetMemoryFdKHR(_vulkan.device, &getFdInfo, &fd);

          const cl_mem_properties props[] = {
              externalMemType,
              (cl_mem_properties)fd,
              0,
          };
#else
          const cl_mem_properties *props = NULL;
#endif

#if 0
                    cl::vector<cl_mem_properties> vprops(
                        sizeof(props) / sizeof(props[0]));
                    std::memcpy(vprops.data(), props,
                                sizeof(cl_mem_properties)
                                    * vprops.size());

                    mems[target][i].reset(new cl::Image2D(
                        context, vprops, CL_MEM_READ_WRITE,
                        cl::ImageFormat(CL_RGBA, CL_FLOAT),
                        _opts.ocean_tex_size, _opts.ocean_tex_size));
#else

          cl_image_format format{};
          format.image_channel_order = CL_RGBA;
          format.image_channel_data_type = CL_FLOAT;

          cl_image_desc desc{};
          desc.image_type = CL_MEM_OBJECT_IMAGE2D;
          desc.image_width = _opts.ocean_tex_size;
          desc.image_height = _opts.ocean_tex_size;

          mems[target][i].reset(new cl::Image2D{
              clCreateImageWithProperties(context(), props, CL_MEM_READ_WRITE,
                                          &format, &desc, NULL, NULL)});
#endif
        } else {
          mems[target][i].reset(new cl::Image2D{
              context, CL_MEM_READ_WRITE, cl::ImageFormat{CL_RGBA, CL_FLOAT},
              _opts.ocean_tex_size, _opts.ocean_tex_size});
        }
      }
    }
  } catch (const cl::Error &e) {
    printf("WaveOpenCLLayer::initComputeResources: OpenCL %s image error: %s\n",
           e.what(), IGetErrorString(e.err()));
    exit(1);
  }
}

////////////////////////////////////////////////////////////////////////////////

void WaveOpenCLLayer::cleanup() {
  for (auto semaphore : signalSemaphores) {
    semaphore.release();
  }

  WaveVulkanLayer::cleanup();
}

////////////////////////////////////////////////////////////////////////////////

bool WaveOpenCLLayer::useExternalMemoryType() {
  return externalMemType == CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR;
}

////////////////////////////////////////////////////////////////////////////////

void WaveOpenCLLayer::pickPhysicalDevice() {
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(_vulkan.instance, &deviceCount, nullptr);

  if (deviceCount == 0)
    throw std::runtime_error("WaveOpenCLLayer::pickPhysicalDevice: failed to "
                             "find GPUs with Vulkan support!");

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(_vulkan.instance, &deviceCount, devices.data());

  cl_uchar uuid[CL_UUID_SIZE_KHR];
  cl_int errNum = clGetDeviceInfo(cl_device(), CL_DEVICE_UUID_KHR,
                                  CL_UUID_SIZE_KHR, uuid, nullptr);
  if (CL_SUCCESS != errNum)
    throw std::runtime_error("WaveOpenCLLayer::pickPhysicalDevice Error: "
                             "clGetDeviceInfo failed with error\n");

  for (cl_int pdIdx = 0; pdIdx < devices.size(); pdIdx++) {
    VkPhysicalDeviceIDPropertiesKHR vkPhysicalDeviceIDPropertiesKHR = {};
    vkPhysicalDeviceIDPropertiesKHR.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR;
    vkPhysicalDeviceIDPropertiesKHR.pNext = NULL;

    VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
    vkPhysicalDeviceProperties2.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
    vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDPropertiesKHR;

    vkGetPhysicalDeviceProperties2(devices[pdIdx],
                                   &vkPhysicalDeviceProperties2);

    if (!memcmp(&uuid, vkPhysicalDeviceIDPropertiesKHR.deviceUUID,
                VK_UUID_SIZE)) {
      std::cout << "Selected physical device = " << pdIdx << std::endl;

      _vulkan.physicalDevice = devices[pdIdx];
      break;
    }
  }

  if (_vulkan.physicalDevice == VK_NULL_HANDLE)
    throw std::runtime_error(
        "WaveOpenCLLayer::pickPhysicalDevice: failed to find a suitable GPU!");

  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(_vulkan.physicalDevice, &properties);

  printf("Running on Vulkan physical device: %s\n", properties.deviceName);
}

////////////////////////////////////////////////////////////////////////////////

void WaveOpenCLLayer::updateSimulation(uint32_t currentImage, float elapsed) {
  cl_int2 patch = cl_int2{(int)(_opts.ocean_grid_size * _opts.mesh_spacing),
                          (int)_opts.ocean_tex_size};

  assert(_opts.group_size > 0);

  cl::NDRange lws = cl::NDRange{_opts.group_size, _opts.group_size};

  if (_opts.twiddle_factors_init) {
    try {
      size_t log_2_N =
          (size_t)((log((float)_opts.ocean_tex_size) / log(2.f)) - 1);

      /// Prepare vector of values to extract results
      std::vector<cl_int> v(_opts.ocean_tex_size);
      for (int i = 0; i < _opts.ocean_tex_size; i++) {
        int x = reverse_bits(i, log_2_N);
        v[i] = x;
      }

      /// Initialize device-side storage
      cl::Buffer bit_reversed_inds_mem{context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(cl_int) * v.size(), v.data()};

      twiddle_kernel.setArg(0, cl_int(_opts.ocean_tex_size));
      twiddle_kernel.setArg(1, bit_reversed_inds_mem);
      twiddle_kernel.setArg(2, *twiddle_factors_mem);

      commandQueue.enqueueNDRangeKernel(
          twiddle_kernel, cl::NullRange,
          cl::NDRange{log_2_N, _opts.ocean_tex_size}, cl::NDRange{1, 16});
      _opts.twiddle_factors_init = false;
    } catch (const cl::Error &e) {
      printf("WaveOpenCLLayer::updateSimulation: twiddle indices: OpenCL %s "
             "kernel error: %s\n",
             e.what(), IGetErrorString(e.err()));
      exit(1);
    }
  }

  // change of some ocean's parameters requires to rebuild initial spectrum
  // image
  if (_opts.changed) {
    try {
      float wind_angle_rad = glm::radians(_opts.wind_angle);
      cl_float4 params =
          cl_float4{_opts.wind_magnitude * glm::cos(wind_angle_rad),
                    _opts.wind_magnitude * glm::sin(wind_angle_rad),
                    _opts.amplitude, _opts.supress_factor};
      init_spectrum_kernel.setArg(0, patch);
      init_spectrum_kernel.setArg(1, params);
      init_spectrum_kernel.setArg(2, *noise_mem);
      init_spectrum_kernel.setArg(3, *h0k_mem);

      commandQueue.enqueueNDRangeKernel(
          init_spectrum_kernel, cl::NullRange,
          cl::NDRange{_opts.ocean_tex_size, _opts.ocean_tex_size}, lws);
      _opts.changed = false;
    } catch (const cl::Error &e) {
      printf("WaveOpenCLLayer::updateSimulation: initial spectrum: OpenCL %s "
             "kernel error: %s\n",
             e.what(), IGetErrorString(e.err()));
      exit(1);
    }
  }

  // ping-pong phase spectrum kernel launch
  try {
    time_spectrum_kernel.setArg(0, elapsed);
    time_spectrum_kernel.setArg(1, patch);
    time_spectrum_kernel.setArg(2, *h0k_mem);
    time_spectrum_kernel.setArg(3, *dxyz_coef_mem[0]);
    time_spectrum_kernel.setArg(4, *dxyz_coef_mem[1]);
    time_spectrum_kernel.setArg(5, *dxyz_coef_mem[2]);

    commandQueue.enqueueNDRangeKernel(
        time_spectrum_kernel, cl::NullRange,
        cl::NDRange{_opts.ocean_tex_size, _opts.ocean_tex_size}, lws);
  } catch (const cl::Error &e) {
    printf("WaveOpenCLLayer::updateSimulation: updateSimulation: OpenCL %s "
           "kernel error: %s\n",
           e.what(), IGetErrorString(e.err()));
    exit(1);
  }

  // perform 1D FFT horizontal and vertical iterations
  size_t log_2_N = (size_t)((log((float)_opts.ocean_tex_size) / log(2.f)) - 1);
  fft_kernel.setArg(1, patch);
  fft_kernel.setArg(2, *twiddle_factors_mem);
  for (cl_int i = 0; i < 3; i++) {
    const cl::Image *displ_swap[] = {dxyz_coef_mem[i].get(),
                                     hkt_pong_mem.get()};
    cl_int2 mode = (cl_int2){0, 0};

    bool ifft_pingpong = false;
    for (int p = 0; p < log_2_N; p++) {
      if (ifft_pingpong) {
        fft_kernel.setArg(3, *displ_swap[1]);
        fft_kernel.setArg(4, *displ_swap[0]);
      } else {
        fft_kernel.setArg(3, *displ_swap[0]);
        fft_kernel.setArg(4, *displ_swap[1]);
      }

      mode.s[1] = p;
      fft_kernel.setArg(0, mode);

      commandQueue.enqueueNDRangeKernel(
          fft_kernel, cl::NullRange,
          cl::NDRange{_opts.ocean_tex_size, _opts.ocean_tex_size}, lws);

      ifft_pingpong = !ifft_pingpong;
    }

    // Cols
    mode.s[0] = 1;
    for (int p = 0; p < log_2_N; p++) {
      if (ifft_pingpong) {
        fft_kernel.setArg(3, *displ_swap[1]);
        fft_kernel.setArg(4, *displ_swap[0]);
      } else {
        fft_kernel.setArg(3, *displ_swap[0]);
        fft_kernel.setArg(4, *displ_swap[1]);
      }

      mode.s[1] = p;
      fft_kernel.setArg(0, mode);

      commandQueue.enqueueNDRangeKernel(
          fft_kernel, cl::NullRange,
          cl::NDRange{_opts.ocean_tex_size, _opts.ocean_tex_size}, lws);

      ifft_pingpong = !ifft_pingpong;
    }

    if (log_2_N % 2) {
      // swap images if pingpong hold on temporary buffer
      std::array<size_t, 3> orig = {0, 0, 0},
                            region = {_opts.ocean_tex_size,
                                      _opts.ocean_tex_size, 1};
      commandQueue.enqueueCopyImage(*displ_swap[0], *displ_swap[1], orig, orig,
                                    region);
    }
  }

  if (_opts.useExternalMemory) {
    for (size_t target = 0; target < IOPT_COUNT; target++) {
      commandQueue.enqueueAcquireExternalMemObjects(
          {*mems[target][currentImage]});
    }
  }

  // inversion
  {
    inversion_kernel.setArg(0, patch);
    inversion_kernel.setArg(1, *dxyz_coef_mem[0]);
    inversion_kernel.setArg(2, *dxyz_coef_mem[1]);
    inversion_kernel.setArg(3, *dxyz_coef_mem[2]);
    inversion_kernel.setArg(4, *mems[IOPT_DISPLACEMENT][currentImage]);
    inversion_kernel.setArg(5, *z_ranges_mem[0]);

    commandQueue.enqueueNDRangeKernel(
        inversion_kernel, cl::NullRange,
        cl::NDRange{_opts.ocean_tex_size, _opts.ocean_tex_size}, lws);
  }

  // min max reduction
  {
    cl::NDRange lws = cl::NDRange{_opts.group_size, _opts.group_size};
    cl_int2 patch =
        cl_int2{(int)_opts.ocean_tex_size / 2, (int)_opts.ocean_tex_size / 2};
    for (int p = 0; p < log_2_N; p++) {
      z_ranges_kernel.setArg(0, patch);
      z_ranges_kernel.setArg(1, *z_ranges_mem[p % 2]);
      z_ranges_kernel.setArg(2, *z_ranges_mem[(p + 1) % 2]);

      commandQueue.enqueueNDRangeKernel(
          z_ranges_kernel, cl::NullRange,
          cl::NDRange{(cl::size_type)patch.x, (cl::size_type)patch.y}, lws);

      patch = cl_int2{patch.x / 2, patch.y / 2};
      if (patch.x < lws.get()[0])
        lws = cl::NDRange{(cl::size_type)patch.x, (cl::size_type)patch.y};
    }
    float buf[2] = {0, 0};
    commandQueue.enqueueReadImage(*z_ranges_mem[log_2_N % 2], true,
                                  cl::array<cl::size_type, 2>{0, 0},
                                  cl::array<cl::size_type, 2>{1, 1}, 0, 0, buf);
    z_range = glm::vec2(buf[0], buf[1]);
  }

  // normals computation
  {
    normals_kernel.setArg(0, patch);
    normals_kernel.setArg(1, *mems[IOPT_DISPLACEMENT][currentImage]);
    normals_kernel.setArg(2, *mems[IOPT_NORMAL_MAP][currentImage]);
    normals_kernel.setArg(3, *mems[IOPT_NORMAL_MAP][currentImage]);

    commandQueue.enqueueNDRangeKernel(
        normals_kernel, cl::NullRange,
        cl::NDRange{_opts.ocean_tex_size, _opts.ocean_tex_size}, lws);
  }

  computeFoam(currentImage, patch);

  if (_opts.useExternalMemory) {
    for (size_t target = 0; target < IOPT_COUNT; target++) {
      commandQueue.enqueueReleaseExternalMemObjects(
          {*mems[target][currentImage]});
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void WaveOpenCLLayer::computeFoam(const uint32_t currentImage,
                                  const cl_int2 &patch) {
  cl::NDRange lws = cl::NDRange{_opts.group_size, _opts.group_size};

  // foam computation kernel
  cl_float3 zr =
      cl_float3{z_range.x, z_range.y, _opts.technique == 0 ? 2.f : 8.f};

  foam_kernel.setArg(0, patch);
  foam_kernel.setArg(1, zr);
  foam_kernel.setArg(2, *noise_mem);
  foam_kernel.setArg(3, *mems[IOPT_DISPLACEMENT][currentImage]);
  foam_kernel.setArg(4, *mems[IOPT_NORMAL_MAP][currentImage]);
  foam_kernel.setArg(5, *mems[IOPT_NORMAL_MAP][currentImage]);

  commandQueue.enqueueNDRangeKernel(
      foam_kernel, cl::NullRange,
      cl::NDRange{_opts.ocean_tex_size, _opts.ocean_tex_size}, lws);
}

////////////////////////////////////////////////////////////////////////////////

void WaveOpenCLLayer::updateSolver(uint32_t currentImage) {
  updateUniforms(currentImage);

  auto end = std::chrono::system_clock::now();

  // time factor of ocean animation
  static float elapsed = 0.f;
  static float elapsed_prev = 0.f;

  if (_opts.animate) {
    std::chrono::duration<float> delta = end - start;
    elapsed = delta.count();
    delta_time = elapsed - elapsed_prev;
    elapsed_prev = elapsed;

    updateSimulation(currentImage, elapsed);

    if (_opts.useExternalMemory) {
      commandQueue.finish();
    } else {
      for (size_t target = 0; target < IOPT_COUNT; target++) {
        size_t rowPitch = 0;
        void *pixels = commandQueue.enqueueMapImage(
            *mems[target][currentImage], CL_TRUE, CL_MAP_READ, {0, 0, 0},
            {_opts.ocean_tex_size, _opts.ocean_tex_size, 1}, &rowPitch,
            nullptr);

        VkDeviceSize imageSize =
            _opts.ocean_tex_size * _opts.ocean_tex_size * 4 * sizeof(float);

        void *data;
        vkMapMemory(_vulkan.device, _vulkan.stagingBufferMemory, 0, imageSize,
                    0, &data);
        memcpy(data, pixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(_vulkan.device, _vulkan.stagingBufferMemory);

        commandQueue.enqueueUnmapMemObject(*mems[target][currentImage], pixels);
        commandQueue.flush();

        transitionImageLayout(
            _vulkan.textureImages[target].images[currentImage],
            VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(_vulkan.stagingBuffer,
                          _vulkan.textureImages[target].images[currentImage],
                          static_cast<uint32_t>(_opts.ocean_tex_size),
                          static_cast<uint32_t>(_opts.ocean_tex_size));
        transitionImageLayout(
            _vulkan.textureImages[target].images[currentImage],
            VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      }
    }
  } else {
    // hold the animation at the same time point
    std::chrono::duration<float> duration(elapsed);
    start = end - std::chrono::duration_cast<std::chrono::seconds>(duration);

    if (_opts.useExternalMemory) {
      commandQueue.finish();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

void WaveOpenCLLayer::checkOpenCLExternalMemorySupport(cl::Device &device) {
  if (isExtensionSupported(device(), "cl_khr_external_memory")) {
    printf("cl_khr_external_memory supported.\n");

    std::vector<cl::ExternalMemoryType> types =
        device.getInfo<CL_DEVICE_EXTERNAL_MEMORY_IMPORT_HANDLE_TYPES_KHR>();
    for (auto type : types) {
#define CASE_TO_STRING(_e)                                                     \
  case _e:                                                                     \
    printf("\t%s\n", #_e);                                                     \
    break;
      switch (static_cast<std::underlying_type<cl::ExternalMemoryType>::type>(
          type)) {

        CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR);
        CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR);
        CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KMT_KHR);
        CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_D3D11_TEXTURE_KHR);
        CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_D3D11_TEXTURE_KMT_KHR);
        CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_D3D12_HEAP_KHR);
        CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_D3D12_RESOURCE_KHR);
        CASE_TO_STRING(CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR);
      default:
        printf("Unknown cl_external_memory_handle_type_khr %04X\n",
               (unsigned int)type);
      }
#undef CASE_TO_STRING
    }

#ifdef _WIN32
    if (std::find(types.begin(), types.end(),
                  CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR) != types.end()) {
      externalMemType = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_WIN32_KHR;
    } else {
      printf("Couldn't find a compatible external memory type "
             "(sample supports OPAQUE_WIN32).\n");
      useExternalMemory = false;
    }
#elif defined(__linux__)
    if (std::find(types.begin(), types.end(),
                  cl::ExternalMemoryType(
                      CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR)) != types.end()) {
      externalMemType = CL_EXTERNAL_MEMORY_HANDLE_DMA_BUF_KHR;
    } else if (std::find(types.begin(), types.end(),
                         cl::ExternalMemoryType(
                             CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR)) !=
               types.end()) {
      externalMemType = CL_EXTERNAL_MEMORY_HANDLE_OPAQUE_FD_KHR;
    } else {
      printf("WaveOpenCLLayer::checkOpenCLExternalMemorySupport: Couldn't find "
             "a compatible external memory type "
             "(sample supports DMA_BUF or OPAQUE_FD).\n");
      _opts.useExternalMemory = false;
    }
#endif
  } else {
    printf("WaveOpenCLLayer::checkOpenCLExternalMemorySupport: Device does not "
           "support cl_khr_external_memory.\n");
    _opts.useExternalMemory = false;
  }
}
