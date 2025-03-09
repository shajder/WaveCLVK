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
#ifndef _WAVE_COMPUTE_LAYER_HPP_
#define _WAVE_COMPUTE_LAYER_HPP_

#include "wave_util.hpp"
#include "wave_render_layer.hpp"

class WaveOpenCLLayer : public WaveVulkanLayer {

public:

    WaveOpenCLLayer(SharedOptions & opts) : WaveVulkanLayer(opts) {}

    virtual void setupFoamSolver(const std::string &);

    virtual void computeFoam(const uint32_t currentImage, const cl_int2 & patch);
    \
    virtual void updateSimulation(const uint32_t currentImage, const float elapsed);

    virtual void cleanup() override;

    virtual void initCompute() override;

    virtual void initComputeResources() override;

    void updateSolver(uint32_t currentImage) override;

protected:

    // OpenCL resources
    cl_external_memory_handle_type_khr externalMemType = 0;

    cl::Context context;
    cl::Device  cl_device;
    cl::CommandQueue commandQueue;

    // generates twiddle factors kernel
    cl::Kernel twiddle_kernel;

    // initial spectrum kernel
    cl::Kernel init_spectrum_kernel;

    // Fourier components image kernel
    cl::Kernel time_spectrum_kernel;

    // FFT kernel
    cl::Kernel fft_kernel;

    // inversion kernel
    cl::Kernel inversion_kernel;

    // building normals kernel
    cl::Kernel normals_kernel;

    // min/max reduction kernel
    cl::Kernel z_ranges_kernel;

    // min/max reduction kernel
    cl::Kernel foam_kernel;

    // FFT intermediate computation storages without vulkan iteroperability
    std::unique_ptr<cl::Image2D> dxyz_coef_mem[3];
    std::unique_ptr<cl::Image2D> hkt_pong_mem;
    std::unique_ptr<cl::Image2D> twiddle_factors_mem;
    std::unique_ptr<cl::Image2D> h0k_mem;
    std::unique_ptr<cl::Image2D> noise_mem;
    std::unique_ptr<cl::Image2D> z_ranges_mem[2];

    size_t ocl_max_img2d_width=0;
    cl_ulong ocl_max_alloc_size=0, ocl_mem_size=0;

    // opencl-vulkan iteroperability resources
    // final computation result with displacements and normal map,
    // needs to follow swap-chain scheme
    std::array<std::vector<std::unique_ptr<cl::Image2D>>, IOPT_COUNT> mems;
    std::vector<cl::Semaphore> signalSemaphores;

    float delta_time=0.f;

public:

    // select vulkan device associated with selected OpenCL platform
    void pickPhysicalDevice() override;

    bool useExternalMemoryType() override;

protected:

    void checkOpenCLExternalMemorySupport(cl::Device& device);
};

#endif //_WAVE_COMPUTE_LAYER_HPP_
