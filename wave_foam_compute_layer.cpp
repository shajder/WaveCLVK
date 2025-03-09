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
#include "wave_foam_compute_layer.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <set>

void WaveOpenCLFoamLayer::initCompute()
{
    WaveOpenCLLayer::initCompute();

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
           platforms[_opts.plat_index]
               .getInfo<CL_PLATFORM_NAME>()
               .c_str());
    std::vector<cl::Device> devices;
    platforms[_opts.plat_index].getDevices(CL_DEVICE_TYPE_ALL,
                                                      &devices);

    printf("Running on device: %s\n",
           devices[_opts.dev_index]
               .getInfo<CL_DEVICE_NAME>()
               .c_str());

    // recreate command queue with out-of-order property to parallelize IFFT and CFD computations
    commandQueue = cl::CommandQueue{ context, cl_device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE };

    auto build_opencl_kernel = [&](const char* src_file, cl::Kernel& kernel,
                                   const char* name) {
        try
        {
            std::string kernel_code = readFile(src_file).data();
            cl::Program program{ context, kernel_code };
            program.build();
            kernel = cl::Kernel{ program, name };
        } catch (const cl::BuildError& e)
        {
            auto bl = e.getBuildLog();
            std::cout << "Build OpenCL " << name << " kernel error: " << std::endl;
            for (auto & elem : bl)
                std::cout << elem.second << std::endl;
            exit(1);
        }
    };

    build_opencl_kernel("kernels/copy.cl", copy_kernel, "copy");
    build_opencl_kernel("kernels/advect.cl", advect_kernel, "advect");
    build_opencl_kernel("kernels/divergence.cl", div_kernel, "divergence");
    build_opencl_kernel("kernels/jacobi.cl", jacobi_kernel, "jacobi");
    build_opencl_kernel("kernels/pressure.cl", pressure_kernel, "pressure");
    build_opencl_kernel("kernels/reduce_foam.cl", max_ranges_kernel, "reduce");
}

void WaveOpenCLFoamLayer::initComputeResources()
{
    WaveOpenCLLayer::initComputeResources();

    size_t gwx = _opts.ocean_tex_size * _opts.foam_scope_mult;
    size_t gwy = _opts.ocean_tex_size * _opts.foam_scope_mult;

    for ( int i=0; i<fld_cont.size(); i++)
    {
        fld_cont[i] = std::make_unique<cl::Image2D>(
                    context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, CL_FLOAT),
                    gwx, gwy);

        flds[i] = fld_cont[i].get();
    }

    divRBTexture = std::make_unique<cl::Image2D>(
                context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT),
                gwx, gwy);

    pressureRBTexture[0] = std::make_unique<cl::Image2D>(
                context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT),
                gwx, gwy);

    pressureRBTexture[1] = std::make_unique<cl::Image2D>(
                context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT),
                gwx, gwy);

    max_ranges_mem[0] = std::make_unique<cl::Image2D>(
        context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT), gwx,
        gwy);

    max_ranges_mem[1] = std::make_unique<cl::Image2D>(
        context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_RG, CL_FLOAT),
        gwx / 2, gwy / 2);
}

std::int16_t WaveOpenCLFoamLayer::getNextFromEventsCache()
{
    std::int16_t id=cache_counter;
    cache_counter++;
    if(id>=wait_evs_cache.size())
    {
        wait_evs_cache.push_back(std::vector<cl::Event>());
    }
    wait_evs_cache[id].resize(1, cl::Event());
    return id;
}

std::vector<cl::Event> * WaveOpenCLFoamLayer::getAddr(const std::int16_t id)
{
    if(static_cast<unsigned>(id)>=wait_evs_cache.size())
    {
        return nullptr;
    }
    return &wait_evs_cache[id];
}


void WaveOpenCLFoamLayer::updateSimulation(uint32_t currentImage, float elapsed)
{
    cl_int2 patch = cl_int2{(int)(_opts.ocean_grid_size * _opts.mesh_spacing), (int)_opts.ocean_tex_size};
    assert (_opts.group_size > 0);
    cl::NDRange lws {16, 16}; // NullRange by default.
    if (_opts.group_size > 0)
    {
        lws = cl::NDRange{ _opts.group_size, _opts.group_size };
    }

    if (_opts.twiddle_factors_init)
    {
        try
        {
            size_t log_2_N = (size_t)((log((float)_opts.ocean_tex_size) / log(2.f))-1);

            /// Prepare vector of values to extract results
            std::vector<cl_int> v(_opts.ocean_tex_size);
            for (int i = 0; i < _opts.ocean_tex_size; i++)
            {
                int x = reverse_bits(i, log_2_N);
                v[i] = x;
            }

            /// Initialize device-side storage
            cl::Buffer bit_reversed_inds_mem{
                context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof(cl_int) * v.size(), v.data()
            };

            twiddle_kernel.setArg(0, cl_int(_opts.ocean_tex_size));
            twiddle_kernel.setArg(1, bit_reversed_inds_mem);
            twiddle_kernel.setArg(2, *twiddle_factors_mem);

            auto evs = getAddr(getNextFromEventsCache());
            commandQueue.enqueueNDRangeKernel(
                twiddle_kernel, cl::NullRange,
                cl::NDRange{log_2_N, _opts.ocean_tex_size}, cl::NDRange{1, 16},
                nullptr, &evs->front());
            _opts.twiddle_factors_init = false;
            cl::Event::waitForEvents(*evs);
            cache_counter=0;
        } catch (const cl::Error &e) {
          printf("twiddle indices: OpenCL %s kernel error: %s\n", e.what(),
                 IGetErrorString(e.err()));
          exit(1);
        }
    }


    // change of some ocean's parameters requires to rebuild initial spectrum image
    if (_opts.changed)
    {
        try
        {
            float wind_angle_rad = glm::radians(_opts.wind_angle);
            cl_float4 params = cl_float4 {
                    _opts.wind_magnitude * glm::cos(wind_angle_rad),
                    _opts.wind_magnitude * glm::sin(wind_angle_rad),
                    _opts.amplitude, _opts.supress_factor
            };
            init_spectrum_kernel.setArg(0, patch);
            init_spectrum_kernel.setArg(1, params);
            init_spectrum_kernel.setArg(2, *noise_mem);
            init_spectrum_kernel.setArg(3, *h0k_mem);

            auto evs = getAddr(getNextFromEventsCache());
            commandQueue.enqueueNDRangeKernel(
                init_spectrum_kernel, cl::NullRange,
                cl::NDRange{_opts.ocean_tex_size, _opts.ocean_tex_size}, lws,
                nullptr, &evs->front());
            cl::Event::waitForEvents(*evs);
            cache_counter=0;

            _opts.changed = false;
        } catch (const cl::Error& e)
        {
            printf("initial spectrum: OpenCL %s kernel error: %s\n", e.what(), IGetErrorString(e.err()));
            exit(1);
        }
    }

    std::int16_t swp_evts[2] = {-1, getNextFromEventsCache()};

    // ping-pong phase spectrum kernel launch
    try
    {
        time_spectrum_kernel.setArg(0, elapsed);
        time_spectrum_kernel.setArg(1, patch);
        time_spectrum_kernel.setArg(2, *h0k_mem);
        time_spectrum_kernel.setArg(3, *dxyz_coef_mem[0]);
        time_spectrum_kernel.setArg(4, *dxyz_coef_mem[1]);
        time_spectrum_kernel.setArg(5, *dxyz_coef_mem[2]);

        commandQueue.enqueueNDRangeKernel(
            time_spectrum_kernel, cl::NullRange,
            cl::NDRange{_opts.ocean_tex_size, _opts.ocean_tex_size}, lws,
            getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());

        std::swap(swp_evts[0], swp_evts[1]);
        swp_evts[1] = getNextFromEventsCache();
    } catch (const cl::Error &e) {
      printf("updateSimulation: OpenCL %s kernel error: %s\n", e.what(),
             IGetErrorString(e.err()));
      exit(1);
    }

    // perform 1D FFT horizontal and vertical iterations
    size_t log_2_N = (size_t)((log((float)_opts.ocean_tex_size) / log(2.f))-1);
    fft_kernel.setArg(1, patch);
    fft_kernel.setArg(2, *twiddle_factors_mem);
    for ( cl_int i=0; i<3; i++)
    {
        const cl::Image * displ_swap[] = {dxyz_coef_mem[i].get(), hkt_pong_mem.get()};
        cl_int2 mode = (cl_int2){0, 0};

        bool ifft_pingpong=false;
        for (int p = 0; p < log_2_N; p++)
        {
            if (ifft_pingpong)
            {
                fft_kernel.setArg(3, *displ_swap[1]);
                fft_kernel.setArg(4, *displ_swap[0]);
            }
            else
            {
                fft_kernel.setArg(3, *displ_swap[0]);
                fft_kernel.setArg(4, *displ_swap[1]);
            }

            mode.s[1] = p;
            fft_kernel.setArg(0, mode);

            commandQueue.enqueueNDRangeKernel(
                fft_kernel, cl::NullRange,
                cl::NDRange{ _opts.ocean_tex_size, _opts.ocean_tex_size }, lws,
                        getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());

            ifft_pingpong = !ifft_pingpong;
            std::swap(swp_evts[0], swp_evts[1]);
            swp_evts[1] = getNextFromEventsCache();
        }

        // Cols
        mode.s[0] = 1;
        for (int p = 0; p < log_2_N; p++)
        {
            if (ifft_pingpong)
            {
                fft_kernel.setArg(3, *displ_swap[1]);
                fft_kernel.setArg(4, *displ_swap[0]);
            }
            else
            {
                fft_kernel.setArg(3, *displ_swap[0]);
                fft_kernel.setArg(4, *displ_swap[1]);
            }

            mode.s[1] = p;
            fft_kernel.setArg(0, mode);

            commandQueue.enqueueNDRangeKernel(
                fft_kernel, cl::NullRange,
                cl::NDRange{_opts.ocean_tex_size, _opts.ocean_tex_size}, lws,
                getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());

            ifft_pingpong = !ifft_pingpong;
            std::swap(swp_evts[0], swp_evts[1]);
            swp_evts[1] = getNextFromEventsCache();
        }

        if (log_2_N%2)
        {
            // swap images if pingpong hold on temporary buffer
            std::array<size_t, 3> orig = {0,0,0}, region={_opts.ocean_tex_size, _opts.ocean_tex_size, 1};
            commandQueue.enqueueCopyImage(*displ_swap[0], *displ_swap[1], orig, orig, region,
                    getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());
            std::swap(swp_evts[0], swp_evts[1]);
            swp_evts[1] = getNextFromEventsCache();
        }
    }

    if (_opts.useExternalMemory)
    {
        for (size_t target=0; target<IOPT_COUNT; target++)
        {
          commandQueue.enqueueAcquireExternalMemObjects(
              {*mems[target][currentImage]}, getAddr(swp_evts[0]),
              &getAddr(swp_evts[1])->front());
          std::swap(swp_evts[0], swp_evts[1]);
          swp_evts[1] = getNextFromEventsCache();
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
            cl::NDRange{_opts.ocean_tex_size, _opts.ocean_tex_size}, lws,
            getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());
        std::swap(swp_evts[0], swp_evts[1]);
        swp_evts[1] = getNextFromEventsCache();
    }

    // min max reduction
    {
        cl::NDRange lws = cl::NDRange{ _opts.group_size, _opts.group_size };
        cl_int2 patch = cl_int2{ (int)_opts.ocean_tex_size/2, (int)_opts.ocean_tex_size/2 };
        for (int p = 0; p < log_2_N; p++)
        {
            z_ranges_kernel.setArg(0, patch);
            z_ranges_kernel.setArg(1, *z_ranges_mem[p%2]);
            z_ranges_kernel.setArg(2, *z_ranges_mem[(p+1)%2]);

            commandQueue.enqueueNDRangeKernel(
                z_ranges_kernel, cl::NullRange,
                cl::NDRange{ (cl::size_type)patch.x, (cl::size_type)patch.y }, lws,
                        getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());

            patch = cl_int2{ patch.x/2, patch.y/2 };
            if (patch.x<lws.get()[0])
                lws = cl::NDRange{ (cl::size_type)patch.x, (cl::size_type)patch.y };

            std::swap(swp_evts[0], swp_evts[1]);
            swp_evts[1] = getNextFromEventsCache();
        }
        float buf[2] = {0,0};
        commandQueue.enqueueReadImage(
            *z_ranges_mem[log_2_N%2], true, cl::array<cl::size_type, 2>{ 0, 0 },
            cl::array<cl::size_type, 2>{ 1, 1 }, 0, 0, buf,
                    getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());
        z_range = glm::vec2(buf[0], buf[1]);
        std::swap(swp_evts[0], swp_evts[1]);
        swp_evts[1] = getNextFromEventsCache();
    }

    // normals computation
    {
        normals_kernel.setArg(0, patch);
        normals_kernel.setArg(1, *mems[IOPT_DISPLACEMENT][currentImage]);
        normals_kernel.setArg(2, *mems[IOPT_NORMAL_MAP][currentImage]);
        normals_kernel.setArg(3, *mems[IOPT_NORMAL_MAP][currentImage]);

        commandQueue.enqueueNDRangeKernel(
            normals_kernel, cl::NullRange,
            cl::NDRange{_opts.ocean_tex_size, _opts.ocean_tex_size}, lws,
            getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());

        final_events = swp_evts[1];
    }

    computeFoam(currentImage, patch);

    if (_opts.useExternalMemory)
    {
        for (size_t target=0; target<IOPT_COUNT; target++)
        {
            commandQueue.enqueueReleaseExternalMemObjects(
                { *mems[target][currentImage] });
        }
    }
}

void WaveOpenCLFoamLayer::setupFoamSolver(const std::string & name)
{
    WaveOpenCLLayer::setupFoamSolver("kernels/foam_cfd.cl");
}

void WaveOpenCLFoamLayer::computeFoam(const uint32_t currentImage, const cl_int2 & patch)
{
    // synchronize main computation path with foam solver
    cl::NDRange lws {16, 16}; // NullRange by default.
    if (_opts.group_size > 0)
    {
        lws = cl::NDRange{ _opts.group_size, _opts.group_size };
    }

    size_t gwx = _opts.ocean_tex_size * _opts.foam_scope_mult;
    size_t gwy = _opts.ocean_tex_size * _opts.foam_scope_mult;

    std::int16_t swp_evts[2] = {-1, getNextFromEventsCache()};

    if (!initialize_foam)
    {
        initialize_foam=true;

        auto evs = getAddr(swp_evts[1]);
        // clear simulation buffers
        std::array<cl::size_type, 2> origin = { 0, 0 };
        std::array<cl::size_type, 2> region = { gwx, gwy };
        for ( int i=0; i<fld_cont.size(); i++)
        {
            commandQueue.enqueueFillImage(
                *fld_cont[i], cl_float4{ { 0.f, 0.f, 0.f, 0.f } }, origin,
                region, nullptr, &evs->back());
            evs->push_back(cl::Event());
        }

        commandQueue.enqueueFillImage(*divRBTexture,
                                      cl_float4{ { 0.f, 0.f, 0.f, 0.f } },
                                      origin, region, nullptr, &evs->back());
        evs->push_back(cl::Event());
        commandQueue.enqueueFillImage(*pressureRBTexture[0],
                                      cl_float4{ { 0.f, 0.f, 0.f, 0.f } },
                                      origin, region, nullptr, &evs->back());
                                      evs->push_back(cl::Event());
        commandQueue.enqueueFillImage(*pressureRBTexture[1],
                                      cl_float4{ { 0.f, 0.f, 0.f, 0.f } },
                                      origin, region, nullptr, &evs->back());
        cl::Event::waitForEvents(*evs);
    }




    float dt=delta_time;
    // min max reduction
    {
        // first copy velocities to reduction buffer
        copy_kernel.setArg(0, *flds[FREAD]);
        copy_kernel.setArg(1, *max_ranges_mem[0]);
        commandQueue.enqueueNDRangeKernel(copy_kernel, cl::NullRange,
                                          cl::NDRange{ gwx, gwy }, lws,
                                          getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());
        std::swap(swp_evts[0], swp_evts[1]);
        swp_evts[1] = getNextFromEventsCache();

        size_t log_2_N = (size_t) (log(gwx)/log(2.f));
        cl::NDRange rlws = cl::NDRange{ lws[0], lws[1] };
        cl_int2 patch = cl_int2{ (int)gwx/2, (int)gwx/2 };
        for (int p = 0; p < log_2_N; p++)
        {
            max_ranges_kernel.setArg(0, patch);
            max_ranges_kernel.setArg(1, *max_ranges_mem[p%2]);
            max_ranges_kernel.setArg(2, *max_ranges_mem[(p+1)%2]);

            commandQueue.enqueueNDRangeKernel(
                max_ranges_kernel, cl::NullRange,
                cl::NDRange{ (cl::size_type)patch.x, (cl::size_type)patch.y }, rlws,
                        getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());

            patch = cl_int2{ patch.x/2, patch.y/2 };
            if (patch.x<rlws.get()[0])
                rlws = cl::NDRange{ (cl::size_type)patch.x, (cl::size_type)patch.y };

            std::swap(swp_evts[0], swp_evts[1]);
            swp_evts[1] = getNextFromEventsCache();
        }
        float buf[2] = {0,0};
        commandQueue.enqueueReadImage(
            *max_ranges_mem[log_2_N%2], true, cl::array<cl::size_type, 2>{ 0, 0 },
            cl::array<cl::size_type, 2>{ 1, 1 }, 0, 0, buf,
                    getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());

        std::swap(swp_evts[0], swp_evts[1]);
        swp_evts[1] = getNextFromEventsCache();

        float vMax = std::max(buf[0], buf[1]);

        if (vMax>0.f&&!std::isnan(vMax))
            dt = std::min(dt, dt * 16.f / vMax);
    }

    // Advection phase
    {
        advect_kernel.setArg(0, (cl_float4){ (float)gwx, (float)gwy, dt, 0.5f }); // damping
        advect_kernel.setArg(1, *flds[FREAD]);
        advect_kernel.setArg(2, *flds[0]); // field read
        advect_kernel.setArg(3, *flds[1]); // field wright
        commandQueue.enqueueNDRangeKernel(advect_kernel, cl::NullRange,
                                          cl::NDRange{gwx, gwy}, lws,
                                          getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());

        std::swap(swp_evts[0], swp_evts[1]);
        swp_evts[1] = getNextFromEventsCache();

        std::swap(flds[0], flds[1]);
    }

    cl_float4 info = cl_float4 { (float)gwx, (float)gwy, dt, mcRevert };

    // Jacobi phase
    {
        div_kernel.setArg(0, info);
        div_kernel.setArg(1, *flds[FREAD]);
        div_kernel.setArg(2, *divRBTexture);
        commandQueue.enqueueNDRangeKernel(div_kernel, cl::NullRange,
                                          cl::NDRange{ gwx, gwy }, lws,
                                          getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());

        std::swap(swp_evts[0], swp_evts[1]);
        swp_evts[1] = getNextFromEventsCache();
    }

    cl_int PREAD = 0, PWRITE = 1;
    cl_int jacobiIterations=20;
    jacobi_kernel.setArg(0, info);
    jacobi_kernel.setArg(1, *divRBTexture);
    for(unsigned i = 0; i < jacobiIterations; ++i)
    {
        jacobi_kernel.setArg(2, *pressureRBTexture[PREAD]);
        jacobi_kernel.setArg(3, *pressureRBTexture[PWRITE]);
        commandQueue.enqueueNDRangeKernel(jacobi_kernel, cl::NullRange,
                                          cl::NDRange{ gwx, gwy }, lws,
                                          getAddr(swp_evts[0]), &getAddr(swp_evts[1])->front());
        std::swap(PREAD, PWRITE);
        std::swap(swp_evts[0], swp_evts[1]);
        swp_evts[1] = getNextFromEventsCache();
    }

    {
        auto fevs = getAddr(final_events);
        if (!fevs)
            fevs = getAddr(swp_evts[1]);
        fevs->push_back(cl::Event());
        pressure_kernel.setArg(0, info);
        pressure_kernel.setArg(1, *pressureRBTexture[PREAD]);
        pressure_kernel.setArg(2, *flds[FREAD]);
        pressure_kernel.setArg(3, *flds[FWRITE]);
        commandQueue.enqueueNDRangeKernel(pressure_kernel, cl::NullRange,
                                          cl::NDRange{ gwx, gwy }, lws,
                                          getAddr(swp_evts[0]), &fevs->back());
        std::swap(flds[FREAD], flds[FWRITE]);
    }


    cache_counter=0;


    float wind_angle_rad = glm::radians(_opts.wind_angle);

    cl_float8 zr = cl_float8{z_range.x,
                             z_range.y,
                             _opts.technique == 0 ? 2.f : 8.f,
                             _opts.wind_magnitude * glm::cos(wind_angle_rad),
                             _opts.wind_magnitude * glm::sin(wind_angle_rad),
                             delta_time,
                             100.f,
                             (float)_opts.foam_scope_mult};


#if 1
    {
        auto fevs = getAddr(final_events);
        foam_kernel.setArg(0, patch);
        foam_kernel.setArg(1, zr);
        foam_kernel.setArg(2, *noise_mem);
        foam_kernel.setArg(3, *mems[IOPT_DISPLACEMENT][currentImage]);
        foam_kernel.setArg(4, *flds[FREAD]);
        foam_kernel.setArg(5, *mems[IOPT_NORMAL_MAP][currentImage]);
        foam_kernel.setArg(6, *flds[FREAD]);
        foam_kernel.setArg(7, *mems[IOPT_NORMAL_MAP][currentImage]);

        commandQueue.enqueueNDRangeKernel(
            foam_kernel, cl::NullRange,
            cl::NDRange{ _opts.ocean_tex_size, _opts.ocean_tex_size }, lws, fevs, nullptr);
    }
#else

    foam_kernel.setArg(0, patch);
    foam_kernel.setArg(1, zr);
    foam_kernel.setArg(2, *noise_mem);
    foam_kernel.setArg(3, *mems[IOPT_DISPLACEMENT][currentImage]);
    foam_kernel.setArg(4, *flds[FREAD]);
    foam_kernel.setArg(5, *mems[IOPT_NORMAL_MAP][currentImage]);
    foam_kernel.setArg(6, *flds[FWRITE]);
    foam_kernel.setArg(7, *mems[IOPT_NORMAL_MAP][currentImage]);

    commandQueue.enqueueNDRangeKernel(
        foam_kernel, cl::NullRange,
        cl::NDRange{ _opts.ocean_tex_size, _opts.ocean_tex_size }, lws);

    std::swap(flds[FREAD], flds[FWRITE]);
#endif
}

