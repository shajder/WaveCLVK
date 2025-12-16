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
#ifndef _WAVE_COMPUTE_FOAM_LAYER_HPP_
#define _WAVE_COMPUTE_FOAM_LAYER_HPP_

#include "wave_util.hpp"
#include "wave_compute_layer.hpp"

class WaveOpenCLFoamLayer : public WaveOpenCLLayer {

public:

    WaveOpenCLFoamLayer(SharedOptions & opts) : WaveOpenCLLayer(opts) {}

    void setupFoamSolver(const std::string &) override;

    void computeFoam(const uint32_t currentImage, const cl_int2 & patch) override;
    \
    void updateSimulation(const uint32_t currentImage, const float elapsed) override;

    void cleanup() override {
        // nothing to do - release handled by wrappers
    }

    void initCompute() override;

    void initComputeResources() override;

protected:

    void updateAdvection(float dt, float dumping, cl::Image2D & velocity, cl::Image2D ** fields );

    std::int16_t getNextFromEventsCache();

    std::vector<cl::Event> * getAddr(const std::int16_t id);

protected:

    cl::Context contextFoam;
    cl::CommandQueue commandQueueFoam;

    // Navier-Stokes fluid resources
    cl::Kernel copy_kernel;
    cl::Kernel advect_kernel;
    cl::Kernel div_kernel;
    cl::Kernel jacobi_kernel;
    cl::Kernel pressure_kernel;
    cl::Kernel max_ranges_kernel;

    std::array<std::unique_ptr<cl::Image2D>, 2> fld_cont;

    cl::Image2D* flds[2];

    std::int16_t cache_counter=0;
    std::vector<std::vector<cl::Event>>  wait_evs_cache;
    std::int16_t final_events=-1;

    std::unique_ptr<cl::Image2D> divRBTexture;
    std::unique_ptr<cl::Image2D> pressureRBTexture[2];

    std::unique_ptr<cl::Image2D> max_ranges_mem[2];

    cl_float mcRevert=0.05f;
    cl_int FREAD = 0, FWRITE = 1;

    bool initialize_foam=false;

};

#endif //_WAVE_COMPUTE_FOAM_LAYER_HPP_
