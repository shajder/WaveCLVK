cmake_minimum_required(VERSION 3.10)
project(WaveCLVK)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Vulkan REQUIRED)
find_package(OpenCL REQUIRED)
find_package(GLFW REQUIRED)
find_package(glm REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

add_subdirectory(external/OpenCL-Headers)
add_subdirectory(external/OpenCL-ICD-Loader)
add_subdirectory(external/OpenCL-CLHPP)

set(SOURCE_FILES
    main.cpp
    wave_render_layer.cpp
    wave_render_layer.hpp
    wave_compute_layer.cpp
    wave_compute_layer.hpp
    wave_foam_compute_layer.cpp
    wave_foam_compute_layer.hpp
    wave_app.cpp
    wave_app.hpp
    wave_util.hpp
    )
set(OPENCL_KERNELS
    kernels/twiddle.cl
    kernels/time_spectrum.cl
    kernels/inversion.cl
    kernels/normals.cl
    kernels/fft_kernel.cl
    kernels/init_spectrum_phillips.cl
    kernels/init_spectrum_jonswap.cl
    kernels/reduce_ranges.cl
    kernels/foam.cl
    kernels/foam_cfd.cl
    kernels/advect.cl
    kernels/divergence.cl
    kernels/jacobi.cl
    kernels/pressure.cl
    kernels/reduce_foam.cl
    kernels/copy.cl
)

set(Vulkan_SHADERS
    shaders/ocean.vert.spv
    shaders/ocean.frag.spv
)

foreach(KERNEL ${OPENCL_KERNELS})
    configure_file(${KERNEL} ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL} COPYONLY)
endforeach()

foreach(SHADER ${Vulkan_SHADERS})
    configure_file(${SHADER} ${CMAKE_CURRENT_BINARY_DIR}/${SHADER} COPYONLY)
endforeach()

if(NOT OPENCL_SAMPLE_VERSION)
    message(STATUS "No OpenCL version specified for sample ${OPENCL_SAMPLE_TARGET}, using OpenCL 3.0.")
    set(OPENCL_SAMPLE_VERSION 300)
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    Vulkan::Vulkan
    OpenCL::OpenCL
    glfw
    Boost::program_options
)

target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    CL_TARGET_OPENCL_VERSION=${OPENCL_SAMPLE_VERSION}
    CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_SAMPLE_VERSION}
    CL_HPP_MINIMUM_OPENCL_VERSION=${OPENCL_SAMPLE_VERSION}
    CL_HPP_ENABLE_EXCEPTIONS
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    external/OpenCL-CLHPP/include
    external/OpenCL-Headers
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GLFW_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${OPENCL_INCLUDE_DIRS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
