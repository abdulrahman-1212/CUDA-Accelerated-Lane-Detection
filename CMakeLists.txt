cmake_minimum_required(VERSION 3.18)
project(lane_detection LANGUAGES CXX CUDA)

# ── C++ / CUDA standards ──────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

# ── Release by default ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── CUDA architecture — targets sm_75 (Turing) and sm_86 (Ampere) by default.
#    Override with: cmake -DCUDA_ARCHS="75;86;89" ..
set(CUDA_ARCHS "75;86" CACHE STRING "CUDA compute capabilities to compile for")
foreach(arch ${CUDA_ARCHS})
  string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

# ── OpenCV ────────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core videoio imgproc)

# ── Sources ───────────────────────────────────────────────────────────────────
add_executable(lane_detection
    main.cpp
    cuda_kernels.cu
)

target_include_directories(lane_detection PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lane_detection PRIVATE ${OpenCV_LIBS} cuda)

# ── Compiler flags ────────────────────────────────────────────────────────────
target_compile_options(lane_detection PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --expt-relaxed-constexpr
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3 -Wall -Wextra
    >
)

# ── Optional: build a benchmark tool ─────────────────────────────────────────
option(BUILD_BENCHMARK "Build frame-throughput benchmark" OFF)
if(BUILD_BENCHMARK)
  add_executable(lane_bench benchmark.cpp cuda_kernels.cu)
  target_include_directories(lane_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(lane_bench PRIVATE ${OpenCV_LIBS} cuda)
endif()
