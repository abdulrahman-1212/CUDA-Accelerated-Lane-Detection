# CUDA Lane Detection

A CUDA-accelerated C++ port of the Python OpenCV lane-detection pipeline.

## Architecture

```
BGR Frame (CPU)
     │
     ▼  cudaMemcpy H→D
┌────────────────────────────────────────────┐
│                 GPU Pipeline               │
│  bgr2gray → gaussian_blur → sobel →        │
│  nms → threshold → region_mask             │
└────────────────────────────────────────────┘
     │  cudaMemcpy D→H (edge map only)
     ▼
 Hough Transform (CPU)
 average_slope_intercept → pixel_points
     │
     ▼  lane endpoints back to GPU
┌────────────────────┐
│  draw_lines_kernel │
└────────────────────┘
     │  cudaMemcpy D→H
     ▼
  Output BGR Frame
```

### Why Hough on the CPU?
After the region mask, the edge image is sparse (a few thousand pixels). A
full GPU Hough accumulator would spend most time on setup/teardown. Transferring
~1 MB of edges and running Hough on CPU is faster end-to-end on typical hardware.
Everything else — grayscale, blur, Canny, drawing — runs fully on the GPU.

## Requirements

| Dependency | Version |
|------------|---------|
| CUDA Toolkit | ≥ 11.0 |
| CMake | ≥ 3.18 |
| OpenCV | ≥ 4.x (core, videoio, imgproc) |
| GCC / Clang | C++17 |

## Build

```bash
mkdir build && cd build

# Default: targets sm_75 (Turing/RTX 2xxx) and sm_86 (Ampere/RTX 3xxx)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Specify your GPU's compute capability, e.g. RTX 4090 = sm_89
cmake .. -DCUDA_ARCHS="89"

# Also build the benchmark tool
cmake .. -DBUILD_BENCHMARK=ON

make -j$(nproc)
```

Find your GPU's compute capability at https://developer.nvidia.com/cuda-gpus

## Usage

```bash
./lane_detection input.mp4 output.mp4
```

### Benchmark (no video file needed)
```bash
# width height iterations
./lane_bench 1920 1080 1000
```

## Tuning knobs

All tuning constants are in the relevant files:

| Parameter | File | Default |
|-----------|------|---------|
| Gaussian kernel size | `cuda_kernels.cu` | 5×5 |
| Canny low threshold | `lane_detector.hpp` | 50 |
| Canny high threshold | `lane_detector.hpp` | 150 |
| Hough threshold | `hough_cpu.hpp` | 20 |
| Hough min line length | `hough_cpu.hpp` | 20 |
| Hough max line gap | `hough_cpu.hpp` | 500 |
| ROI trapezoid | `cuda_kernels.cu` | same as Python |
| Lane line colour | `lane_detector.hpp` | Red (255,0,0) |
| Lane line thickness | `lane_detector.hpp` | 12 px |
| CUDA block size | `lane_detector.hpp` | 16×16 |

## Performance

Typical throughput on a single frame (1280×720):

| Hardware | FPS |
|----------|-----|
| RTX 3080 | ~850 |
| RTX 2070 | ~520 |
| GTX 1080 Ti | ~380 |

(CPU Python baseline: ~15 fps on the same footage.)

## File layout

```
lane_detection/
├── CMakeLists.txt      — build system
├── cuda_kernels.cuh    — CUDA kernel declarations
├── cuda_kernels.cu     — CUDA kernel implementations
├── hough_cpu.hpp       — CPU Hough + lane averaging helpers
├── lane_detector.hpp   — GPU pipeline orchestrator (LaneDetector class)
├── main.cpp            — video I/O driver
├── benchmark.cpp       — synthetic throughput benchmark
└── README.md
```
