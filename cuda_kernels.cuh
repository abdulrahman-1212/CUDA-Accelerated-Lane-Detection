#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// ─── Grayscale ────────────────────────────────────────────────────────────────
__global__ void bgr2gray_kernel(const uint8_t* __restrict__ src,
                                 uint8_t* __restrict__ dst,
                                 int width, int height);

// ─── Gaussian Blur (5×5, σ≈1) ─────────────────────────────────────────────────
__global__ void gaussian_blur_kernel(const uint8_t* __restrict__ src,
                                      uint8_t* __restrict__ dst,
                                      int width, int height);

// ─── Canny (Sobel + NMS + double threshold + hysteresis) ──────────────────────
__global__ void sobel_kernel(const uint8_t* __restrict__ src,
                              float* __restrict__ mag,
                              int8_t* __restrict__ dir,
                              int width, int height);

__global__ void nms_kernel(const float* __restrict__ mag,
                            const int8_t* __restrict__ dir,
                            float* __restrict__ out,
                            int width, int height);

__global__ void threshold_kernel(const float* __restrict__ nms,
                                  uint8_t* __restrict__ edges,
                                  int width, int height,
                                  float low_t, float high_t);

// ─── Region mask ──────────────────────────────────────────────────────────────
__global__ void region_mask_kernel(uint8_t* __restrict__ edges,
                                    int width, int height);

// ─── Draw lane lines onto frame ───────────────────────────────────────────────
__global__ void draw_lines_kernel(uint8_t* __restrict__ frame,
                                   int width, int height,
                                   int x1l, int y1l, int x2l, int y2l,
                                   int x1r, int y1r, int x2r, int y2r,
                                   uint8_t r, uint8_t g, uint8_t b,
                                   int thickness);
