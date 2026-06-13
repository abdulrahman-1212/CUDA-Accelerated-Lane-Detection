#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

#include "cuda_kernels.cuh"
#include "hough_cpu.hpp"

// ─── CUDA error-check macro ───────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess)                                              \
            throw std::runtime_error(std::string("CUDA error at ")          \
                + __FILE__ + ":" + std::to_string(__LINE__)                  \
                + " — " + cudaGetErrorString(err));                         \
    } while (0)

class LaneDetector {
public:
    // Allocate device buffers for a fixed frame size.
    LaneDetector(int width, int height)
        : width_(width), height_(height), pixels_(width * height)
    {
        size_t rgb_sz  = pixels_ * 3;
        size_t gray_sz = pixels_;
        size_t flt_sz  = pixels_ * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_bgr_,   rgb_sz));
        CUDA_CHECK(cudaMalloc(&d_gray_,  gray_sz));
        CUDA_CHECK(cudaMalloc(&d_blur_,  gray_sz));
        CUDA_CHECK(cudaMalloc(&d_mag_,   flt_sz));
        CUDA_CHECK(cudaMalloc(&d_nms_,   flt_sz));
        CUDA_CHECK(cudaMalloc(&d_dir_,   pixels_ * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_edges_, gray_sz));
        CUDA_CHECK(cudaMalloc(&d_frame_out_, rgb_sz));

        // Pinned host memory for fast DMA transfers
        CUDA_CHECK(cudaMallocHost(&h_edges_, gray_sz));
        CUDA_CHECK(cudaMallocHost(&h_frame_out_, rgb_sz));
    }

    ~LaneDetector() {
        cudaFree(d_bgr_); cudaFree(d_gray_); cudaFree(d_blur_);
        cudaFree(d_mag_); cudaFree(d_nms_);  cudaFree(d_dir_);
        cudaFree(d_edges_); cudaFree(d_frame_out_);
        cudaFreeHost(h_edges_); cudaFreeHost(h_frame_out_);
    }

    // Process one BGR frame (rows × cols × 3, row-major).
    // Returns a BGR image with lane lines drawn on it.
    std::vector<uint8_t> process_frame(const uint8_t* bgr_in)
    {
        dim3 block(16, 16);
        dim3 grid((width_  + block.x - 1) / block.x,
                  (height_ + block.y - 1) / block.y);

        // ── Upload ──────────────────────────────────────────────────────────
        CUDA_CHECK(cudaMemcpy(d_bgr_, bgr_in, pixels_ * 3, cudaMemcpyHostToDevice));

        // ── BGR → Gray ──────────────────────────────────────────────────────
        bgr2gray_kernel<<<grid, block>>>(d_bgr_, d_gray_, width_, height_);

        // ── Gaussian blur ───────────────────────────────────────────────────
        gaussian_blur_kernel<<<grid, block>>>(d_gray_, d_blur_, width_, height_);

        // ── Sobel ───────────────────────────────────────────────────────────
        sobel_kernel<<<grid, block>>>(d_blur_, d_mag_, d_dir_, width_, height_);

        // ── NMS ─────────────────────────────────────────────────────────────
        nms_kernel<<<grid, block>>>(d_mag_, d_dir_, d_nms_, width_, height_);

        // ── Double threshold (low=50, high=150) ──────────────────────────────
        threshold_kernel<<<grid, block>>>(d_nms_, d_edges_, width_, height_, 50.f, 150.f);

        // ── Region mask ─────────────────────────────────────────────────────
        region_mask_kernel<<<grid, block>>>(d_edges_, width_, height_);

        // ── Download edge map for CPU Hough ─────────────────────────────────
        CUDA_CHECK(cudaMemcpy(h_edges_, d_edges_, pixels_, cudaMemcpyDeviceToHost));

        // ── Hough Transform on CPU ───────────────────────────────────────────
        auto hough_lines = hough_transform_cpu(h_edges_, width_, height_);
        auto [left_lane, right_lane] = lane_lines_from_hough(hough_lines, height_);

        // ── Copy original frame as output base ───────────────────────────────
        CUDA_CHECK(cudaMemcpy(d_frame_out_, d_bgr_, pixels_ * 3, cudaMemcpyDeviceToDevice));

        // ── Draw lanes on GPU ────────────────────────────────────────────────
        int x1l=0,y1l=0,x2l=0,y2l=0;
        int x1r=0,y1r=0,x2r=0,y2r=0;
        if (left_lane.valid)  { x1l=left_lane.x1;  y1l=left_lane.y1;  x2l=left_lane.x2;  y2l=left_lane.y2; }
        if (right_lane.valid) { x1r=right_lane.x1; y1r=right_lane.y1; x2r=right_lane.x2; y2r=right_lane.y2; }

        draw_lines_kernel<<<grid, block>>>(
            d_frame_out_, width_, height_,
            x1l, y1l, x2l, y2l,
            x1r, y1r, x2r, y2r,
            255, 0, 0,   // Red in RGB → B=0,G=0,R=255 (stored BGR)
            12);

        // ── Download result ──────────────────────────────────────────────────
        CUDA_CHECK(cudaMemcpy(h_frame_out_, d_frame_out_, pixels_ * 3, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        return std::vector<uint8_t>(h_frame_out_, h_frame_out_ + pixels_ * 3);
    }

private:
    int    width_, height_, pixels_;

    // Device buffers
    uint8_t* d_bgr_       = nullptr;
    uint8_t* d_gray_      = nullptr;
    uint8_t* d_blur_      = nullptr;
    float*   d_mag_       = nullptr;
    float*   d_nms_       = nullptr;
    int8_t*  d_dir_       = nullptr;
    uint8_t* d_edges_     = nullptr;
    uint8_t* d_frame_out_ = nullptr;

    // Pinned host buffers
    uint8_t* h_edges_     = nullptr;
    uint8_t* h_frame_out_ = nullptr;
};
