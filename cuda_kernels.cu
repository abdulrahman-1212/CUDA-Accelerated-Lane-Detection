#include "cuda_kernels.cuh"
#include <math.h>

// ─── Helpers ──────────────────────────────────────────────────────────────────
static __device__ __forceinline__ int clamp_idx(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ─── BGR → Grayscale ──────────────────────────────────────────────────────────
__global__ void bgr2gray_kernel(const uint8_t* __restrict__ src,
                                 uint8_t* __restrict__ dst,
                                 int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    float b = src[idx + 0];
    float g = src[idx + 1];
    float r = src[idx + 2];
    dst[y * width + x] = (uint8_t)(0.114f * b + 0.587f * g + 0.299f * r);
}

// ─── 5×5 Gaussian blur ────────────────────────────────────────────────────────
// Kernel: [1,4,6,4,1]/16 separable — done here as a single 2D pass for clarity.
static __constant__ float gauss5[5][5] = {
    {1,  4,  6,  4, 1},
    {4, 16, 24, 16, 4},
    {6, 24, 36, 24, 6},
    {4, 16, 24, 16, 4},
    {1,  4,  6,  4, 1}
};
static __constant__ float gauss5_sum = 256.0f;

__global__ void gaussian_blur_kernel(const uint8_t* __restrict__ src,
                                      uint8_t* __restrict__ dst,
                                      int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float acc = 0.0f;
    for (int ky = -2; ky <= 2; ++ky)
        for (int kx = -2; kx <= 2; ++kx) {
            int nx = clamp_idx(x + kx, 0, width  - 1);
            int ny = clamp_idx(y + ky, 0, height - 1);
            acc += gauss5[ky + 2][kx + 2] * src[ny * width + nx];
        }
    dst[y * width + x] = (uint8_t)(acc / gauss5_sum);
}

// ─── Sobel (gradient magnitude + quantised direction) ─────────────────────────
// Direction codes: 0=0°, 1=45°, 2=90°, 3=135°
__global__ void sobel_kernel(const uint8_t* __restrict__ src,
                              float* __restrict__ mag,
                              int8_t* __restrict__ dir,
                              int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    auto px = [&](int dx, int dy) -> float {
        return src[clamp_idx(y+dy,0,height-1) * width + clamp_idx(x+dx,0,width-1)];
    };

    float gx = -px(-1,-1) - 2*px(0,-1) - px(1,-1)
               +px(-1, 1) + 2*px(0, 1) + px(1, 1);
    float gy = -px(-1,-1) - 2*px(-1,0) - px(-1,1)
               +px( 1,-1) + 2*px( 1,0) + px( 1,1);

    float m = sqrtf(gx*gx + gy*gy);
    float angle = atan2f(fabsf(gy), fabsf(gx)) * 180.0f / M_PI;

    int8_t d;
    if      (angle < 22.5f)  d = 0;
    else if (angle < 67.5f)  d = 1;
    else if (angle < 112.5f) d = 2;
    else if (angle < 157.5f) d = 3;
    else                     d = 0;

    int idx = y * width + x;
    mag[idx] = m;
    dir[idx] = d;
}

// ─── Non-Maximum Suppression ───────────────────────────────────────────────────
__global__ void nms_kernel(const float* __restrict__ mag,
                            const int8_t* __restrict__ dir,
                            float* __restrict__ out,
                            int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float m = mag[idx];
    int8_t d = dir[idx];

    int dx1=0, dy1=0, dx2=0, dy2=0;
    if      (d == 0) { dx1= 1; dy1= 0; dx2=-1; dy2= 0; }
    else if (d == 1) { dx1= 1; dy1= 1; dx2=-1; dy2=-1; }
    else if (d == 2) { dx1= 0; dy1= 1; dx2= 0; dy2=-1; }
    else             { dx1=-1; dy1= 1; dx2= 1; dy2=-1; }

    int nx1 = clamp_idx(x+dx1,0,width-1), ny1 = clamp_idx(y+dy1,0,height-1);
    int nx2 = clamp_idx(x+dx2,0,width-1), ny2 = clamp_idx(y+dy2,0,height-1);

    float n1 = mag[ny1*width+nx1];
    float n2 = mag[ny2*width+nx2];

    out[idx] = (m >= n1 && m >= n2) ? m : 0.0f;
}

// ─── Double-threshold + simple strong-edge mask ───────────────────────────────
__global__ void threshold_kernel(const float* __restrict__ nms,
                                  uint8_t* __restrict__ edges,
                                  int width, int height,
                                  float low_t, float high_t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float v = nms[idx];
    if      (v >= high_t) edges[idx] = 255;
    else if (v >= low_t ) edges[idx] = 128;  // weak — kept simple (no hysteresis walk on GPU)
    else                  edges[idx] = 0;
}

// ─── Region of interest mask (trapezoid, same proportions as Python) ──────────
__global__ void region_mask_kernel(uint8_t* __restrict__ edges,
                                    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Trapezoid vertices (same as Python code)
    float bl_x = width  * 0.1f, bl_y = height * 0.95f;
    float tl_x = width  * 0.4f, tl_y = height * 0.6f;
    float tr_x = width  * 0.6f, tr_y = height * 0.6f;
    float br_x = width  * 0.9f, br_y = height * 0.95f;

    // Point-in-trapezoid test via four edge half-planes
    auto side = [](float px, float py, float ax, float ay,
                    float bx, float by) -> float {
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    };

    float fx = (float)x, fy = (float)y;
    bool inside =
        side(fx,fy, bl_x,bl_y, tl_x,tl_y) >= 0 &&
        side(fx,fy, tl_x,tl_y, tr_x,tr_y) >= 0 &&
        side(fx,fy, tr_x,tr_y, br_x,br_y) >= 0 &&
        side(fx,fy, br_x,br_y, bl_x,bl_y) >= 0;

    if (!inside) edges[y * width + x] = 0;
}

// ─── Draw a thick line onto the BGR frame ─────────────────────────────────────
__global__ void draw_lines_kernel(uint8_t* __restrict__ frame,
                                   int width, int height,
                                   int x1l, int y1l, int x2l, int y2l,
                                   int x1r, int y1r, int x2r, int y2r,
                                   uint8_t r, uint8_t g, uint8_t b,
                                   int thickness)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Distance from point (x,y) to segment (x1,y1)-(x2,y2)
    auto dist_seg = [](float px, float py,
                        float ax, float ay,
                        float bx, float by) -> float {
        float dx = bx - ax, dy = by - ay;
        float len2 = dx*dx + dy*dy;
        if (len2 < 1e-6f) {
            float ex = px-ax, ey = py-ay;
            return sqrtf(ex*ex+ey*ey);
        }
        float t = ((px-ax)*dx + (py-ay)*dy) / len2;
        t = fmaxf(0.f, fminf(1.f, t));
        float qx = ax + t*dx - px;
        float qy = ay + t*dy - py;
        return sqrtf(qx*qx + qy*qy);
    };

    float half = thickness * 0.5f;
    float fx = (float)x, fy = (float)y;

    bool on_left  = dist_seg(fx,fy, x1l,y1l, x2l,y2l) <= half;
    bool on_right = dist_seg(fx,fy, x1r,y1r, x2r,y2r) <= half;

    if (on_left || on_right) {
        int base = (y * width + x) * 3;
        frame[base + 0] = b;
        frame[base + 1] = g;
        frame[base + 2] = r;
    }
}
