// Synthetic benchmark — measures pure GPU pipeline throughput without disk I/O.
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include "lane_detector.hpp"

int main(int argc, char** argv)
{
    int width  = argc > 1 ? std::atoi(argv[1]) : 1280;
    int height = argc > 2 ? std::atoi(argv[2]) : 720;
    int iters  = argc > 3 ? std::atoi(argv[3]) : 500;

    std::cout << "Benchmark: " << width << "×" << height
              << "  iterations: " << iters << "\n";

    // Synthetic noisy road frame
    cv::Mat fake_frame(height, width, CV_8UC3);
    cv::randu(fake_frame, cv::Scalar(50, 80, 50), cv::Scalar(200, 200, 200));

    LaneDetector detector(width, height);

    // Warm-up
    for (int i = 0; i < 10; ++i)
        detector.process_frame(fake_frame.data);
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
        detector.process_frame(fake_frame.data);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << std::fixed << std::setprecision(2)
              << "Total : " << ms       << " ms\n"
              << "Per   : " << ms/iters << " ms/frame\n"
              << "FPS   : " << iters / (ms / 1000.0) << "\n";
    return 0;
}
