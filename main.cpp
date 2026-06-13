#include <iostream>
#include <chrono>
#include <string>

#include <opencv2/opencv.hpp>
#include "lane_detector.hpp"

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: lane_detection <input_video> <output_video>\n";
        return 1;
    }

    std::string input_path  = argv[1];
    std::string output_path = argv[2];

    // ── Open input ────────────────────────────────────────────────────────────
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open input video: " << input_path << "\n";
        return 1;
    }

    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total  = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    std::cout << "Input  : " << input_path  << "\n"
              << "Output : " << output_path << "\n"
              << "Size   : " << width << "×" << height << "  FPS: " << fps
              << "  Frames: " << total << "\n";

    // ── Print CUDA device info ────────────────────────────────────────────────
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        std::cerr << "No CUDA-capable GPU found.\n";
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU    : " << prop.name
              << "  SM " << prop.major << "." << prop.minor
              << "  Mem " << prop.totalGlobalMem / (1024*1024) << " MB\n\n";

    // ── Open output ───────────────────────────────────────────────────────────
    cv::VideoWriter writer(output_path,
                           cv::VideoWriter::fourcc('m','p','4','v'),
                           fps,
                           cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Error: cannot open output video: " << output_path << "\n";
        return 1;
    }

    // ── Initialise detector (allocates GPU buffers once) ─────────────────────
    LaneDetector detector(width, height);

    // ── Process frames ────────────────────────────────────────────────────────
    cv::Mat frame, out_frame(height, width, CV_8UC3);
    int frame_idx = 0;

    auto t_start = std::chrono::steady_clock::now();

    while (cap.read(frame)) {
        ++frame_idx;

        // Run GPU pipeline
        auto result = detector.process_frame(frame.data);

        // Wrap result in cv::Mat (no copy) and write
        std::memcpy(out_frame.data, result.data(), result.size());
        writer.write(out_frame);

        // Progress report every 30 frames
        if (frame_idx % 30 == 0) {
            auto now  = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - t_start).count();
            double cur_fps = frame_idx / dt;
            std::cout << "\rFrame " << frame_idx << "/" << total
                      << "  GPU throughput: " << std::fixed
                      << std::setprecision(1) << cur_fps << " fps   " << std::flush;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "\n\nDone. " << frame_idx << " frames in "
              << std::fixed << std::setprecision(2) << elapsed << " s  ("
              << frame_idx / elapsed << " fps average)\n";

    cap.release();
    writer.release();
    return 0;
}
