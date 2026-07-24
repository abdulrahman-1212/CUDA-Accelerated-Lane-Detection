cmake_minimum_required(VERSION 3.10)
project(CudaLaneDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enable CUDA language support
enable_language(CUDA)

# Find OpenCV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(lane_detection main.cu)

# Link libraries
target_link_libraries(lane_detection ${OpenCV_LIBS})