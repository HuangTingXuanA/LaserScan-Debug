cmake_minimum_required(VERSION 3.26)
project(LaserCalib LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# 1. 调试设置
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -fopenmp -fno-omit-frame-pointer")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0")

# 2. 包含路径
include_directories(${CMAKE_SOURCE_DIR}/include)
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(fmt REQUIRED)

# 3. 源文件
file(GLOB_RECURSE CPP_SRCS src/**.cpp )
file(GLOB_RECURSE CUDA_SRCS src/cuda/*.cu)

# 4. 可执行文件
add_executable(${PROJECT_NAME} ${CPP_SRCS} ${CUDA_SRCS})

# 5. 链接头文件
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 5. 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    TBB::tbb
    cuda
    cudart
    cublas
    cusolver
    curand
    fmt::fmt
)

# 6. CUDA 目标属性
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 86
)