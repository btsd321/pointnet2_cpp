
cmake_minimum_required(VERSION 3.16)
project(pointnet2_cpp VERSION 1.0.0 LANGUAGES CXX)

# CUDA编译选项
option(WITH_CUDA "是否编译CUDA支持" ON)
option(WITH_TEST "是否编译测试程序" ON)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_DEBUG_POSTFIX "_d")

# 如果有vcpkg，使用vcpkg工具链
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
    message(STATUS "Vcpkg ROOT: $ENV{VCPKG_ROOT}")
    message(STATUS "Using vcpkg toolchain from VCPKG_ROOT")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    include("${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    message(STATUS "Using local vcpkg toolchain")
endif()

# 设置编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -DDEBUG")
    message(STATUS "Debug模式: 启用调试信息，禁用优化")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -DNDEBUG")
    message(STATUS "Release模式: 启用最高优化，禁用调试信息")
else()
    # 默认Release模式
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -DNDEBUG")
    message(STATUS "默认Release模式: 启用最高优化")
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# 查找依赖包
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "Start finding dependencies...")
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
find_package(VTK REQUIRED)
find_package(PCL CONFIG REQUIRED COMPONENTS common io kdtree search visualization features)
find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)
if(WITH_CUDA)
    find_package(CUDA QUIET)
endif()
if(WITH_TEST)
    find_package(OpenCV REQUIRED)
    find_package(yaml-cpp CONFIG REQUIRED)
    find_package(RapidJSON CONFIG REQUIRED)
endif()

# 设置LibTorch路径
set(LIBTORCH_ROOT "/home/lixinlong/Library/libtorch/install")
if(EXISTS "${LIBTORCH_ROOT}/share/cmake/Torch")
    set(CMAKE_PREFIX_PATH "${LIBTORCH_ROOT};${CMAKE_PREFIX_PATH}")
    message(STATUS "Using LibTorch from: ${LIBTORCH_ROOT}")
else()
    message(FATAL_ERROR "LibTorch not found at ${LIBTORCH_ROOT}. Please set LIBTORCH_ROOT to the correct path.")
endif()

# 设置编译标志
message(STATUS "Found PCL: ${PCL_VERSION}")
message(STATUS "Found Eigen3: ${Eigen3_VERSION}")
message(STATUS "Found Python3: ${Python3_VERSION}")
message(STATUS "Found VTK: ${VTK_VERSION}")
message(STATUS "Found OpenMP: ${OpenMP_FOUND}")
message(STATUS "Found LibTorch: ${Torch_VERSION}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


if(WITH_CUDA AND CUDA_FOUND)
    message(STATUS "Found CUDA: ${CUDA_VERSION}")
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
elseif(WITH_CUDA)
    message(FATAL_ERROR "WITH_CUDA=ON 但未找到CUDA")
endif()
if(WITH_TEST)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(STATUS "WITH_TEST=OFF，跳过OpenCV查找")
endif()

message(STATUS "LibTorch found: ${Torch_VERSION}")
message(STATUS "PCL found: ${PCL_VERSION}")
if(WITH_CUDA AND CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
endif()
if(WITH_TEST)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(STATUS "WITH_TEST=OFF，跳过OpenCV查找")
endif()

if(WITH_TEST)
# argparse是header-only库，直接包含vcpkg安装的路径
if(EXISTS "~/SoftWare/vcpkg/installed/x64-linux/include/argparse")
    set(ARGPARSE_INCLUDE_DIR "~/SoftWare/vcpkg/installed/x64-linux/include")
    message(STATUS "Found argparse in vcpkg")
endif()
endif()

# 收集pointnet2相关源文件
if(WITH_CUDA AND CUDA_FOUND)
    file(GLOB POINTNET2_SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/pointnet2/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/pointnet2/cuda/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/pointnet2/cuda/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/pointnet2/pt/*.cpp"
    )
else()
    file(GLOB POINTNET2_SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/src/pointnet2/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/pointnet2/pt/*.cpp"
    )
endif()
# 收集pointnet2相关头文件
file(GLOB POINTNET2_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/pointnet2/*.h")
# 收集测试源文件
file(GLOB TEST_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/test/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test/src/common/*.cpp"
)

# 收集
set(PROJECT_SOURCES
    "${POINTNET2_SOURCES}"
)

set(PROJECT_HEADERS
    "${POINTNET2_HEADERS}"
)

# # 创建静态库
# add_library(pointnet2 STATIC ${PROJECT_SOURCES})

# # 基础链接库
# target_link_libraries(pointnet2 Eigen3::Eigen)

# # 链接必需库
# target_link_libraries(pointnet2 ${PCL_LIBRARIES})
# target_link_libraries(pointnet2 ${TORCH_LIBRARIES})
# target_link_libraries(pointnet2 ${Python3_LIBRARIES})
# target_link_libraries(pointnet2 ${VTK_LIBRARIES})
# target_link_libraries(pointnet2 ${OpenMP_CXX_LIBRARIES})
# if(WITH_CUDA AND CUDA_FOUND)
#     target_link_libraries(pointnet2 ${CUDA_LIBRARIES})
# endif()

# # 设置库的包含目录
# target_include_directories(pointnet2 PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     ${EIGEN3_INCLUDE_DIR}
#     ${PCL_INCLUDE_DIRS}
#     ${Python3_INCLUDE_DIRS}
#     ${VTK_INCLUDE_DIRS}
#     ${OpenMP_CXX_INCLUDE_DIRS}
# )

# # 安装规则
# install(TARGETS pointnet2
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
#     RUNTIME DESTINATION bin
# )

# install(FILES ${PROJECT_HEADERS}
#     DESTINATION include
# )

if(WITH_TEST)
# 创建测试可执行文件
add_executable(pointnet2_test 
    "${PROJECT_SOURCES}"
    "${TEST_SOURCES}"
)

# 设置测试项目的包含目录
target_include_directories(pointnet2_test PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test/include
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${ARGPARSE_INCLUDE_DIR}
    $<TARGET_PROPERTY:yaml-cpp,INTERFACE_INCLUDE_DIRECTORIES>
    ${RapidJSON_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${OpenMP_CXX_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(pointnet2_test 
    ${PCL_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    yaml-cpp 
    ${Python3_LIBRARIES}
    ${VTK_LIBRARIES}
    ${OpenMP_CXX_LIBRARIES}
)

if(WITH_CUDA AND CUDA_FOUND)
    target_link_libraries(pointnet2_test ${CUDA_LIBRARIES})
endif()

install(TARGETS pointnet2_test
    DESTINATION bin
)

# 添加测试
enable_testing()
 
endif()


# 打印配置信息
message(STATUS "")
message(STATUS "=================== DSNet C++ Inference Configuration ===================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Eigen3 Version: ${Eigen3_VERSION}")
message(STATUS "LibTorch Version: ${Torch_VERSION}")
message(STATUS "PCL Version: ${PCL_VERSION}")
if(WITH_CUDA AND CUDA_FOUND)
    message(STATUS "CUDA Enabled: Yes")
    message(STATUS "CUDA Version: ${CUDA_VERSION}")
    message(STATUS "Argparse Include: ${ARGPARSE_INCLUDE_DIR}")
else()
    message(STATUS "CUDA Enabled: No")
endif()
if(WITH_TEST)
    message(STATUS "WITH_TEST: Yes")
    message(STATUS "yaml-cpp Version: ${yaml-cpp_VERSION}")
    message(STATUS "RapidJSON Version: ${RapidJSON_VERSION}")
    message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
else()
    message(STATUS "WITH_TEST: No")
    message(STATUS "OpenCV Enabled: No")
endif()
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================================================")
message(STATUS "")