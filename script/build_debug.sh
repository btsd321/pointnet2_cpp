#!/bin/bash
set -e

# 构建目录
WORKSPACE_DIR="$(dirname $(dirname "$0"))"
BUILD_DIR="$WORKSPACE_DIR/build"

export VCPKG_ROOT="$HOME/SoftWare/vcpkg"
export Torch_DIR=/home/lixinlong/Library/libtorch/install/share/cmake
export CUDA_HOME=/usr/local/cuda-11.8
export CUDACXX="$CUDA_HOME/bin/nvcc"

rm -rf "$BUILD_DIR"

# 打印调试信息
echo "[DEBUG] WORKSPACE_DIR: $WORKSPACE_DIR"
echo "[DEBUG] BUILD_DIR: $BUILD_DIR"
echo "[DEBUG] 当前脚本路径: $0"

# 检查并创建构建目录
if [ ! -d "$BUILD_DIR" ]; then
    echo "[INFO] 创建构建目录: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# 检查CMakeLists.txt是否存在
if [ ! -f "$WORKSPACE_DIR/CMakeLists.txt" ]; then
    echo "[ERROR] $WORKSPACE_DIR/CMakeLists.txt 未找到，终止编译。"
    exit 1
fi

# 配置CMake
echo "[INFO] 配置CMake..."
cmake -S  "$WORKSPACE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install"

# 编译（进入build目录后运行make）
echo "[INFO] 开始编译..."
cd "$BUILD_DIR"
echo "[DEBUG] 当前工作目录: $(pwd)"
make -j$(nproc)
cd -

# 可选：安装
# echo "[INFO] 安装..."
# cmake --install "$BUILD_DIR"

echo "[SUCCESS] Build finished. Executables are in $BUILD_DIR/bin"