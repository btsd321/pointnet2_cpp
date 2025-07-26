#!/bin/bash
set -e

# 构建目录
WORKSPACE_DIR="$(dirname $(dirname "$0"))"
BUILD_DIR="$WORKSPACE_DIR/build/debug"

export VCPKG_ROOT="$HOME/SoftWare/vcpkg"
export Torch_DIR=/home/lixinlong/Library/libtorch/install/share/cmake
export CUDA_HOME=/usr/local/cuda-11.8
export CUDACXX="$CUDA_HOME/bin/nvcc"

echo "[INFO] 当前脚本路径: $0"
echo "[INFO] WORKSPACE_DIR: $WORKSPACE_DIR"
echo "[INFO] BUILD_DIR: $BUILD_DIR"

# 检查并创建构建目录
if [ ! -d "$BUILD_DIR" ]; then
    echo "[INFO] 创建构建目录: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

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