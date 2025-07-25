// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#ifdef USE_CUDA

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

// CUDA核函数的最大线程数
#define TOTAL_THREADS 512

// 根据任务规模work_size, 自动选择最优线程数(为2的幂且不超过TOTAL_THREADS)
// 返回值：最优线程数
inline int opt_n_threads(int work_size)
{
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

    return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

// 根据二维任务规模x和y, 自动选择最优的block配置(线程块大小)
// 返回值：dim3类型的block配置(x_threads, y_threads, 1)
inline dim3 opt_block_config(int x, int y)
{
    const int x_threads = opt_n_threads(x);
    const int y_threads = max(min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
    dim3 block_config(x_threads, y_threads, 1);

    return block_config;
}

// CUDA错误检查宏：每次调用后检查是否有CUDA错误, 若有则输出错误信息并退出程序
#define CUDA_CHECK_ERRORS()                                                            \
    do                                                                                 \
    {                                                                                  \
        cudaError_t err = cudaGetLastError();                                          \
        if (cudaSuccess != err)                                                        \
        {                                                                              \
            fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",             \
                    cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, __FILE__); \
            exit(-1);                                                                  \
        }                                                                              \
    } while (0)

#endif  // USE_CUDA
