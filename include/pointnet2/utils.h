// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <vector>

#include <torch/extension.h>

#ifdef USE_CUDA
#include "pointnet2/cuda_utils.h"
#include <ATen/cuda/CUDAContext.h>
#endif

#include <torch/extension.h>
#include "pointnet2/utils_group.h"          // 分组/采样相关
#include "pointnet2/utils_interpolation.h"  // 插值/邻域相关

// PyTorch版本兼容性检查
#if defined(TORCH_VERSION_MAJOR) && defined(TORCH_VERSION_MINOR)
#if TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 5)
#define AT_CHECK TORCH_CHECK
#endif
#else
// 如果无法检测到版本，尝试使用TORCH_CHECK
#ifndef AT_CHECK
#define AT_CHECK TORCH_CHECK
#endif
#endif

#ifdef USE_CUDA
/*
 * @brief 检查输入张量是否为CUDA张量的宏
 *
 * 用于断言x必须是CUDA类型的张量, 否则报错。
 * 常用于自定义CUDA算子的输入检查。
 *
 * @param x (Tensor) 需要检查的输入张量
 */
#define CHECK_CUDA(x)                                                \
    do                                                               \
    {                                                                \
        AT_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor"); \
    } while (0)
#endif

/*
 * @brief 检查输入张量是否为连续内存的宏
 *
 * 用于断言x必须是内存连续(contiguous)的张量, 否则报错。
 * 常用于CUDA算子输入, 保证数据排布正确。
 *
 * @param x (Tensor) 需要检查的输入张量
 */
#define CHECK_CONTIGUOUS(x)                                             \
    do                                                                  \
    {                                                                   \
        AT_CHECK(x.is_contiguous(), #x " must be a contiguous tensor"); \
    } while (0)

/*
 * @brief 检查输入张量是否为int类型的宏
 *
 * 用于断言x必须是int类型的张量, 否则报错。
 * 常用于索引等需要整型张量的CUDA算子输入检查。
 *
 * @param x (Tensor) 需要检查的输入张量
 */
#define CHECK_IS_INT(x)                                                                \
    do                                                                                 \
    {                                                                                  \
        AT_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor"); \
    } while (0)

/*
 * @brief 检查输入张量是否为float类型的宏
 *
 * 用于断言x必须是float类型的张量, 否则报错。
 * 常用于特征、坐标等需要浮点型张量的CUDA算子输入检查。
 *
 * @param x (Tensor) 需要检查的输入张量
 */
#define CHECK_IS_FLOAT(x)                                                                 \
    do                                                                                    \
    {                                                                                     \
        AT_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be a float tensor"); \
    } while (0)

/*
 * @brief PyTorch版本兼容性宏定义
 *
 * 用于处理不同PyTorch版本之间的API差异
 */

#ifdef USE_CUDA
// 检查张量是否在CUDA设备上的兼容性宏
#if defined(TORCH_VERSION_MAJOR) && defined(TORCH_VERSION_MINOR)
#if TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 5)
#define IS_CUDA_TENSOR(x) (x.device().is_cuda())
#else
#define IS_CUDA_TENSOR(x) (x.type().is_cuda())
#endif
#else
// 默认使用新版本API，如果编译失败则需要手动切换
#define IS_CUDA_TENSOR(x) (x.device().is_cuda())
#endif
#endif

// 获取张量数据指针的兼容性宏
#if defined(TORCH_VERSION_MAJOR) && defined(TORCH_VERSION_MINOR)
#if TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 5)
#define TENSOR_DATA_PTR(x, type) (x.data_ptr<type>())
#else
#define TENSOR_DATA_PTR(x, type) (x.data<type>())
#endif
#else
// 默认使用新版本API，如果编译失败则需要手动切换
#define TENSOR_DATA_PTR(x, type) (x.data_ptr<type>())
#endif
