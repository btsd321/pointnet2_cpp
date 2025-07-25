// 插值/邻域相关
#pragma once

#include <vector>

#include <torch/extension.h>

#ifdef USE_CUDA
#include "pointnet2/cuda_utils.h"
namespace pointnet2::utils::cuda
{

/*
 * @brief 三近邻查找(Three Nearest Neighbors)
 *
 * 对于每个unknowns中的点, 查找knows中距离最近的3个点。
 * 常用于点云插值和特征传播阶段。
 *
 * @param unknowns (Tensor) 需要插值的点坐标, 形状为[B, n, 3]
 * @param knows    (Tensor) 已知点的坐标, 形状为[B, m, 3]
 * @return         (vector<Tensor>) 返回两个Tensor：
 *                   - idx: 最近3个点的索引, 形状为[B, n, 3]
 *                   - dist: 最近3个点的距离, 形状为[B, n, 3]
 */
std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows);

/*
 * @brief 三线性插值(Three Interpolate)
 *
 * 根据给定的索引idx和权重weight, 对输入特征points进行三线性插值, 得到目标点的特征。
 *
 * @param points (Tensor) 已知点的特征, 形状为[B, C, m]
 * @param idx    (Tensor) 三近邻的索引, 形状为[B, n, 3]
 * @param weight (Tensor) 三近邻的插值权重, 形状为[B, n, 3]
 * @return       (Tensor) 插值后的特征, 形状为[B, C, n]
 */
at::Tensor three_interpolate(at::Tensor points, at::Tensor idx, at::Tensor weight);

/*
 * @brief 三线性插值的反向传播(Three Interpolate Grad)
 *
 * 计算三线性插值操作的梯度, 将上游梯度grad_out根据索引和权重累加回原始特征points。
 *
 * @param grad_out (Tensor) 上游梯度, 形状为[B, C, n]
 * @param idx      (Tensor) 三近邻的索引, 形状为[B, n, 3]
 * @param weight   (Tensor) 三近邻的插值权重, 形状为[B, n, 3]
 * @param m        (int)    已知点的数量m
 * @return         (Tensor) points的梯度, 形状为[B, C, m]
 */
at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx, at::Tensor weight,
                                  const int m);

}  // namespace pointnet2::utils::cuda
#endif  // USE_CUDA

namespace pointnet2::utils::pt
{

/*
 * @brief 三近邻查找(Three Nearest Neighbors)
 *
 * 对于每个unknowns中的点, 查找knows中距离最近的3个点。
 * 常用于点云插值和特征传播阶段。
 *
 * @param unknowns (Tensor) 需要插值的点坐标, 形状为[B, n, 3]
 * @param knows    (Tensor) 已知点的坐标, 形状为[B, m, 3]
 * @return         (vector<Tensor>) 返回两个Tensor：
 *                   - idx: 最近3个点的索引, 形状为[B, n, 3]
 *                   - dist: 最近3个点的距离, 形状为[B, n, 3]
 */
std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows);

/*
 * @brief 三线性插值(Three Interpolate)
 *
 * 根据给定的索引idx和权重weight, 对输入特征points进行三线性插值, 得到目标点的特征。
 *
 * @param points (Tensor) 已知点的特征, 形状为[B, C, m]
 * @param idx    (Tensor) 三近邻的索引, 形状为[B, n, 3]
 * @param weight (Tensor) 三近邻的插值权重, 形状为[B, n, 3]
 * @return       (Tensor) 插值后的特征, 形状为[B, C, n]
 */
at::Tensor three_interpolate(at::Tensor points, at::Tensor idx, at::Tensor weight);

/*
 * @brief 三线性插值的反向传播(Three Interpolate Grad)
 *
 * 计算三线性插值操作的梯度, 将上游梯度grad_out根据索引和权重累加回原始特征points。
 *
 * @param grad_out (Tensor) 上游梯度, 形状为[B, C, n]
 * @param idx      (Tensor) 三近邻的索引, 形状为[B, n, 3]
 * @param weight   (Tensor) 三近邻的插值权重, 形状为[B, n, 3]
 * @param m        (int)    已知点的数量m
 * @return         (Tensor) points的梯度, 形状为[B, C, m]
 */
at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx, at::Tensor weight,
                                  const int m);

}  // namespace pointnet2::utils::pt
