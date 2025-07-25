// 分组/采样相关
#pragma once

#include <vector>

#include <torch/extension.h>

#ifdef USE_CUDA
#include "pointnet2/cuda_utils.h"
namespace pointnet2::utils::cuda
{
/*
 * @brief 点云分组操作(Group Points)
 *
 * 根据给定的索引idx, 从输入点云特征points中采样, 返回分组后的特征。
 * 常用于PointNet++等点云网络的局部特征聚合阶段, 实现对每个采样中心点的邻域特征提取。
 *
 * @param points (Tensor) 输入点云特征, 形状为[B, C, N], B为batch, C为通道数, N为点数
 * @param idx    (Tensor) 分组采样的索引, 形状为[B, npoint, nsample], npoint为采样中心点数,
 * nsample为每个中心的邻域点数
 * @return       (Tensor) 分组后的特征, 形状为[B, C, npoint, nsample]
 */
at::Tensor group_points(at::Tensor points, at::Tensor idx);

/*
 * @brief 点云分组操作的反向传播(Group Points Grad)
 *
 * 计算group_points操作的梯度, 将上游梯度grad_out根据采样索引idx累加回原始输入特征points的位置。
 * 用于训练时反向传播, 确保梯度正确传递到原始点云特征。
 *
 * @param grad_out (Tensor) 上游梯度, 形状为[B, C, npoint, nsample]
 * @param idx      (Tensor) 分组采样的索引, 形状为[B, npoint, nsample]
 * @param n        (int)    原始点的数量N
 * @return         (Tensor) 输入特征points的梯度, 形状为[B, C, N]
 */
at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);

/*
 * @brief 球查询(Ball Query)操作, 用于点云处理中的邻域搜索。
 *
 * 该函数用于在点云xyz中, 以new_xyz为中心, 查找半径radius内的邻域点, 最多返回nsample个点的索引。
 * 常用于PointNet++等点云网络的特征提取阶段。
 *
 * @param new_xyz   (Tensor) 查询中心点的坐标, 形状为[B, npoint, 3]
 * @param xyz       (Tensor) 原始点云的坐标, 形状为[B, N, 3]
 * @param radius    (float)  球查询的半径
 * @param nsample   (int)    每个球内最多采样的点数
 * @return          (Tensor) 返回每个中心点对应的邻域点索引, 形状为[B, npoint, nsample]
 */
at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius, const int nsample);

/*
 * @brief 点云特征采样(Gather Points)
 *
 * 根据给定的索引idx, 从输入点云特征points中采样, 返回采样后的特征。
 * 常用于根据采样点索引提取对应的特征。
 *
 * @param points (Tensor) 输入点云特征, 形状为[B, C, N]
 * @param idx    (Tensor) 采样点的索引, 形状为[B, npoint]
 * @return       (Tensor) 采样后的特征, 形状为[B, C, npoint]
 */
at::Tensor gather_points(at::Tensor points, at::Tensor idx);

/*
 * @brief 点云特征采样的反向传播(Gather Points Grad)
 *
 * 计算gather_points操作的梯度, 将上游梯度grad_out根据采样索引idx累加回原始输入特征points的位置。
 * 用于训练时反向传播, 确保梯度正确传递到原始点云特征。
 *
 * @param grad_out (Tensor) 上游梯度, 形状为[B, C, npoint]
 * @param idx      (Tensor) 采样点的索引, 形状为[B, npoint]
 * @param n        (int)    原始点的数量N
 * @return         (Tensor) 输入特征points的梯度, 形状为[B, C, N]
 */
at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);

/*
 * @brief 最远点采样(Furthest Point Sampling, FPS)
 *
 * 在输入点云points中, 按照最远点策略采样出nsamples个点的索引。
 * 常用于点云下采样, 保证采样点分布均匀, 覆盖整个点云空间。
 *
 * @param points   (Tensor) 输入点云坐标, 形状为[B, N, 3]
 * @param nsamples (int)    需要采样的点数
 * @return         (Tensor) 采样点的索引, 形状为[B, nsamples]
 */
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples);

}  // namespace pointnet2::utils::cuda
#endif  // USE_CUDA

namespace pointnet2::utils::pt
{
/*
 * @brief 点云分组操作(Group Points)
 *
 * 根据给定的索引idx, 从输入点云特征points中采样, 返回分组后的特征。
 * 常用于PointNet++等点云网络的局部特征聚合阶段, 实现对每个采样中心点的邻域特征提取。
 *
 * @param points (Tensor) 输入点云特征, 形状为[B, C, N], B为batch, C为通道数, N为点数
 * @param idx    (Tensor) 分组采样的索引, 形状为[B, npoint, nsample], npoint为采样中心点数,
 * nsample为每个中心的邻域点数
 * @return       (Tensor) 分组后的特征, 形状为[B, C, npoint, nsample]
 */
at::Tensor group_points(at::Tensor points, at::Tensor idx);

/*
 * @brief 点云分组操作的反向传播(Group Points Grad)
 *
 * 计算group_points操作的梯度, 将上游梯度grad_out根据采样索引idx累加回原始输入特征points的位置。
 * 用于训练时反向传播, 确保梯度正确传递到原始点云特征。
 *
 * @param grad_out (Tensor) 上游梯度, 形状为[B, C, npoint, nsample]
 * @param idx      (Tensor) 分组采样的索引, 形状为[B, npoint, nsample]
 * @param n        (int)    原始点的数量N
 * @return         (Tensor) 输入特征points的梯度, 形状为[B, C, N]
 */
at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);

/*
 * @brief 球查询(Ball Query)操作, 用于点云处理中的邻域搜索。
 *
 * 该函数用于在点云xyz中, 以new_xyz为中心, 查找半径radius内的邻域点, 最多返回nsample个点的索引。
 * 常用于PointNet++等点云网络的特征提取阶段。
 *
 * @param new_xyz   (Tensor) 查询中心点的坐标, 形状为[B, npoint, 3]
 * @param xyz       (Tensor) 原始点云的坐标, 形状为[B, N, 3]
 * @param radius    (float)  球查询的半径
 * @param nsample   (int)    每个球内最多采样的点数
 * @return          (Tensor) 返回每个中心点对应的邻域点索引, 形状为[B, npoint, nsample]
 */
at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius, const int nsample);

/*
 * @brief 点云特征采样(Gather Points)
 *
 * 根据给定的索引idx, 从输入点云特征points中采样, 返回采样后的特征。
 * 常用于根据采样点索引提取对应的特征。
 *
 * @param points (Tensor) 输入点云特征, 形状为[B, C, N]
 * @param idx    (Tensor) 采样点的索引, 形状为[B, npoint]
 * @return       (Tensor) 采样后的特征, 形状为[B, C, npoint]
 */
at::Tensor gather_points(at::Tensor points, at::Tensor idx);

/*
 * @brief 点云特征采样的反向传播(Gather Points Grad)
 *
 * 计算gather_points操作的梯度, 将上游梯度grad_out根据采样索引idx累加回原始输入特征points的位置。
 * 用于训练时反向传播, 确保梯度正确传递到原始点云特征。
 *
 * @param grad_out (Tensor) 上游梯度, 形状为[B, C, npoint]
 * @param idx      (Tensor) 采样点的索引, 形状为[B, npoint]
 * @param n        (int)    原始点的数量N
 * @return         (Tensor) 输入特征points的梯度, 形状为[B, C, N]
 */
at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int n);

/*
 * @brief 最远点采样(Furthest Point Sampling, FPS)
 *
 * 在输入点云points中, 按照最远点策略采样出nsamples个点的索引。
 * 常用于点云下采样, 保证采样点分布均匀, 覆盖整个点云空间。
 *
 * @param points   (Tensor) 输入点云坐标, 形状为[B, N, 3]
 * @param nsamples (int)    需要采样的点数
 * @return         (Tensor) 采样点的索引, 形状为[B, nsamples]
 */
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples);

}  // namespace pointnet2::utils::pt
