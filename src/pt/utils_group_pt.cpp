// 分组/采样相关

#include "pointnet2/utils.h"

namespace pointnet2::utils::pt
{
/*
 * @brief 球查询主函数(PyTorch接口)
 *
 * 在点云xyz中, 以new_xyz为中心, 查找半径radius内的邻域点, 最多返回nsample个点的索引。
 * 支持CUDA实现, 不支持CPU。
 *
 * @param new_xyz   (Tensor) 查询中心点的坐标, 形状为[B, npoint, 3]
 * @param xyz       (Tensor) 原始点云的坐标, 形状为[B, N, 3]
 * @param radius    (float)  球查询半径
 * @param nsample   (int)    每个球内最多采样的点数
 * @return          (Tensor) 返回每个中心点对应的邻域点索引, 形状为[B, npoint, nsample]
 */
at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius, const int nsample)
{
    // 检查输入张量是否为连续内存
    CHECK_CONTIGUOUS(new_xyz);
    CHECK_CONTIGUOUS(xyz);
    // 检查输入张量是否为float类型
    CHECK_IS_FLOAT(new_xyz);
    CHECK_IS_FLOAT(xyz);

    // 如果new_xyz在CUDA上, 则xyz也必须在CUDA上
    if (IS_CUDA_TENSOR(new_xyz))
    {
        CHECK_CUDA(xyz);
    }

    // 创建输出张量idx, 初始化为0, 形状为[B, npoint, nsample], 类型为int
    at::Tensor idx = torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                                  at::device(new_xyz.device()).dtype(at::ScalarType::Int));

    // // 如果输入为CUDA张量, 调用CUDA核函数包装器
    // if (IS_CUDA_TENSOR(new_xyz))
    // {
    //     query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1), radius,
    //     nsample,
    //                                     TENSOR_DATA_PTR(new_xyz, float),
    //                                     TENSOR_DATA_PTR(xyz, float), TENSOR_DATA_PTR(idx, int));
    // }
    // else
    // {
    //     // 仅支持CUDA实现, CPU暂不支持
    //     AT_CHECK(false, "CPU not supported");
    // }
    // TODO: CPU实现

    // 返回邻域点索引张量
    return idx;
}

}  // namespace pointnet2::utils::pt
