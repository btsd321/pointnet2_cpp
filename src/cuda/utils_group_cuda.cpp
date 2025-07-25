// 分组/采样相关

#include "pointnet2/utils.h"

#ifdef USE_CUDA
namespace pointnet2::utils::cuda
{
/*
 * @brief CUDA球查询核函数的包装器声明
 *
 * 该函数在.cu文件中实现, 用于在GPU上执行球查询操作。
 *
 * @param b        批量大小(batch size)
 * @param n        每批次点云的点数
 * @param m        每批次查询中心点的数量
 * @param radius   球查询半径
 * @param nsample  每个球内最多采样的点数
 * @param new_xyz  查询中心点的坐标指针, 形状为(b, m, 3)
 * @param xyz      原始点云的坐标指针, 形状为(b, n, 3)
 * @param idx      输出, 每个中心点对应的邻域点索引指针, 形状为(b, m, nsample)
 */
void query_ball_point_kernel_wrapper(int b, int n, int m, float radius, int nsample,
                                     const float *new_xyz, const float *xyz, int *idx);

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

    // 如果输入为CUDA张量, 调用CUDA核函数包装器
    if (IS_CUDA_TENSOR(new_xyz))
    {
        query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1), radius, nsample,
                                        TENSOR_DATA_PTR(new_xyz, float),
                                        TENSOR_DATA_PTR(xyz, float), TENSOR_DATA_PTR(idx, int));
    }
    else
    {
        // 仅支持CUDA实现, CPU暂不支持
        AT_CHECK(false, "CPU not supported");
    }

    // 返回邻域点索引张量
    return idx;
}

/*
 * @brief 点云分组操作CUDA核函数的包装器声明
 *
 * 该函数在.cu文件中实现, 用于在GPU上执行分组采样操作。
 *
 * @param b        批量大小
 * @param c        特征通道数
 * @param n        每批次点云的点数
 * @param npoints  采样中心点数量
 * @param nsample  每个中心点的邻域采样点数
 * @param points   输入点云特征指针
 * @param idx      分组采样的索引指针
 * @param out      输出分组后的特征指针
 */
void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample, const float *points,
                                 const int *idx, float *out);

/*
 * @brief 点云分组操作反向传播CUDA核函数的包装器声明
 *
 * 该函数在.cu文件中实现, 用于在GPU上执行分组采样的反向传播操作。
 *
 * @param b           批量大小
 * @param c           特征通道数
 * @param n           每批次点云的点数
 * @param npoints     采样中心点数量
 * @param nsample     每个中心点的邻域采样点数
 * @param grad_out    上游梯度指针
 * @param idx         分组采样的索引指针
 * @param grad_points 输出, 输入特征的梯度指针
 */
void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                      const float *grad_out, const int *idx, float *grad_points);

/*
 * @brief 点云分组操作主函数(PyTorch接口)
 *
 * 根据给定的采样索引idx, 从输入点云特征points中采样, 返回分组后的特征。
 * 常用于PointNet++等点云网络的局部特征聚合阶段, 实现对每个采样中心点的邻域特征提取。
 *
 * @param points (Tensor) 输入点云特征, 形状为[B, C, N]
 * @param idx    (Tensor) 分组采样的索引, 形状为[B, npoint, nsample]
 * @return       (Tensor) 分组后的特征, 形状为[B, C, npoint, nsample]
 */
at::Tensor group_points(at::Tensor points, at::Tensor idx)
{
    // 检查输入张量是否为连续内存、类型是否正确
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);

    // 如果points在CUDA上, 则idx也必须在CUDA上
    if (IS_CUDA_TENSOR(points))
    {
        CHECK_CUDA(idx);
    }

    // 创建输出张量, 初始化为0, 形状为[B, C, npoint, nsample]
    at::Tensor output = torch::zeros({points.size(0), points.size(1), idx.size(1), idx.size(2)},
                                     at::device(points.device()).dtype(at::ScalarType::Float));

    // 如果输入为CUDA张量, 调用CUDA核函数包装器
    if (IS_CUDA_TENSOR(points))
    {
        group_points_kernel_wrapper(points.size(0), points.size(1), points.size(2), idx.size(1),
                                    idx.size(2), TENSOR_DATA_PTR(points, float),
                                    TENSOR_DATA_PTR(idx, int), TENSOR_DATA_PTR(output, float));
    }
    else
    {
        // 仅支持CUDA实现, CPU暂不支持
        AT_CHECK(false, "CPU not supported");
    }

    // 返回分组后的特征
    return output;
}

/*
 * @brief 点云分组操作反向传播主函数(PyTorch接口)
 *
 * 计算group_points操作的梯度, 将上游梯度grad_out根据采样索引idx累加回原始输入特征points的位置。
 * 用于训练时反向传播, 确保梯度正确传递到原始点云特征。
 *
 * @param grad_out (Tensor) 上游梯度, 形状为[B, C, npoint, nsample]
 * @param idx      (Tensor) 分组采样的索引, 形状为[B, npoint, nsample]
 * @param n        (int)    原始点的数量N
 * @return         (Tensor) 输入特征points的梯度, 形状为[B, C, N]
 */
at::Tensor group_points_grad(at::Tensor grad_out, at::Tensor idx, const int n)
{
    // 检查输入张量是否为连续内存、类型是否正确
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);

    // 如果grad_out在CUDA上, 则idx也必须在CUDA上
    if (IS_CUDA_TENSOR(grad_out))
    {
        CHECK_CUDA(idx);
    }

    // 创建输出张量, 初始化为0, 形状为[B, C, N]
    at::Tensor output = torch::zeros({grad_out.size(0), grad_out.size(1), n},
                                     at::device(grad_out.device()).dtype(at::ScalarType::Float));

    // 如果输入为CUDA张量, 调用CUDA核函数包装器
    if (IS_CUDA_TENSOR(grad_out))
    {
        group_points_grad_kernel_wrapper(grad_out.size(0), grad_out.size(1), n, idx.size(1),
                                         idx.size(2), TENSOR_DATA_PTR(grad_out, float),
                                         TENSOR_DATA_PTR(idx, int), TENSOR_DATA_PTR(output, float));
    }
    else
    {
        // 仅支持CUDA实现, CPU暂不支持
        AT_CHECK(false, "CPU not supported");
    }

    // 返回输入特征的梯度
    return output;
}

// CUDA核函数包装器声明, 具体实现在.cu文件中
void gather_points_kernel_wrapper(int b, int c, int n, int npoints, const float *points,
                                  const int *idx, float *out);
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints, const float *grad_out,
                                       const int *idx, float *grad_points);
void furthest_point_sampling_kernel_wrapper(int b, int n, int m, const float *dataset, float *temp,
                                            int *idxs);

/*
 * @brief 点云特征采样主函数(PyTorch接口)
 *
 * 根据给定的采样索引idx, 从输入点云特征points中采样, 返回采样后的特征。
 * 常用于根据采样点索引提取对应的特征。
 *
 * @param points (Tensor) 输入点云特征, 形状为[B, C, N]
 * @param idx    (Tensor) 采样点的索引, 形状为[B, npoints]
 * @return       (Tensor) 采样后的特征, 形状为[B, C, npoints]
 */
at::Tensor gather_points(at::Tensor points, at::Tensor idx)
{
    // 检查输入张量是否为连续内存且类型正确
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);

    // 如果points在CUDA上, 则idx也必须在CUDA上
    if (IS_CUDA_TENSOR(points))
    {
        CHECK_CUDA(idx);
    }

    // 创建输出张量, 初始化为0, 形状为[B, C, npoints]
    at::Tensor output = torch::zeros({points.size(0), points.size(1), idx.size(1)},
                                     at::device(points.device()).dtype(at::ScalarType::Float));

    // 调用CUDA核函数进行特征采样
    if (IS_CUDA_TENSOR(points))
    {
        gather_points_kernel_wrapper(points.size(0), points.size(1), points.size(2), idx.size(1),
                                     TENSOR_DATA_PTR(points, float), TENSOR_DATA_PTR(idx, int),
                                     TENSOR_DATA_PTR(output, float));
    }
    else
    {
        // 仅支持CUDA实现, CPU暂不支持
        AT_CHECK(false, "CPU not supported");
    }

    // 返回采样后的特征
    return output;
}

/*
 * @brief 点云特征采样反向传播主函数(PyTorch接口)
 *
 * 计算gather_points操作的梯度, 将上游梯度grad_out根据采样索引idx累加回原始输入特征points的位置。
 * 用于训练时反向传播, 确保梯度正确传递到原始点云特征。
 *
 * @param grad_out (Tensor) 上游梯度, 形状为[B, C, npoints]
 * @param idx      (Tensor) 采样点的索引, 形状为[B, npoints]
 * @param n        (int)    原始点的数量N
 * @return         (Tensor) 输入特征points的梯度, 形状为[B, C, N]
 */
at::Tensor gather_points_grad(at::Tensor grad_out, at::Tensor idx, const int n)
{
    // 检查输入张量是否为连续内存且类型正确
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);

    // 如果grad_out在CUDA上, 则idx也必须在CUDA上
    if (IS_CUDA_TENSOR(grad_out))
    {
        CHECK_CUDA(idx);
    }

    // 创建输出张量, 初始化为0, 形状为[B, C, N]
    at::Tensor output = torch::zeros({grad_out.size(0), grad_out.size(1), n},
                                     at::device(grad_out.device()).dtype(at::ScalarType::Float));

    // 调用CUDA核函数进行特征采样的反向传播
    if (IS_CUDA_TENSOR(grad_out))
    {
        gather_points_grad_kernel_wrapper(
            grad_out.size(0), grad_out.size(1), n, idx.size(1), TENSOR_DATA_PTR(grad_out, float),
            TENSOR_DATA_PTR(idx, int), TENSOR_DATA_PTR(output, float));
    }
    else
    {
        // 仅支持CUDA实现, CPU暂不支持
        AT_CHECK(false, "CPU not supported");
    }

    // 返回输入特征的梯度
    return output;
}

/*
 * @brief 最远点采样主函数(PyTorch接口)
 *
 * 在输入点云points中, 按照最远点策略采样出nsamples个点的索引, 保证采样点分布均匀。
 * 常用于点云下采样, 覆盖整个点云空间。
 *
 * @param points   (Tensor) 输入点云坐标, 形状为[B, N, 3]
 * @param nsamples (int)    需要采样的点数
 * @return         (Tensor) 采样点的索引, 形状为[B, nsamples]
 */
at::Tensor furthest_point_sampling(at::Tensor points, const int nsamples)
{
    // 检查输入张量是否为连续内存且类型正确
    CHECK_CONTIGUOUS(points);
    CHECK_IS_FLOAT(points);

    // 创建输出张量, 存储采样点索引, 形状为[B, nsamples]
    at::Tensor output = torch::zeros({points.size(0), nsamples},
                                     at::device(points.device()).dtype(at::ScalarType::Int));

    // 创建临时距离缓存, 初始化为较大值, 形状为[B, N]
    at::Tensor tmp = torch::full({points.size(0), points.size(1)}, 1e10,
                                 at::device(points.device()).dtype(at::ScalarType::Float));

    // 调用CUDA核函数进行最远点采样
    if (IS_CUDA_TENSOR(points))
    {
        furthest_point_sampling_kernel_wrapper(
            points.size(0), points.size(1), nsamples, TENSOR_DATA_PTR(points, float),
            TENSOR_DATA_PTR(tmp, float), TENSOR_DATA_PTR(output, int));
    }
    else
    {
        // 仅支持CUDA实现, CPU暂不支持
        AT_CHECK(false, "CPU not supported");
    }

    // 返回采样点索引
    return output;
}

}  // namespace pointnet2::utils::cuda

#endif  // USE_CUDA
