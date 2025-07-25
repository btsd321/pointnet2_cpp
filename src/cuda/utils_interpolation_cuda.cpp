// 插值/邻域相关

#include "pointnet2/utils.h"

#ifdef USE_CUDA
namespace pointnet2::utils::cuda
{
// CUDA核函数包装器声明, 具体实现在.cu文件中
void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown, const float *known,
                             float *dist2, int *idx);
void three_interpolate_kernel_wrapper(int b, int c, int m, int n, const float *points,
                                      const int *idx, const float *weight, float *out);
void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m, const float *grad_out,
                                           const int *idx, const float *weight, float *grad_points);

/*
 * @brief 三近邻查找主函数(PyTorch接口)
 *
 * 对于每个unknowns中的点, 查找knows中距离最近的3个点, 返回距离和索引。
 * 常用于点云插值和特征传播阶段。
 *
 * @param unknowns (Tensor) 需要插值的点坐标, 形状为[B, n, 3]
 * @param knows    (Tensor) 已知点的坐标, 形状为[B, m, 3]
 * @return         (vector<Tensor>) 返回两个Tensor：
 *                   - dist2: 最近3个点的距离, 形状为[B, n, 3]
 *                   - idx: 最近3个点的索引, 形状为[B, n, 3]
 */
std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows)
{
    // 检查输入张量是否为连续内存且类型正确
    CHECK_CONTIGUOUS(unknowns);
    CHECK_CONTIGUOUS(knows);
    CHECK_IS_FLOAT(unknowns);
    CHECK_IS_FLOAT(knows);

    // 如果unknowns在CUDA上, 则knows也必须在CUDA上
    if (IS_CUDA_TENSOR(unknowns))
    {
        CHECK_CUDA(knows);
    }

    // 创建输出张量, 初始化为0
    at::Tensor idx = torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                                  at::device(unknowns.device()).dtype(at::ScalarType::Int));
    at::Tensor dist2 = torch::zeros({unknowns.size(0), unknowns.size(1), 3},
                                    at::device(unknowns.device()).dtype(at::ScalarType::Float));

    // 调用CUDA核函数进行三近邻查找
    if (IS_CUDA_TENSOR(unknowns))
    {
        three_nn_kernel_wrapper(unknowns.size(0), unknowns.size(1), knows.size(1),
                                TENSOR_DATA_PTR(unknowns, float), TENSOR_DATA_PTR(knows, float),
                                TENSOR_DATA_PTR(dist2, float), TENSOR_DATA_PTR(idx, int));
    }
    else
    {
        // 仅支持CUDA实现, CPU暂不支持
        AT_CHECK(false, "CPU not supported");
    }

    // 返回距离和索引
    return {dist2, idx};
}

/*
 * @brief 三线性插值主函数(PyTorch接口)
 *
 * 根据三近邻的索引和权重, 对输入特征points进行插值, 得到目标点的特征。
 * 常用于点云特征的插值传播。
 *
 * @param points (Tensor) 已知点的特征, 形状为[B, C, m]
 * @param idx    (Tensor) 三近邻的索引, 形状为[B, n, 3]
 * @param weight (Tensor) 三近邻的插值权重, 形状为[B, n, 3]
 * @return       (Tensor) 插值后的特征, 形状为[B, C, n]
 */
at::Tensor three_interpolate(at::Tensor points, at::Tensor idx, at::Tensor weight)
{
    // 检查输入张量是否为连续内存且类型正确
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(weight);
    CHECK_IS_FLOAT(points);
    CHECK_IS_INT(idx);
    CHECK_IS_FLOAT(weight);

    // 如果points在CUDA上, 则idx和weight也必须在CUDA上
    if (IS_CUDA_TENSOR(points))
    {
        CHECK_CUDA(idx);
        CHECK_CUDA(weight);
    }

    // 创建输出张量, 初始化为0
    at::Tensor output = torch::zeros({points.size(0), points.size(1), idx.size(1)},
                                     at::device(points.device()).dtype(at::ScalarType::Float));

    // 调用CUDA核函数进行三线性插值
    if (IS_CUDA_TENSOR(points))
    {
        three_interpolate_kernel_wrapper(points.size(0), points.size(1), points.size(2),
                                         idx.size(1), TENSOR_DATA_PTR(points, float),
                                         TENSOR_DATA_PTR(idx, int), TENSOR_DATA_PTR(weight, float),
                                         TENSOR_DATA_PTR(output, float));
    }
    else
    {
        // 仅支持CUDA实现, CPU暂不支持
        AT_CHECK(false, "CPU not supported");
    }

    // 返回插值后的特征
    return output;
}

/*
 * @brief 三线性插值反向传播主函数(PyTorch接口)
 *
 * 计算三线性插值操作的梯度, 将上游梯度grad_out根据索引和权重累加回原始特征points的位置。
 * 用于训练时反向传播, 确保梯度正确传递到原始点云特征。
 *
 * @param grad_out (Tensor) 上游梯度, 形状为[B, C, n]
 * @param idx      (Tensor) 三近邻的索引, 形状为[B, n, 3]
 * @param weight   (Tensor) 三近邻的插值权重, 形状为[B, n, 3]
 * @param m        (int)    已知点的数量m
 * @return         (Tensor) 输入特征points的梯度, 形状为[B, C, m]
 */
at::Tensor three_interpolate_grad(at::Tensor grad_out, at::Tensor idx, at::Tensor weight,
                                  const int m)
{
    // 检查输入张量是否为连续内存且类型正确
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(idx);
    CHECK_CONTIGUOUS(weight);
    CHECK_IS_FLOAT(grad_out);
    CHECK_IS_INT(idx);
    CHECK_IS_FLOAT(weight);

    // 如果grad_out在CUDA上, 则idx和weight也必须在CUDA上
    if (IS_CUDA_TENSOR(grad_out))
    {
        CHECK_CUDA(idx);
        CHECK_CUDA(weight);
    }

    // 创建输出张量, 初始化为0
    at::Tensor output = torch::zeros({grad_out.size(0), grad_out.size(1), m},
                                     at::device(grad_out.device()).dtype(at::ScalarType::Float));

    // 调用CUDA核函数进行三线性插值的反向传播
    if (IS_CUDA_TENSOR(grad_out))
    {
        three_interpolate_grad_kernel_wrapper(
            grad_out.size(0), grad_out.size(1), grad_out.size(2), m,
            TENSOR_DATA_PTR(grad_out, float), TENSOR_DATA_PTR(idx, int),
            TENSOR_DATA_PTR(weight, float), TENSOR_DATA_PTR(output, float));
    }
    else
    {
        // 仅支持CUDA实现, CPU暂不支持
        AT_CHECK(false, "CPU not supported");
    }

    // 返回输入特征的梯度
    return output;
}
}  // namespace pointnet2::utils::cuda

#endif  // USE_CUDA
