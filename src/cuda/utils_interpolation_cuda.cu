#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pointnet2/utils.h"

namespace pointnet2::utils::cuda
{
/*
 * @brief 三近邻查找CUDA核函数
 *
 * 对于每个unknown中的点, 查找known中距离最近的3个点, 返回距离和索引。
 * 常用于点云插值和特征传播阶段。
 *
 * @param b      批量大小
 * @param n      需要查找的点数(unknown的数量)
 * @param m      已知点的数量(known的数量)
 * @param unknown 输入待查找点坐标, 形状为(b, n, 3)
 * @param known   输入已知点坐标, 形状为(b, m, 3)
 * @param dist2   输出, 最近3个点的距离, 形状为(b, n, 3)
 * @param idx     输出, 最近3个点的索引, 形状为(b, n, 3)
 */
__global__ void three_nn_kernel(int b, int n, int m, const float *__restrict__ unknown,
                                const float *__restrict__ known, float *__restrict__ dist2,
                                int *__restrict__ idx)
{
    int batch_index = blockIdx.x;
    // 指针偏移到当前batch的数据
    unknown += batch_index * n * 3;
    known += batch_index * m * 3;
    dist2 += batch_index * n * 3;
    idx += batch_index * n * 3;

    int index = threadIdx.x;
    int stride = blockDim.x;
    // 每个线程负责间隔stride的unknown点
    for (int j = index; j < n; j += stride)
    {
        float ux = unknown[j * 3 + 0];
        float uy = unknown[j * 3 + 1];
        float uz = unknown[j * 3 + 2];

        // 初始化最近三个点的距离和索引
        double best1 = 1e40, best2 = 1e40, best3 = 1e40;
        int besti1 = 0, besti2 = 0, besti3 = 0;
        // 遍历所有已知点, 寻找最近的三个
        for (int k = 0; k < m; ++k)
        {
            float x = known[k * 3 + 0];
            float y = known[k * 3 + 1];
            float z = known[k * 3 + 2];
            float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
            if (d < best1)
            {
                best3 = best2;
                besti3 = besti2;
                best2 = best1;
                besti2 = besti1;
                best1 = d;
                besti1 = k;
            }
            else if (d < best2)
            {
                best3 = best2;
                besti3 = besti2;
                best2 = d;
                besti2 = k;
            }
            else if (d < best3)
            {
                best3 = d;
                besti3 = k;
            }
        }
        // 写入输出：最近三个点的距离和索引
        dist2[j * 3 + 0] = best1;
        dist2[j * 3 + 1] = best2;
        dist2[j * 3 + 2] = best3;

        idx[j * 3 + 0] = besti1;
        idx[j * 3 + 1] = besti2;
        idx[j * 3 + 2] = besti3;
    }
}

/*
 * @brief 三近邻查找CUDA核函数的包装器
 *
 * 设置CUDA流和核函数参数, 并调用three_nn_kernel执行三近邻查找。
 *
 * @param b      批量大小
 * @param n      需要查找的点数
 * @param m      已知点的数量
 * @param unknown 输入待查找点坐标
 * @param known   输入已知点坐标
 * @param dist2   输出, 最近3个点的距离
 * @param idx     输出, 最近3个点的索引
 */
void three_nn_kernel_wrapper(int b, int n, int m, const float *unknown, const float *known,
                             float *dist2, int *idx)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 启动核函数, 每个batch一个block, block内线程数根据n自适应
    three_nn_kernel<<<b, opt_n_threads(n), 0, stream>>>(b, n, m, unknown, known, dist2, idx);

    CUDA_CHECK_ERRORS();
}

/*
 * @brief 三线性插值CUDA核函数
 *
 * 根据三近邻的索引和权重, 对输入特征points进行插值, 得到目标点的特征。
 *
 * @param b      批量大小
 * @param c      特征通道数
 * @param m      已知点的数量
 * @param n      目标点数量
 * @param points 输入已知点特征, 形状为(b, c, m)
 * @param idx    三近邻的索引, 形状为(b, n, 3)
 * @param weight 三近邻的插值权重, 形状为(b, n, 3)
 * @param out    输出插值后的特征, 形状为(b, c, n)
 */
__global__ void three_interpolate_kernel(int b, int c, int m, int n,
                                         const float *__restrict__ points,
                                         const int *__restrict__ idx,
                                         const float *__restrict__ weight, float *__restrict__ out)
{
    int batch_index = blockIdx.x;
    // 指针偏移到当前batch的数据
    points += batch_index * m * c;
    idx += batch_index * n * 3;
    weight += batch_index * n * 3;
    out += batch_index * n * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    // 每个线程负责间隔stride的通道和目标点
    for (int i = index; i < c * n; i += stride)
    {
        const int l = i / n;  // 当前特征通道
        const int j = i % n;  // 当前目标点
        float w1 = weight[j * 3 + 0];
        float w2 = weight[j * 3 + 1];
        float w3 = weight[j * 3 + 2];

        int i1 = idx[j * 3 + 0];
        int i2 = idx[j * 3 + 1];
        int i3 = idx[j * 3 + 2];

        // 按权重对三近邻特征插值
        out[i] = points[l * m + i1] * w1 + points[l * m + i2] * w2 + points[l * m + i3] * w3;
    }
}

/*
 * @brief 三线性插值CUDA核函数的包装器
 *
 * 设置CUDA流和核函数参数, 并调用three_interpolate_kernel执行插值操作。
 *
 * @param b      批量大小
 * @param c      特征通道数
 * @param m      已知点的数量
 * @param n      目标点数量
 * @param points 输入已知点特征
 * @param idx    三近邻的索引
 * @param weight 三近邻的插值权重
 * @param out    输出插值后的特征
 */
void three_interpolate_kernel_wrapper(int b, int c, int m, int n, const float *points,
                                      const int *idx, const float *weight, float *out)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 启动核函数, 每个batch一个block, block内线程数根据n和c自适应
    three_interpolate_kernel<<<b, opt_block_config(n, c), 0, stream>>>(b, c, m, n, points, idx,
                                                                       weight, out);

    CUDA_CHECK_ERRORS();
}

/*
 * @brief 三线性插值反向传播CUDA核函数
 *
 * 计算三线性插值操作的梯度, 将上游梯度grad_out根据索引和权重累加回原始特征grad_points。
 * 用于训练时反向传播, 确保梯度正确传递到原始点云特征。
 *
 * @param b           批量大小
 * @param c           特征通道数
 * @param n           目标点数量
 * @param m           已知点的数量
 * @param grad_out    上游梯度, 形状为(b, c, n)
 * @param idx         三近邻的索引, 形状为(b, n, 3)
 * @param weight      三近邻的插值权重, 形状为(b, n, 3)
 * @param grad_points 输出, 输入特征的梯度, 形状为(b, c, m)
 */
__global__ void three_interpolate_grad_kernel(int b, int c, int n, int m,
                                              const float *__restrict__ grad_out,
                                              const int *__restrict__ idx,
                                              const float *__restrict__ weight,
                                              float *__restrict__ grad_points)
{
    int batch_index = blockIdx.x;
    // 指针偏移到当前batch的数据
    grad_out += batch_index * n * c;
    idx += batch_index * n * 3;
    weight += batch_index * n * 3;
    grad_points += batch_index * m * c;

    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    // 每个线程负责间隔stride的通道和目标点
    for (int i = index; i < c * n; i += stride)
    {
        const int l = i / n;  // 当前特征通道
        const int j = i % n;  // 当前目标点
        float w1 = weight[j * 3 + 0];
        float w2 = weight[j * 3 + 1];
        float w3 = weight[j * 3 + 2];

        int i1 = idx[j * 3 + 0];
        int i2 = idx[j * 3 + 1];
        int i3 = idx[j * 3 + 2];

        // 使用原子加操作将梯度按权重累加到对应的输入特征位置
        atomicAdd(grad_points + l * m + i1, grad_out[i] * w1);
        atomicAdd(grad_points + l * m + i2, grad_out[i] * w2);
        atomicAdd(grad_points + l * m + i3, grad_out[i] * w3);
    }
}

/*
 * @brief 三线性插值反向传播CUDA核函数的包装器
 *
 * 设置CUDA流和核函数参数, 并调用three_interpolate_grad_kernel执行梯度累加操作。
 *
 * @param b           批量大小
 * @param c           特征通道数
 * @param n           目标点数量
 * @param m           已知点的数量
 * @param grad_out    上游梯度
 * @param idx         三近邻的索引
 * @param weight      三近邻的插值权重
 * @param grad_points 输出, 输入特征的梯度
 */
void three_interpolate_grad_kernel_wrapper(int b, int c, int n, int m, const float *grad_out,
                                           const int *idx, const float *weight, float *grad_points)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 启动核函数, 每个batch一个block, block内线程数根据n和c自适应
    three_interpolate_grad_kernel<<<b, opt_block_config(n, c), 0, stream>>>(
        b, c, n, m, grad_out, idx, weight, grad_points);

    CUDA_CHECK_ERRORS();
}
}  // namespace pointnet2::utils::cuda
