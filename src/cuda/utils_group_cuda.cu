#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pointnet2/utils.h"

namespace pointnet2::utils::cuda
{
/*
 * @brief 球查询CUDA核函数
 *
 * 在点云xyz中, 以new_xyz为中心, 查找半径radius内的邻域点, 最多返回nsample个点的索引。
 * 用于点云处理中的邻域搜索, 常见于PointNet++等网络。
 *
 * @param b        批量大小(batch size)
 * @param n        每批次点云的点数
 * @param m        每批次查询中心点的数量
 * @param radius   球查询半径
 * @param nsample  每个球内最多采样的点数
 * @param new_xyz  查询中心点的坐标, 形状为(b, m, 3)
 * @param xyz      原始点云的坐标, 形状为(b, n, 3)
 * @param idx      输出, 每个中心点对应的邻域点索引, 形状为(b, m, nsample)
 */
__global__ void query_ball_point_kernel(int b, int n, int m, float radius, int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz, int *__restrict__ idx)
{
    // 计算当前处理的batch索引
    int batch_index = blockIdx.x;
    // 指针偏移到当前batch的数据
    xyz += batch_index * n * 3;
    new_xyz += batch_index * m * 3;
    idx += m * nsample * batch_index;

    // 当前线程负责的中心点索引起点
    int index = threadIdx.x;
    // 每个线程的步长
    int stride = blockDim.x;

    float radius2 = radius * radius;  // 预先计算半径平方, 便于距离比较
    // 遍历本batch的所有中心点, 每个线程负责间隔stride的中心点
    for (int j = index; j < m; j += stride)
    {
        float new_x = new_xyz[j * 3 + 0];
        float new_y = new_xyz[j * 3 + 1];
        float new_z = new_xyz[j * 3 + 2];
        // cnt用于统计已找到的邻域点数
        for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k)
        {
            float x = xyz[k * 3 + 0];
            float y = xyz[k * 3 + 1];
            float z = xyz[k * 3 + 2];
            // 计算欧氏距离的平方
            float d2 =
                (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
            // 如果距离小于半径, 说明k点在球内
            if (d2 < radius2)
            {
                // 第一个找到的点, 先将所有位置填为该点索引, 保证至少有一个有效索引
                if (cnt == 0)
                {
                    for (int l = 0; l < nsample; ++l)
                    {
                        idx[j * nsample + l] = k;
                    }
                }
                // 记录当前找到的点的索引
                idx[j * nsample + cnt] = k;
                ++cnt;
            }
        }
    }
}

/*
 * @brief 球查询CUDA核函数的包装器
 *
 * 负责设置CUDA流和核函数参数, 并调用核函数执行球查询操作。
 *
 * @param b        批量大小(batch size)
 * @param n        每批次点云的点数
 * @param m        每批次查询中心点的数量
 * @param radius   球查询半径
 * @param nsample  每个球内最多采样的点数
 * @param new_xyz  查询中心点的坐标, 形状为(b, m, 3)
 * @param xyz      原始点云的坐标, 形状为(b, n, 3)
 * @param idx      输出, 每个中心点对应的邻域点索引, 形状为(b, m, nsample)
 */
void query_ball_point_kernel_wrapper(int b, int n, int m, float radius, int nsample,
                                     const float *new_xyz, const float *xyz, int *idx)
{
    // 获取当前CUDA流
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // 启动核函数, 每个batch一个block, block内线程数根据m自适应
    query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(b, n, m, radius, nsample, new_xyz,
                                                                xyz, idx);

    // 检查CUDA错误
    CUDA_CHECK_ERRORS();
}

/*
 * @brief 点云分组操作的CUDA核函数
 *
 * 根据给定的采样索引idx, 从输入点云特征points中采样, 返回分组后的特征out。
 * 常用于PointNet++等点云网络的局部特征聚合阶段, 实现对每个采样中心点的邻域特征提取。
 *
 * @param b        批量大小(batch size)
 * @param c        特征通道数
 * @param n        每批次点云的点数
 * @param npoints  每批次采样中心点的数量
 * @param nsample  每个中心点的邻域采样点数
 * @param points   输入点云特征, 形状为(b, c, n)
 * @param idx      分组采样的索引, 形状为(b, npoints, nsample)
 * @param out      输出分组后的特征, 形状为(b, c, npoints, nsample)
 */
__global__ void group_points_kernel(int b, int c, int n, int npoints, int nsample,
                                    const float *__restrict__ points, const int *__restrict__ idx,
                                    float *__restrict__ out)
{
    // 当前处理的batch索引
    int batch_index = blockIdx.x;
    // 指针偏移到当前batch的数据
    points += batch_index * n * c;
    idx += batch_index * npoints * nsample;
    out += batch_index * npoints * nsample * c;

    // 计算当前线程负责的特征通道和采样中心点的索引
    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    // 遍历所有通道和采样中心点, 每个线程负责间隔stride的任务
    for (int i = index; i < c * npoints; i += stride)
    {
        const int l = i / npoints;  // 当前特征通道
        const int j = i % npoints;  // 当前采样中心点
        // 遍历每个中心点的邻域采样
        for (int k = 0; k < nsample; ++k)
        {
            int ii = idx[j * nsample + k];  // 采样点在原始点云中的索引
            // 将采样到的特征写入输出
            out[(l * npoints + j) * nsample + k] = points[l * n + ii];
        }
    }
}

/*
 * @brief 点云分组操作CUDA核函数的包装器
 *
 * 设置CUDA流和核函数参数, 并调用group_points_kernel执行分组采样操作。
 *
 * @param b        批量大小
 * @param c        特征通道数
 * @param n        每批次点云的点数
 * @param npoints  采样中心点数量
 * @param nsample  每个中心点的邻域采样点数
 * @param points   输入点云特征
 * @param idx      分组采样的索引
 * @param out      输出分组后的特征
 */
void group_points_kernel_wrapper(int b, int c, int n, int npoints, int nsample, const float *points,
                                 const int *idx, float *out)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 启动CUDA核函数, 每个batch一个block, block内线程数根据npoints和c自适应
    group_points_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(b, c, n, npoints, nsample,
                                                                        points, idx, out);

    CUDA_CHECK_ERRORS();
}

/*
 * @brief 点云分组操作反向传播的CUDA核函数
 *
 * 计算group_points操作的梯度,
 * 将上游梯度grad_out根据采样索引idx累加回原始输入特征grad_points的位置。 用于训练时反向传播,
 * 确保梯度正确传递到原始点云特征。
 *
 * @param b           批量大小
 * @param c           特征通道数
 * @param n           每批次点云的点数
 * @param npoints     采样中心点数量
 * @param nsample     每个中心点的邻域采样点数
 * @param grad_out    上游梯度, 形状为(b, c, npoints, nsample)
 * @param idx         分组采样的索引, 形状为(b, npoints, nsample)
 * @param grad_points 输出, 输入特征的梯度, 形状为(b, c, n)
 */
__global__ void group_points_grad_kernel(int b, int c, int n, int npoints, int nsample,
                                         const float *__restrict__ grad_out,
                                         const int *__restrict__ idx,
                                         float *__restrict__ grad_points)
{
    // 当前处理的batch索引
    int batch_index = blockIdx.x;
    // 指针偏移到当前batch的数据
    grad_out += batch_index * npoints * nsample * c;
    idx += batch_index * npoints * nsample;
    grad_points += batch_index * n * c;

    // 计算当前线程负责的特征通道和采样中心点的索引
    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.y * blockDim.x;
    // 遍历所有通道和采样中心点, 每个线程负责间隔stride的任务
    for (int i = index; i < c * npoints; i += stride)
    {
        const int l = i / npoints;  // 当前特征通道
        const int j = i % npoints;  // 当前采样中心点
        // 遍历每个中心点的邻域采样
        for (int k = 0; k < nsample; ++k)
        {
            int ii = idx[j * nsample + k];  // 采样点在原始点云中的索引
            // 使用原子加操作将梯度累加到对应的输入特征位置
            atomicAdd(grad_points + l * n + ii, grad_out[(l * npoints + j) * nsample + k]);
        }
    }
}

/*
 * @brief 点云分组操作反向传播CUDA核函数的包装器
 *
 * 设置CUDA流和核函数参数, 并调用group_points_grad_kernel执行梯度累加操作。
 *
 * @param b           批量大小
 * @param c           特征通道数
 * @param n           每批次点云的点数
 * @param npoints     采样中心点数量
 * @param nsample     每个中心点的邻域采样点数
 * @param grad_out    上游梯度
 * @param idx         分组采样的索引
 * @param grad_points 输出, 输入特征的梯度
 */
void group_points_grad_kernel_wrapper(int b, int c, int n, int npoints, int nsample,
                                      const float *grad_out, const int *idx, float *grad_points)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 启动CUDA核函数, 每个batch一个block, block内线程数根据npoints和c自适应
    group_points_grad_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
        b, c, n, npoints, nsample, grad_out, idx, grad_points);

    CUDA_CHECK_ERRORS();
}

/*
 * @brief 点云特征采样CUDA核函数
 *
 * 根据给定的采样索引idx, 从输入点云特征points中采样, 返回采样后的特征out。
 * 常用于根据采样点索引提取对应的特征。
 *
 * @param b      批量大小
 * @param c      特征通道数
 * @param n      每批次点云的点数
 * @param m      采样点数量
 * @param points 输入点云特征, 形状为(b, c, n)
 * @param idx    采样点的索引, 形状为(b, m)
 * @param out    输出采样后的特征, 形状为(b, c, m)
 */
__global__ void gather_points_kernel(int b, int c, int n, int m, const float *__restrict__ points,
                                     const int *__restrict__ idx, float *__restrict__ out)
{
    // 遍历batch和通道, 每个线程处理一个采样点
    for (int i = blockIdx.x; i < b; i += gridDim.x)
    {
        for (int l = blockIdx.y; l < c; l += gridDim.y)
        {
            for (int j = threadIdx.x; j < m; j += blockDim.x)
            {
                int a = idx[i * m + j];  // 获取采样点在原始点云中的索引
                out[(i * c + l) * m + j] = points[(i * c + l) * n + a];  // 采样赋值
            }
        }
    }
}

/*
 * @brief 点云特征采样CUDA核函数的包装器
 *
 * 设置CUDA流和核函数参数, 并调用gather_points_kernel执行采样操作。
 *
 * @param b      批量大小
 * @param c      特征通道数
 * @param n      每批次点云的点数
 * @param npoints 采样点数量
 * @param points 输入点云特征
 * @param idx    采样点的索引
 * @param out    输出采样后的特征
 */
void gather_points_kernel_wrapper(int b, int c, int n, int npoints, const float *points,
                                  const int *idx, float *out)
{
    gather_points_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                           at::cuda::getCurrentCUDAStream()>>>(b, c, n, npoints, points, idx, out);

    CUDA_CHECK_ERRORS();
}

/*
 * @brief 点云特征采样反向传播CUDA核函数
 *
 * 计算gather_points操作的梯度,
 * 将上游梯度grad_out根据采样索引idx累加回原始输入特征grad_points的位置。 用于训练时反向传播,
 * 确保梯度正确传递到原始点云特征。
 *
 * @param b         批量大小
 * @param c         特征通道数
 * @param n         每批次点云的点数
 * @param m         采样点数量
 * @param grad_out  上游梯度, 形状为(b, c, m)
 * @param idx       采样点的索引, 形状为(b, m)
 * @param grad_points 输出, 输入特征的梯度, 形状为(b, c, n)
 */
__global__ void gather_points_grad_kernel(int b, int c, int n, int m,
                                          const float *__restrict__ grad_out,
                                          const int *__restrict__ idx,
                                          float *__restrict__ grad_points)
{
    // 遍历batch和通道, 每个线程处理一个采样点
    for (int i = blockIdx.x; i < b; i += gridDim.x)
    {
        for (int l = blockIdx.y; l < c; l += gridDim.y)
        {
            for (int j = threadIdx.x; j < m; j += blockDim.x)
            {
                int a = idx[i * m + j];  // 获取采样点在原始点云中的索引
                // 使用原子加操作将梯度累加到对应的输入特征位置
                atomicAdd(grad_points + (i * c + l) * n + a, grad_out[(i * c + l) * m + j]);
            }
        }
    }
}

/*
 * @brief 点云特征采样反向传播CUDA核函数的包装器
 *
 * 设置CUDA流和核函数参数, 并调用gather_points_grad_kernel执行梯度累加操作。
 *
 * @param b         批量大小
 * @param c         特征通道数
 * @param n         每批次点云的点数
 * @param npoints   采样点数量
 * @param grad_out  上游梯度
 * @param idx       采样点的索引
 * @param grad_points 输出, 输入特征的梯度
 */
void gather_points_grad_kernel_wrapper(int b, int c, int n, int npoints, const float *grad_out,
                                       const int *idx, float *grad_points)
{
    gather_points_grad_kernel<<<dim3(b, c, 1), opt_n_threads(npoints), 0,
                                at::cuda::getCurrentCUDAStream()>>>(b, c, n, npoints, grad_out, idx,
                                                                    grad_points);

    CUDA_CHECK_ERRORS();
}

/*
 * @brief 辅助函数：更新距离和索引的共享内存
 *
 * 用于最远点采样过程中, 比较并保留较大距离及其对应索引。
 *
 * @param dists   距离数组(共享内存)
 * @param dists_i 距离对应的索引数组(共享内存)
 * @param idx1    第一个比较的下标
 * @param idx2    第二个比较的下标
 */
__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i, int idx1, int idx2)
{
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

/*
 * @brief 最远点采样CUDA核函数(模板版本)
 *
 * 在输入点云dataset中, 按照最远点策略采样出m个点的索引, 保证采样点分布均匀。
 * 常用于点云下采样, 覆盖整个点云空间。
 *
 * @tparam block_size CUDA block的线程数
 * @param b        批量大小
 * @param n        每批次点云的点数
 * @param m        需要采样的点数
 * @param dataset  输入点云坐标, 形状为(b, n, 3)
 * @param temp     临时距离缓存, 形状为(b, n)
 * @param idxs     输出采样点的索引, 形状为(b, m)
 */
template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(int b, int n, int m,
                                               const float *__restrict__ dataset,
                                               float *__restrict__ temp, int *__restrict__ idxs)
{
    if (m <= 0)
        return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    dataset += batch_index * n * 3;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
        idxs[0] = old;  // 第一个采样点索引初始化为0

    __syncthreads();
    // 逐步采样m个点, 每次选择距离当前已采样点集合最远的点
    for (int j = 1; j < m; j++)
    {
        int besti = 0;
        float best = -1;
        float x1 = dataset[old * 3 + 0];
        float y1 = dataset[old * 3 + 1];
        float z1 = dataset[old * 3 + 2];
        for (int k = tid; k < n; k += stride)
        {
            float x2, y2, z2;
            x2 = dataset[k * 3 + 0];
            y2 = dataset[k * 3 + 1];
            z2 = dataset[k * 3 + 2];
            float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
            if (mag <= 1e-3)
                continue;  // 跳过无效点

            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

            float d2 = min(d, temp[k]);
            temp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        // 归约操作, 找到本block内距离最大的点及其索引
        if (block_size >= 512)
        {
            if (tid < 256)
            {
                __update(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (block_size >= 256)
        {
            if (tid < 128)
            {
                __update(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (block_size >= 128)
        {
            if (tid < 64)
            {
                __update(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (block_size >= 64)
        {
            if (tid < 32)
            {
                __update(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (block_size >= 32)
        {
            if (tid < 16)
            {
                __update(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (block_size >= 16)
        {
            if (tid < 8)
            {
                __update(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (block_size >= 8)
        {
            if (tid < 4)
            {
                __update(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (block_size >= 4)
        {
            if (tid < 2)
            {
                __update(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (block_size >= 2)
        {
            if (tid < 1)
            {
                __update(dists, dists_i, tid, tid + 1);
            }
            __syncthreads();
        }

        old = dists_i[0];  // 选出距离最远的点作为下一个采样点
        if (tid == 0)
            idxs[j] = old;
    }
}

/*
 * @brief 最远点采样CUDA核函数的包装器
 *
 * 根据输入点云的规模自动选择合适的block size,
 * 设置CUDA流并调用furthest_point_sampling_kernel执行采样操作。
 *
 * @param b        批量大小
 * @param n        每批次点云的点数
 * @param m        需要采样的点数
 * @param dataset  输入点云坐标
 * @param temp     临时距离缓存
 * @param idxs     输出采样点的索引
 */
void furthest_point_sampling_kernel_wrapper(int b, int n, int m, const float *dataset, float *temp,
                                            int *idxs)
{
    unsigned int n_threads = opt_n_threads(n);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 根据线程数选择对应模板实例, 保证效率
    switch (n_threads)
    {
        case 512:
            furthest_point_sampling_kernel<512>
                <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 256:
            furthest_point_sampling_kernel<256>
                <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 128:
            furthest_point_sampling_kernel<128>
                <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 64:
            furthest_point_sampling_kernel<64>
                <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 32:
            furthest_point_sampling_kernel<32>
                <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 16:
            furthest_point_sampling_kernel<16>
                <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 8:
            furthest_point_sampling_kernel<8>
                <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 4:
            furthest_point_sampling_kernel<4>
                <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 2:
            furthest_point_sampling_kernel<2>
                <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        case 1:
            furthest_point_sampling_kernel<1>
                <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
            break;
        default:
            furthest_point_sampling_kernel<512>
                <<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    }

    CUDA_CHECK_ERRORS();
}

}  // namespace pointnet2::utils::cuda
