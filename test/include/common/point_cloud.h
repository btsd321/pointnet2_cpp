// 基于std::vector的点云类
#pragma once

#include <vector>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <torch/torch.h>

#include "common/point.h"

namespace common
{

    class PointCloud
    {
    public:
        using EigenPointCloud = Eigen::Matrix<double, Eigen::Dynamic, 6>;
        using PclPointCloud = pcl::PointCloud<pcl::PointNormal>;

        enum class ReferenceFrame
        {
            UNDEFINED = 0, // 未定义参考坐标系
            WORLD,         // 世界坐标系
            CAMERA,        // 相机坐标系
            SCENE          // 场景坐标系（桌子中心为原点）
        };

        enum class Scale
        {
            SCALE_M = 0,     // 单位：米
            SCALE_NORMALIZED // 归一化单位
        };

        std::vector<common::Point> points; // 点云坐标

        PointCloud() = default; // 默认构造函数
        PointCloud(std::vector<common::Point> points) : points(points), _reference_frame(ReferenceFrame::UNDEFINED) {}
        PointCloud(std::vector<common::Point> points, ReferenceFrame frame) : points(points), _reference_frame(frame) {}

        static PointCloud from_pclcloud(const PclPointCloud &pcl_cloud);       // 从pcl点云构造
        static PointCloud from_torchcloud(const torch::Tensor &tensor);        // 从torch tensor构造
        static PointCloud from_eigencloud(const EigenPointCloud &eigen_cloud); // 从Eigen点云构造

        PclPointCloud to_pclcloud() const;     // 转换为pcl点云
        torch::Tensor to_torchcloud() const;   // 转换为torch tensor
        EigenPointCloud to_eigencloud() const; // 转换为Eigen点云

        void transform(ReferenceFrame new_frame, const Eigen::Matrix4d &transform); // 变换坐标系
        void set_reference_frame(ReferenceFrame frame);                             // 设置参考坐标系
        ReferenceFrame get_reference_frame() const;                                 // 获取参考坐标系

    private:
        ReferenceFrame _reference_frame = ReferenceFrame::UNDEFINED; // 参考坐标系
        Scale _scale = Scale::SCALE_M;                               // 单位
    };
}
