#pragma once

#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <torch/torch.h>

namespace common
{
    class Point
    {
        using EigenPoint = Eigen::Matrix<double, 1, 6>; // 6维点，包含位置和法向量，1行6列
        using PclPoint = pcl::PointNormal;              // PCL点类型
    public:
        Point() : x(0), y(0), z(0), nx(0), ny(0), nz(0) {}
        Point(double x, double y, double z, double nx, double ny, double nz) : x(x), y(y), z(z), nx(nx), ny(ny), nz(nz) {}
        double x, y, z;    // 坐标
        double nx, ny, nz; // 法向量

        static Point from_pcl_point(const PclPoint &pcl_point);
        static Point from_torch_tensor(const torch::Tensor &tensor);
        static Point from_eigen_point(const EigenPoint &eigen_point);

        PclPoint to_pcl_point() const;
        torch::Tensor to_torch_tensor() const;
        EigenPoint to_eigen_point() const;
    };
} // namespace common
