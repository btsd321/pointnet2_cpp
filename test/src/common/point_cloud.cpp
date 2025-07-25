// 基于std::vector的点云类
#include <pcl/visualization/pcl_visualizer.h>
#include "common/point_cloud.h"

namespace common
{
    PointCloud PointCloud::from_pclcloud(const PclPointCloud &pcl_cloud)
    {
        PointCloud cloud;
        for (const auto &p : pcl_cloud.points)
        {
            cloud.points.emplace_back(Point::from_pcl_point(p));
        }
        return cloud;
    }

    PointCloud PointCloud::from_torchcloud(const torch::Tensor &tensor)
    {
        // 检查形状是否为(1, n, 6)
        if (tensor.dim() != 3 || tensor.size(0) != 1 || tensor.size(2) != 6)
        {
            throw std::runtime_error("Input tensor must have shape (1, n, 6)");
        }
        int n = tensor.size(1);
        PointCloud cloud;
        for (int i = 0; i < n; ++i)
        {
            cloud.points.emplace_back(Point::from_torch_tensor(tensor[0][i]));
        }
        return cloud;
    }

    PointCloud PointCloud::from_eigencloud(const EigenPointCloud &eigen_cloud)
    {
        PointCloud cloud;
        for (int i = 0; i < eigen_cloud.rows(); ++i)
        {
            cloud.points.emplace_back(Point::from_eigen_point(eigen_cloud.row(i)));
        }
        return cloud;
    }

    PointCloud::PclPointCloud PointCloud::to_pclcloud() const
    {
        PclPointCloud pcl_cloud;
        for (const auto &p : points)
        {
            pcl_cloud.points.emplace_back(p.to_pcl_point());
        }
        return pcl_cloud;
    }

    torch::Tensor PointCloud::to_torchcloud() const
    {
        torch::Tensor tensor = torch::zeros({static_cast<int64_t>(this->points.size()), 6});
        for (int i = 0; i < this->points.size(); ++i)
        {
            tensor[i] = this->points[i].to_torch_tensor();
        }
        return tensor;
    }

    PointCloud::EigenPointCloud PointCloud::to_eigencloud() const
    {
        EigenPointCloud eigen_cloud(this->points.size(), 6);
        for (int i = 0; i < this->points.size(); ++i)
        {
            eigen_cloud.row(i) = this->points[i].to_eigen_point();
        }
        return eigen_cloud;
    }

    // 变换坐标系
    void PointCloud::transform(ReferenceFrame new_frame, const Eigen::Matrix4d &transform)
    {
        if (new_frame == this->_reference_frame)
        {
            throw std::runtime_error("The new reference frame is the same as the current one.");
            return;
        }
        // 将所有点变换到新坐标系
        // 构造 Nx3 坐标矩阵和 Nx3 法向量矩阵
        Eigen::MatrixXd coords(points.size(), 3);
        Eigen::MatrixXd normals(points.size(), 3);
        for (size_t i = 0; i < points.size(); ++i)
        {
            coords.row(i) << points[i].x, points[i].y, points[i].z;
            normals.row(i) << points[i].nx, points[i].ny, points[i].nz;
        }

        // 坐标齐次化
        Eigen::MatrixXd coords_hom(points.size(), 4);
        coords_hom << coords, Eigen::VectorXd::Ones(points.size());

        // 变换
        Eigen::MatrixXd new_coords_hom = coords_hom * transform.transpose();
        Eigen::MatrixXd new_coords = new_coords_hom.leftCols(3);

        // 法向量只做旋转
        Eigen::Matrix3d rot = transform.block<3, 3>(0, 0);
        Eigen::MatrixXd new_normals = normals * rot.transpose();

        // 回写
        for (size_t i = 0; i < points.size(); ++i)
        {
            points[i].x = new_coords(i, 0);
            points[i].y = new_coords(i, 1);
            points[i].z = new_coords(i, 2);
            points[i].nx = new_normals(i, 0);
            points[i].ny = new_normals(i, 1);
            points[i].nz = new_normals(i, 2);
        }
        this->_reference_frame = new_frame;
    }

    // 设置参考坐标系
    void PointCloud::set_reference_frame(ReferenceFrame frame)
    {
        this->_reference_frame = frame;
    }

    // 获取参考坐标系
    PointCloud::ReferenceFrame PointCloud::get_reference_frame() const
    {
        return this->_reference_frame;
    }

    // 可视化
    void PointCloud::display() const
    {
        auto pcl_cloud = this->to_pclcloud().makeShared();
        pcl::visualization::PCLVisualizer::Ptr viewer = std::make_shared<pcl::visualization::PCLVisualizer>("3D Viewer");
        // 设置背景颜色为灰色
        viewer->setBackgroundColor(100, 100, 100);
        // 参数表示坐标轴的长度，单位：mm,红色为x轴,绿色为y轴,蓝色为z轴
        viewer->addCoordinateSystem(100, "global");
        // 创建一个自定义颜色处理器，将点云颜色设置为红色
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> colorHandler1(pcl_cloud, 255, 0, 0); // 红
        // 添加带颜色的点云到viewer中，命名为"colored cloud"
        viewer->addPointCloud<pcl::PointNormal>(pcl_cloud, colorHandler1, "point_cloud");
        // 设置点云的大小，单位：mm
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point_cloud");
        // 循环显示
        while (!viewer->wasStopped())
        {
            viewer->spinOnce(100);
        }
    }

}