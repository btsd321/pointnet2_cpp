#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include "common/depth_img.h"

namespace common
{
    DepthImg::DepthImg(cv::Mat mat, DepthParams params, DepthImgScale scale)
        : cv::Mat(mat), params(params), scale(scale)
    {
    }

    DepthImg DepthImg::from_png(std::string path, DepthParams params)
    {
        cv::Mat mat;
        try
        {
            mat = cv::imread(path, cv::IMREAD_UNCHANGED);
            if (mat.empty())
            {
                throw std::runtime_error("Failed to read depth image from " + path);
            }
            if (mat.type() != CV_16UC1)
            {
                throw std::runtime_error("Depth image should be CV_16UC1 type");
            }
            if (mat.channels() != 1)
            {
                throw std::runtime_error("Depth image should be single channel");
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }

        // 如果mat的最大值是65535，则说明打开的是归一化深度图
        // 遍历mat计算最大值
        double max_val = cv::Mat(mat).at<uint16_t>(0, 0);

        for (int i = 0; i < mat.rows; ++i)
        {
            for (int j = 0; j < mat.cols; ++j)
            {
                if (mat.at<uint16_t>(i, j) > max_val)
                {
                    max_val = mat.at<uint16_t>(i, j);
                }
                if (max_val == std::numeric_limits<uint16_t>::max())
                {
                    // 如果最大值是65535，则说明是归一化深度图
                    return DepthImg::from_png(path, params, DepthImgScale::SCALE_NORMALIZED);
                }
            }
        }

        return DepthImg::from_png(path, params, DepthImgScale::SCALE_MM);
    }

    DepthImg DepthImg::from_png(std::string path, DepthParams params, DepthImgScale scale)
    {
        cv::Mat mat;
        try
        {
            mat = cv::imread(path, cv::IMREAD_ANYDEPTH);
            if (mat.empty())
            {
                throw std::runtime_error("Failed to read depth image from " + path);
            }
            if (mat.type() != CV_16UC1)
            {
                throw std::runtime_error("Depth image should be CV_16UC1 type");
            }
            if (mat.channels() != 1)
            {
                throw std::runtime_error("Depth image should be single channel");
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }

        return DepthImg(mat, params, scale);
    }

    void DepthImg::change_scale(DepthImgScale new_scale)
    {
        if (this->scale == new_scale)
        {
            return; // 如果新旧缩放比例相同，直接返回当前对象
        }
        cv::Mat scaled_mat;
        if (new_scale == DepthImgScale::SCALE_MM)
        {
            // 根据params的clip_start和clip_end进行缩放
            auto clip_start = this->params.clip_start * 1000; // 转换为毫米
            auto clip_end = this->params.clip_end * 1000;     // 转换为毫米
            cv::normalize(*this, scaled_mat, clip_start, clip_end, cv::NORM_MINMAX, CV_16UC1);
        }
        else if (new_scale == DepthImgScale::SCALE_NORMALIZED)
        {
            // mm转归一化深度图
            cv::normalize(*this, scaled_mat, 0, std::numeric_limits<uint16_t>::max(), cv::NORM_MINMAX, CV_16UC1);
        }
        *this = DepthImg(scaled_mat, this->params, new_scale);
    }

    PointCloud DepthImg::create_point_cloud(const CameraInfo &camera_info, const cv::Mat &mask)
    {
        PointCloud output_cloud;
        const auto &fx = camera_info.intrinsic_matrix(0, 0);
        const auto &fy = camera_info.intrinsic_matrix(1, 1);
        const auto &cx = camera_info.intrinsic_matrix(0, 2);
        const auto &cy = camera_info.intrinsic_matrix(1, 2);

        this->change_scale(DepthImgScale::SCALE_MM); // 确保格式为mm

        // 统一掩码为8位
        cv::Mat mask_bin;
        if (mask.type() == CV_8UC1)
        {
            mask_bin = mask;
        }
        else if (mask.type() == CV_16UC1)
        {
            mask.convertTo(mask_bin, CV_8U, 1.0 / 256.0); // 简单归一化
        }
        else
        {
            throw std::runtime_error("Unsupported mask type for point cloud generation");
        }

        std::vector<cv::Point> nonzero_pts;
        cv::findNonZero(mask_bin, nonzero_pts);
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        for (const auto &pt : nonzero_pts)
        {
            int i = pt.y, j = pt.x;
            double depth = this->at<uint16_t>(i, j) / 1000.0; // 转换为米
            common::Point point;
            point.x = (j - cx) * depth / fx;
            point.y = (i - cy) * depth / fy;
            point.z = depth;
            output_cloud.points.emplace_back(point);
            pcl_cloud->points.emplace_back(pcl::PointXYZ(point.x, point.y, point.z));
        }
        output_cloud.set_reference_frame(PointCloud::ReferenceFrame::CAMERA);

        // 计算法向量并归一化
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(pcl_cloud);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(0.015); // 邻域半径，单位：m，和训练保持一致，不能随意更改
        pcl::PointCloud<pcl::Normal>::Ptr normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
        ne.compute(*normals);
        // 统一法向量方向：都指向负Z轴方向（向下）
        for (int i = 0; i < normals->points.size(); ++i)
        {
            auto &n = normals->points[i];
            if (n.normal_z > 0)
            {
                n.normal_x = -n.normal_x;
                n.normal_y = -n.normal_y;
                n.normal_z = -n.normal_z;
            }

            auto length = std::sqrt(n.normal_x * n.normal_x + n.normal_y * n.normal_y + n.normal_z * n.normal_z);
            // n.normal_x /= length;
            // n.normal_y /= length;
            // n.normal_z /= length;
            output_cloud.points.at(i).nx = n.normal_x / length;
            output_cloud.points.at(i).ny = n.normal_y / length;
            output_cloud.points.at(i).nz = n.normal_z / length;
        }

        return output_cloud;
    }
} // namespace common