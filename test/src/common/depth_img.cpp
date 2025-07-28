#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include "common/depth_img.h"

namespace common
{
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
            // mask_bin = mask;
            // 二值化， 分割阈值为150，大于150的是物体，反之是背景
            cv::threshold(mask, mask_bin, 150, 255, cv::THRESH_BINARY);
        }
        // else if (mask.type() == CV_16UC1)
        // {
        //     mask.convertTo(mask_bin, CV_8U, 1.0 / 256.0); // 简单归一化
        // }
        else
        {
            throw std::runtime_error("Unsupported mask type for point cloud generation");
        }

        // 可视化mask_bin
        cv::imshow("mask_bin", mask_bin);
        cv::waitKey(0);
        // 计算mask_bin中最大值和最小值
        double max_val = cv::Mat(mask_bin).at<uint8_t>(0, 0);
        double min_val = cv::Mat(mask_bin).at<uint8_t>(0, 0);
        for (int i = 0; i < mask_bin.rows; ++i)
        {
            for (int j = 0; j < mask_bin.cols; ++j)
            {
                if (mask_bin.at<uint8_t>(i, j) > max_val)
                {
                    max_val = mask_bin.at<uint8_t>(i, j);
                }
                if (mask_bin.at<uint8_t>(i, j) < min_val)
                {
                    min_val = mask_bin.at<uint8_t>(i, j);
                }
            }
        }
        std::cout << "Mask min value: " << min_val << ", max value: " << max_val << std::endl;

        std::vector<cv::Point> nonzero_pts;
        cv::findNonZero(mask_bin, nonzero_pts);
        if (nonzero_pts.size() <= 100)
        {
            throw std::runtime_error("Too few points in mask for point cloud generation");
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::PointCloud<pcl::PointXYZ>::Ptr resampled_pcl_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
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

        // 点云统计滤波，去除离群点
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(pcl_cloud);
        sor.setMeanK(40);            // 计算平均值，去除离群点
        sor.setStddevMulThresh(1.0); // 计算标准差，去除离群点
        sor.filter(*filtered_pcl_cloud);

        // 计算法向量并归一化
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(filtered_pcl_cloud);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
        ne.setSearchMethod(tree);
        ne.setRadiusSearch(0.015); // 邻域半径，单位：m，和训练保持一致，不能随意更改
        pcl::PointCloud<pcl::Normal>::Ptr normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
        pcl::PointCloud<pcl::Normal>::Ptr resampled_normals = std::make_shared<pcl::PointCloud<pcl::Normal>>();
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
            n.normal_x /= length;
            n.normal_y /= length;
            n.normal_z /= length;
            // output_cloud.points.at(i).nx = n.normal_x / length;
            // output_cloud.points.at(i).ny = n.normal_y / length;
            // output_cloud.points.at(i).nz = n.normal_z / length;
        }

        // 点云重采样
        if (filtered_pcl_cloud->size() <= 100)
        {
            throw std::runtime_error("Too few points after filtering for point cloud generation");
        }
        else if (filtered_pcl_cloud->size() < this->params.num_input_points)
        {
            // 需要上采样，直接重复点云
            *resampled_pcl_cloud = *filtered_pcl_cloud;
            *resampled_normals = *normals;
            int num_repeat = this->params.num_input_points / filtered_pcl_cloud->size() - 1;
            int num_extra = this->params.num_input_points % filtered_pcl_cloud->size();
            while (num_repeat)
            {
                num_repeat--;
                resampled_pcl_cloud->insert(resampled_pcl_cloud->end(), filtered_pcl_cloud->begin(), filtered_pcl_cloud->end());
                resampled_normals->insert(resampled_normals->end(), normals->begin(), normals->end());
            }
            resampled_pcl_cloud->insert(resampled_pcl_cloud->end(), filtered_pcl_cloud->begin(), filtered_pcl_cloud->begin() + num_extra);
            resampled_normals->insert(resampled_normals->end(), normals->begin(), normals->begin() + num_extra);
        }
        else if (filtered_pcl_cloud->size() == this->params.num_input_points)
        {
        }
        else //(filtered_pcl_cloud->size() > this->params.num_input_points)
        {
            // 需要下采样 TODO
        }

        output_cloud.set_reference_frame(PointCloud::ReferenceFrame::CAMERA);
        return output_cloud;
    }
} // namespace common