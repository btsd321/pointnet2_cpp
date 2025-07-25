#pragma once

#include <filesystem>

#include <opencv2/opencv.hpp>

#include "common/depth_params.h"
#include "common/point_cloud.h"
#include "common/camera_info.h"

namespace common
{
    enum class DepthImgScale
    {
        // 单位：毫米
        SCALE_MM = 0,
        // 归一化单位
        SCALE_NORMALIZED
    };

    class DepthImg : public cv::Mat
    {
    public:
        DepthImg() = delete; // 禁止默认构造函数

        DepthImgScale scale;
        DepthParams params;

        static DepthImg from_png(std::string path, DepthParams params); // 根据最大值自动判断
        static DepthImg from_png(std::string path, DepthParams params, DepthImgScale scale);

        void change_scale(DepthImgScale new_scale);
        PointCloud create_point_cloud(const CameraInfo &camera_info, const cv::Mat &mask);

    protected:
        DepthImg(cv::Mat mat, DepthParams params, DepthImgScale scale) : cv::Mat(mat), params(params), scale(scale) {}
    };
} // namespace common