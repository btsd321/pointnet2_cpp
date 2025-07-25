#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace common
{

    class CameraInfo
    {
    public:
        double focal_length;                    // 焦距,单位：mm
        Eigen::Vector2d sensor_size;            // 传感器尺寸,单位：mm
        Eigen::Vector3d cam_translation_vector; // 相机相对于桌子中心的位移向量，单位：m
        Eigen::Vector4d cam_quaternions;        // 四元数旋转矩阵(格式xyzw)
        Eigen::Matrix4d extrinsic_matrix;       // 相机相对于桌子中心的4x4外参矩阵，单位：m
        Eigen::Matrix3d intrinsic_matrix;       // 3x3内参矩阵

        CameraInfo() = default;

        // 从yaml文件读取
        static CameraInfo from_yaml(const std::string &yaml_path);
    };
}