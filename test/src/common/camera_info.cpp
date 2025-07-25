#include <iostream>

#include <yaml-cpp/yaml.h>

#include "common/camera_info.h"

namespace common
{
    CameraInfo CameraInfo::from_yaml(const std::string &yaml_path)
    {
        CameraInfo info;
        YAML::Node node = YAML::LoadFile(yaml_path);
        // 读取标量
        info.focal_length = node["focal_length"].as<double>();
        // 读取向量
        for (int i = 0; i < 2; ++i)
            info.sensor_size(i) = node["sensor_size"][i].as<double>();
        for (int i = 0; i < 3; ++i)
            info.cam_translation_vector(i) = node["cam_translation_vector"][i].as<double>();
        for (int i = 0; i < 4; ++i)
            info.cam_quaternions(i) = node["cam_quaternions"][i].as<double>();
        // 读取矩阵
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                info.extrinsic_matrix(r, c) = node["extrinsic_matrix"][r][c].as<double>();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                info.intrinsic_matrix(r, c) = node["intrinsic_matrix"][r][c].as<double>();
        return info;
    }
}