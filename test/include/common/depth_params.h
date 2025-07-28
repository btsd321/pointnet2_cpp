#pragma once
#include <string>

namespace common
{
    class DepthParams
    {
    public:
        std::string name;        // 名称
        double clip_start;       // 裁剪起始，单位：m
        double clip_end;         // 裁剪结束，单位：m
        double max_val_in_depth; // 深度最大值(65535)
        int num_input_points;    // 输入模型的点数

        DepthParams() = default;
        static DepthParams from_json(const std::string &json_path);
    };
}
