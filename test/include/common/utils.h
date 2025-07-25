#pragma once

#include <filesystem>

#include <torch/torch.h>

#include "common/camera_info.h"
#include "common/depth_img.h"

namespace common
{
    typedef struct InferenceInputStruct
    {
        std::filesystem::path checkpoint_path;  // 模型检查点路径
        std::filesystem::path rgb_img_path;     // 输入图像路径
        std::filesystem::path depth_img_path;   // 输入深度图路径
        std::filesystem::path mask_img_path;    // 输入掩膜图路径
        std::filesystem::path camera_info_path; // 相机信息路径
        std::filesystem::path params_path;      // 参数文件路径
        bool use_cuda;                          // 是否使用CUDA
        bool save_result;                       // 是否保存结果
        std::filesystem::path output_dir;       // 输出路径
    } InferenceInput;

    typedef struct ModelInputStruct
    {
        torch::Tensor points;             // 点云数据（[B, N, 6]）
        bool use_cuda;                    // 是否使用CUDA
        bool save_result;                 // 是否保存结果
        std::filesystem::path output_dir; // 输出路径
    } ModelInput;

    /*
     * @brief 推理输入预处理，输出模型输入格式
     *
     * 对输入的推理输入进行预处理，输出模型输入格式。
     *
     * @param inference_input (InferenceInput) 推理输入
     * @return (ModelInput) 模型输入
     */
    void preprocess(InferenceInput inference_input, ModelInput &model_input);
}