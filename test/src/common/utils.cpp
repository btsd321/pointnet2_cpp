#include "common/utils.h"

namespace common
{

    /*
     * @brief 推理输入预处理，输出模型输入格式
     *
     * 对输入的推理输入进行预处理，输出模型输入格式。
     *
     * @param inference_input (InferenceInput) 推理输入
     * @return (ModelInput) 模型输入
     */
    void preprocess(InferenceInput inference_input, ModelInput &model_input)
    {
        try
        {
            auto rgb_img = cv::imread(inference_input.rgb_img_path.string(), cv::IMREAD_COLOR);
            auto camera_info = CameraInfo::from_yaml(inference_input.camera_info_path.string());
            auto depth_params = DepthParams::from_json(inference_input.params_path.string());
            auto mask_img = cv::imread(inference_input.mask_img_path.string(), cv::IMREAD_GRAYSCALE);
            auto depth_img = DepthImg::from_png(inference_input.depth_img_path.string(), depth_params, DepthImgScale::SCALE_NORMALIZED);

            // mask图中白色为物体其余为背景
            // 计算遮罩
            auto valid_mask = (mask_img == 255);
            // 计算点云
            auto point_cloud = depth_img.create_point_cloud(camera_info, valid_mask);
            point_cloud.display(); // 可视化点云
            model_input.points = point_cloud.to_torchcloud();
            model_input.use_cuda = inference_input.use_cuda;
            model_input.save_result = inference_input.save_result;
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            throw std::runtime_error("Failed to preprocess inference input");
        }
    }
}