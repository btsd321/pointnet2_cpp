#include <argparse/argparse.hpp>
#include "common/utils.h"
#include "test.h"

namespace test
{
    static void input_pack(int argc, char *argv[], common::InferenceInput &input)
    {
        argparse::ArgumentParser args("test");
        args.add_argument("--checkpoint_path")
            .help("Path to the checkpoint file")
            .required();
        args.add_argument("--rgb_img_path")
            .help("Path to the RGB image file")
            .required();
        args.add_argument("--depth_img_path")
            .help("Path to the depth image file")
            .required();
        args.add_argument("--mask_img_path")
            .help("Path to the mask image file")
            .required();
        args.add_argument("--camera_info_path")
            .help("Path to the camera info file")
            .required();
        args.add_argument("--params_path")
            .help("Path to the parameters file")
            .required();
        args.add_argument("--output_dir")
            .help("Path to the output directory")
            .required();
        args.add_argument("--use_cuda")
            .help("Use CUDA for inference")
            .default_value(true)
            .implicit_value(true);
        args.add_argument("--save_result")
            .help("Save the inference result")
            .default_value(false)
            .implicit_value(true);

        try
        {
            args.parse_args(argc, argv);
        }
        catch (const std::exception &err)
        {
            std::cerr << err.what() << std::endl;
            std::cerr << args;
            throw err;
        }

        try
        {
            input.checkpoint_path = std::filesystem::path(args.get<std::string>("checkpoint_path"));
            input.rgb_img_path = std::filesystem::path(args.get<std::string>("rgb_img_path"));
            input.depth_img_path = std::filesystem::path(args.get<std::string>("depth_img_path"));
            input.mask_img_path = std::filesystem::path(args.get<std::string>("mask_img_path"));
            input.camera_info_path = std::filesystem::path(args.get<std::string>("camera_info_path"));
            input.params_path = std::filesystem::path(args.get<std::string>("params_path"));
            input.use_cuda = args.is_used("use_cuda");
            input.save_result = args.is_used("save_result");
            input.output_dir = std::filesystem::path(args.get<std::string>("output_dir"));
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
            throw std::runtime_error("Failed to parse input arguments");
            return;
        }

        // 判断所有输入文件是否存在，不存在则报错
        if (!std::filesystem::exists(input.checkpoint_path))
        {
            throw std::runtime_error("Checkpoint file does not exist: " + input.checkpoint_path.string());
        }
        if (!std::filesystem::exists(input.rgb_img_path))
        {
            throw std::runtime_error("RGB image file does not exist: " + input.rgb_img_path.string());
        }
        if (!std::filesystem::exists(input.depth_img_path))
        {
            throw std::runtime_error("Depth image file does not exist: " + input.depth_img_path.string());
        }
        if (!std::filesystem::exists(input.mask_img_path))
        {
            throw std::runtime_error("Mask image file does not exist: " + input.mask_img_path.string());
        }
        if (!std::filesystem::exists(input.camera_info_path))
        {
            throw std::runtime_error("Camera info file does not exist: " + input.camera_info_path.string());
        }
        if (!std::filesystem::exists(input.params_path))
        {
            throw std::runtime_error("Parameters file does not exist: " + input.params_path.string());
        }
        // 输出目录不存在则新建
        if (!std::filesystem::exists(input.output_dir))
        {
            std::filesystem::create_directories(input.output_dir);
            std::cout << "Output directory created: " << input.output_dir.string() << std::endl;
        }
        else if (!std::filesystem::is_directory(input.output_dir))
        {
            throw std::runtime_error("Output path is not a directory: " + input.output_dir.string());
        }

        return;
    }
}

int main(int argc, char *argv[])
{
    common::InferenceInput inference_input;
    common::ModelInput model_input;
    try
    {
        test::input_pack(argc, argv, inference_input);
        common::preprocess(inference_input, model_input);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        exit(1);
    }
    return 0;
}