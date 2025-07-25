// #pragma once

// #include <Eigen/Dense>
// #include <memory>
// #include <vector>

// #include "dsnet_model_loader.h"
// #include "utils/utils.h"

// namespace pointnet2
// {

// /**
//  * @brief 最远点采样实现
//  */
// class FarthestPointSampling
// {
//    public:
//     /**
//      * @brief 使用FPS算法采样指定数量的点
//      * @param points 输入点云 [N, 3]
//      * @param num_samples 采样点数
//      * @return 采样点的索引
//      */
//     static std::vector<int> sample(const Eigen::MatrixXf& points, int num_samples);
// };

// /**
//  * @brief 球查询实现
//  */
// class BallQuery
// {
//    public:
//     /**
//      * @brief 在指定半径内查找邻域点
//      * @param query_points 查询点 [M, 3]
//      * @param support_points 支撑点 [N, 3]
//      * @param radius 球半径
//      * @param max_samples 每个球内最大采样点数
//      * @return 邻域点索引 [M, max_samples]
//      */
//     static std::vector<std::vector<int>> query(const Eigen::MatrixXf& query_points,
//                                                const Eigen::MatrixXf& support_points, float
//                                                radius, int max_samples);
// };

// /**
//  * @brief 三邻域插值实现
//  */
// class ThreeNN
// {
//    public:
//     /**
//      * @brief 查找三个最近邻并计算插值权重
//      * @param unknown_points 未知点 [M, 3]
//      * @param known_points 已知点 [N, 3]
//      * @return {distances [M, 3], indices [M, 3], weights [M, 3]}
//      */
//     static std::tuple<Eigen::MatrixXf, Eigen::MatrixXi, Eigen::MatrixXf> compute(
//         const Eigen::MatrixXf& unknown_points, const Eigen::MatrixXf& known_points);

//     /**
//      * @brief 使用三邻域插值传播特征
//      * @param features 已知点特征 [C, N]
//      * @param indices 邻域索引 [M, 3]
//      * @param weights 插值权重 [M, 3]
//      * @return 插值后的特征 [C, M]
//      */
//     static Eigen::MatrixXf interpolate(const Eigen::MatrixXf& features,
//                                        const Eigen::MatrixXi& indices,
//                                        const Eigen::MatrixXf& weights);
// };

// /**
//  * @brief MLP层实现
//  */
// class MLPLayer
// {
//    private:
//     std::vector<ConvWeights> conv_layers_;
//     std::vector<NormWeights> norm_layers_;
//     std::vector<bool> has_norm_;

//    public:
//     MLPLayer(const MLPWeights& weights);

//     /**
//      * @brief 前向传播
//      * @param input 输入特征 [C_in, N]
//      * @return 输出特征 [C_out, N]
//      */
//     Eigen::MatrixXf forward(const Eigen::MatrixXf& input) const;

//    private:
//     Eigen::MatrixXf applyConv1d(const Eigen::MatrixXf& input, const ConvWeights& conv) const;
//     Eigen::MatrixXf applyGroupNorm(const Eigen::MatrixXf& input, const NormWeights& norm) const;
//     Eigen::MatrixXf applyReLU(const Eigen::MatrixXf& input) const;
// };

// /**
//  * @brief Set Abstraction层实现
//  */
// class SetAbstractionLayer
// {
//    private:
//     int npoint_;
//     std::vector<float> radii_;
//     std::vector<int> nsamples_;
//     std::vector<MLPLayer> mlps_;
//     bool use_xyz_;

//    public:
//     SetAbstractionLayer(int npoint, const std::vector<float>& radii,
//                         const std::vector<int>& nsamples,
//                         const std::vector<MLPWeights>& mlp_weights, bool use_xyz = true);

//     /**
//      * @brief 前向传播
//      * @param points 输入点坐标 [N, 3]
//      * @param features 输入特征 [C, N] (可为空)
//      * @return {new_points [M, 3], new_features [C_out, M]}
//      */
//     std::pair<Eigen::MatrixXf, Eigen::MatrixXf> forward(
//         const Eigen::MatrixXf& points, const Eigen::MatrixXf& features = Eigen::MatrixXf());

//    private:
//     Eigen::MatrixXf groupAndPool(const Eigen::MatrixXf& points, const Eigen::MatrixXf&
//     new_points,
//                                  const Eigen::MatrixXf& features, float radius, int nsample,
//                                  const MLPLayer& mlp);
// };

// /**
//  * @brief Feature Propagation层实现
//  */
// class FeaturePropagationLayer
// {
//    private:
//     MLPLayer mlp_;

//    public:
//     FeaturePropagationLayer(const MLPWeights& mlp_weights);

//     /**
//      * @brief 前向传播
//      * @param unknown_points 未知点坐标 [M, 3]
//      * @param known_points 已知点坐标 [N, 3]
//      * @param unknown_features 未知点特征 [C1, M] (可为空)
//      * @param known_features 已知点特征 [C2, N]
//      * @return 传播后的特征 [C_out, M]
//      */
//     Eigen::MatrixXf forward(const Eigen::MatrixXf& unknown_points,
//                             const Eigen::MatrixXf& known_points,
//                             const Eigen::MatrixXf& unknown_features,
//                             const Eigen::MatrixXf& known_features);
// };

// /**
//  * @brief PointNet++骨干网络
//  */
// class PointNetBackbone
// {
//    private:
//     std::vector<SetAbstractionLayer> sa_layers_;
//     std::vector<FeaturePropagationLayer> fp_layers_;
//     ModelWeights::Config config_;

//    public:
//     PointNetBackbone(const PointNetWeights& weights, const ModelWeights::Config& config);

//     /**
//      * @brief 前向传播
//      * @param input_points 输入点云 [N, 6] (x,y,z,nx,ny,nz)
//      * @return {point_features [128, N], global_features [128, 1]}
//      */
//     std::pair<Eigen::MatrixXf, Eigen::MatrixXf> forward(const Eigen::MatrixXf& input_points);

//    private:
//     void buildNetwork(const PointNetWeights& weights);
// };

// }  // namespace pointnet2
