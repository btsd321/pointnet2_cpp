
// #include "pointnet2/pointnet_backbone.h"

// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <numeric>
// #include <random>

// namespace pointnet2
// {

// // 计算两点之间的欧几里得距离平方
// inline static float distance_squared(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2)
// {
//     return (p1 - p2).squaredNorm();
// }

// // ================= FarthestPointSampling =================
// std::vector<int> FarthestPointSampling::sample(const Eigen::MatrixXf& points, int num_samples)
// {
//     int num_points = points.cols();
//     if (num_samples >= num_points)
//     {
//         std::vector<int> indices(num_points);
//         std::iota(indices.begin(), indices.end(), 0);
//         return indices;
//     }

//     std::vector<int> sampled_indices;
//     std::vector<float> min_distances(num_points, std::numeric_limits<float>::max());

//     // 第一个点随机选择
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_int_distribution<> dis(0, num_points - 1);
//     int first_idx = dis(gen);
//     sampled_indices.push_back(first_idx);

//     // 更新到第一个点的距离
//     for (int i = 0; i < num_points; ++i)
//     {
//         if (i != first_idx)
//         {
//             Eigen::Vector3f p1 = points.block<3, 1>(0, first_idx);
//             Eigen::Vector3f p2 = points.block<3, 1>(0, i);
//             min_distances[i] = distance_squared(p1, p2);
//         }
//     }
//     min_distances[first_idx] = 0.0f;

//     // 迭代选择最远点
//     for (int s = 1; s < num_samples; ++s)
//     {
//         int farthest_idx = -1;
//         float max_min_distance = -1.0f;
//         for (int i = 0; i < num_points; ++i)
//         {
//             if (min_distances[i] > max_min_distance)
//             {
//                 max_min_distance = min_distances[i];
//                 farthest_idx = i;
//             }
//         }
//         sampled_indices.push_back(farthest_idx);

//         // 更新最小距离
//         Eigen::Vector3f new_point = points.block<3, 1>(0, farthest_idx);
//         for (int i = 0; i < num_points; ++i)
//         {
//             if (min_distances[i] > 0)
//             {
//                 Eigen::Vector3f curr_point = points.block<3, 1>(0, i);
//                 float dist_to_new = distance_squared(curr_point, new_point);
//                 min_distances[i] = std::min(min_distances[i], dist_to_new);
//             }
//         }
//         min_distances[farthest_idx] = 0.0f;
//     }
//     return sampled_indices;
// }

// // ================= BallQuery =================
// std::vector<std::vector<int>> BallQuery::query(const Eigen::MatrixXf& query_points,
//                                                const Eigen::MatrixXf& support_points, float
//                                                radius, int max_samples)
// {
//     int num_query_points = query_points.cols();
//     int num_support_points = support_points.cols();
//     float radius_squared = radius * radius;
//     std::vector<std::vector<int>> grouped_indices(num_query_points);

//     for (int i = 0; i < num_query_points; ++i)
//     {
//         Eigen::Vector3f query_point = query_points.block<3, 1>(0, i);
//         std::vector<int> neighbors;
//         for (int j = 0; j < num_support_points; ++j)
//         {
//             Eigen::Vector3f support_point = support_points.block<3, 1>(0, j);
//             float dist_sq = distance_squared(query_point, support_point);
//             if (dist_sq <= radius_squared)
//             {
//                 neighbors.push_back(j);
//                 if ((int)neighbors.size() >= max_samples)
//                     break;
//             }
//         }
//         // 如果没有找到足够的邻居，用最近的点重复填充
//         if (neighbors.empty())
//         {
//             int closest_idx = 0;
//             float min_dist = std::numeric_limits<float>::max();
//             for (int j = 0; j < num_support_points; ++j)
//             {
//                 Eigen::Vector3f support_point = support_points.block<3, 1>(0, j);
//                 float dist_sq = distance_squared(query_point, support_point);
//                 if (dist_sq < min_dist)
//                 {
//                     min_dist = dist_sq;
//                     closest_idx = j;
//                 }
//             }
//             neighbors.push_back(closest_idx);
//         }
//         // 如果邻居数量不足，重复最后一个邻居
//         while ((int)neighbors.size() < max_samples)
//         {
//             neighbors.push_back(neighbors.back());
//         }
//         grouped_indices[i] = std::move(neighbors);
//     }
//     return grouped_indices;
// }

// // ================= MLPLayer =================
// MLPLayer::MLPLayer(const MLPWeights& weights)
// {
//     conv_layers_ = weights.conv_layers;
//     norm_layers_ = weights.norm_layers;
//     has_norm_ = weights.has_norm;
// }

// Eigen::MatrixXf MLPLayer::forward(const Eigen::MatrixXf& input) const
// {
//     Eigen::MatrixXf output = input;
//     for (size_t i = 0; i < conv_layers_.size(); ++i)
//     {
//         output = applyConv1d(output, conv_layers_[i]);
//         if (i < has_norm_.size() && has_norm_[i] && i < norm_layers_.size())
//         {
//             output = applyGroupNorm(output, norm_layers_[i]);
//         }
//         output = applyReLU(output);
//     }
//     return output;
// }

// Eigen::MatrixXf MLPLayer::applyConv1d(const Eigen::MatrixXf& input, const ConvWeights& conv)
// const
// {
//     Eigen::MatrixXf output = conv.weight * input;
//     if (conv.has_bias)
//     {
//         for (int i = 0; i < output.cols(); ++i)
//         {
//             output.col(i) += conv.bias;
//         }
//     }
//     return output;
// }

// Eigen::MatrixXf MLPLayer::applyGroupNorm(const Eigen::MatrixXf& input,
//                                          const NormWeights& norm) const
// {
//     Eigen::MatrixXf output = input;
//     for (int i = 0; i < output.rows(); ++i)
//     {
//         for (int j = 0; j < output.cols(); ++j)
//         {
//             float normalized =
//                 (output(i, j) - norm.running_mean(i)) / std::sqrt(norm.running_var(i) +
//                 norm.eps);
//             output(i, j) = normalized * norm.weight(i) + norm.bias(i);
//         }
//     }
//     return output;
// }

// Eigen::MatrixXf MLPLayer::applyReLU(const Eigen::MatrixXf& input) const
// {
//     return input.cwiseMax(0.0f);
// }

// // ================= SetAbstractionLayer =================
// SetAbstractionLayer::SetAbstractionLayer(int npoint, const std::vector<float>& radii,
//                                          const std::vector<int>& nsamples,
//                                          const std::vector<MLPWeights>& mlp_weights, bool
//                                          use_xyz)
//     : npoint_(npoint), radii_(radii), nsamples_(nsamples), use_xyz_(use_xyz)
// {
//     for (const auto& weights : mlp_weights)
//     {
//         mlps_.emplace_back(weights);
//     }
// }

// std::pair<Eigen::MatrixXf, Eigen::MatrixXf> SetAbstractionLayer::forward(
//     const Eigen::MatrixXf& points, const Eigen::MatrixXf& features)
// {
//     // Step 1: 使用FPS进行下采样
//     std::vector<int> sampled_indices = FarthestPointSampling::sample(points, npoint_);
//     // 提取采样后的坐标
//     Eigen::MatrixXf new_points(3, sampled_indices.size());
//     for (size_t i = 0; i < sampled_indices.size(); ++i)
//     {
//         new_points.col(i) = points.col(sampled_indices[i]);
//     }
//     // Step 2: 多尺度特征提取
//     std::vector<Eigen::MatrixXf> multi_scale_features;
//     for (size_t r = 0; r < radii_.size() && r < nsamples_.size() && r < mlps_.size(); ++r)
//     {
//         Eigen::MatrixXf scale_features =
//             groupAndPool(points, new_points, features, radii_[r], nsamples_[r], mlps_[r]);
//         multi_scale_features.push_back(scale_features);
//     }
//     // Step 3: 连接多尺度特征
//     if (multi_scale_features.empty())
//     {
//         return {new_points, Eigen::MatrixXf::Zero(0, sampled_indices.size())};
//     }
//     int total_dim = 0;
//     for (const auto& feat : multi_scale_features)
//     {
//         total_dim += feat.rows();
//     }
//     Eigen::MatrixXf concatenated_features(total_dim, sampled_indices.size());
//     int current_row = 0;
//     for (const auto& feat : multi_scale_features)
//     {
//         concatenated_features.block(current_row, 0, feat.rows(), feat.cols()) = feat;
//         current_row += feat.rows();
//     }
//     return {new_points, concatenated_features};
// }

// Eigen::MatrixXf SetAbstractionLayer::groupAndPool(const Eigen::MatrixXf& points,
//                                                   const Eigen::MatrixXf& new_points,
//                                                   const Eigen::MatrixXf& features, float radius,
//                                                   int nsample, const MLPLayer& mlp)
// {
//     // 球查询
//     auto grouped_indices = BallQuery::query(new_points, points, radius, nsample);
//     // 提取分组特征
//     int xyz_dim = use_xyz_ ? 3 : 0;
//     int feature_dim = features.rows();
//     int total_dim = xyz_dim + feature_dim;
//     Eigen::MatrixXf grouped_features(total_dim, new_points.cols() * nsample);
//     for (size_t i = 0; i < grouped_indices.size(); ++i)
//     {
//         Eigen::Vector3f center = new_points.col(i);
//         for (size_t j = 0; j < grouped_indices[i].size(); ++j)
//         {
//             int neighbor_idx = grouped_indices[i][j];
//             int output_idx = i * nsample + j;
//             int current_dim = 0;
//             // 相对坐标 (xyz - center)
//             if (use_xyz_)
//             {
//                 Eigen::Vector3f relative_pos = points.col(neighbor_idx) - center;
//                 grouped_features.block<3, 1>(current_dim, output_idx) = relative_pos;
//                 current_dim += 3;
//             }
//             // 原始特征
//             if (feature_dim > 0)
//             {
//                 grouped_features.block(current_dim, output_idx, feature_dim, 1) =
//                     features.col(neighbor_idx);
//             }
//         }
//     }
//     // 通过MLP处理特征
//     Eigen::MatrixXf processed_features = mlp.forward(grouped_features);
//     // 最大池化：在每个采样点的邻域内进行最大池化
//     int output_dim = processed_features.rows();
//     Eigen::MatrixXf pooled_features = Eigen::MatrixXf::Zero(output_dim, new_points.cols());
//     for (int i = 0; i < new_points.cols(); ++i)
//     {
//         for (int j = 0; j < nsample; ++j)
//         {
//             int input_idx = i * nsample + j;
//             pooled_features.col(i) =
//                 pooled_features.col(i).cwiseMax(processed_features.col(input_idx));
//         }
//     }
//     return pooled_features;
// }

// // ================= FeaturePropagationLayer =================
// FeaturePropagationLayer::FeaturePropagationLayer(const MLPWeights& mlp_weights) :
// mlp_(mlp_weights)
// {
// }

// Eigen::MatrixXf FeaturePropagationLayer::forward(const Eigen::MatrixXf& points1,
//                                                  const Eigen::MatrixXf& points2,
//                                                  const Eigen::MatrixXf& features1,
//                                                  const Eigen::MatrixXf& features2)
// {
//     if (points2.cols() == 0)
//     {
//         return features1;
//     }
//     // 使用距离加权插值进行特征传播
//     Eigen::MatrixXf interpolated_features = Eigen::MatrixXf::Zero(features2.rows(),
//     points1.cols()); for (int i = 0; i < points1.cols(); ++i)
//     {
//         Eigen::Vector3f query_point = points1.col(i);
//         // 找到k个最近邻 (k=3)
//         const int k = 3;
//         std::vector<std::pair<float, int>> distances;
//         for (int j = 0; j < points2.cols(); ++j)
//         {
//             Eigen::Vector3f ref_point = points2.col(j);
//             float dist = distance_squared(query_point, ref_point);
//             distances.push_back({dist, j});
//         }
//         std::partial_sort(distances.begin(), distances.begin() + std::min(k,
//         (int)points2.cols()),
//                           distances.end());
//         // 计算加权插值
//         float total_weight = 0.0f;
//         for (int ki = 0; ki < std::min(k, (int)points2.cols()); ++ki)
//         {
//             float dist = std::sqrt(distances[ki].first);
//             float weight = (dist < 1e-8f) ? 1.0f : 1.0f / (dist + 1e-8f);
//             int neighbor_idx = distances[ki].second;
//             interpolated_features.col(i) += weight * features2.col(neighbor_idx);
//             total_weight += weight;
//         }
//         if (total_weight > 0)
//         {
//             interpolated_features.col(i) /= total_weight;
//         }
//     }
//     // 如果有features1，将其与插值结果连接
//     if (features1.rows() > 0)
//     {
//         Eigen::MatrixXf concatenated(features1.rows() + features2.rows(), points1.cols());
//         concatenated.topRows(features1.rows()) = features1;
//         concatenated.bottomRows(features2.rows()) = interpolated_features;
//         // 通过MLP处理连接后的特征
//         return mlp_.forward(concatenated);
//     }
//     else
//     {
//         // 只处理插值特征
//         return mlp_.forward(interpolated_features);
//     }
// }

// }  // namespace pointnet2
