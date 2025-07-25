#include "common/point.h"

namespace common
{

    Point Point::from_pcl_point(const PclPoint &pcl_point)
    {
        return Point(pcl_point.x, pcl_point.y, pcl_point.z, pcl_point.normal_x, pcl_point.normal_y, pcl_point.normal_z);
    }

    Point Point::from_torch_tensor(const torch::Tensor &tensor)
    {
        return Point(tensor[0].item<double>(), tensor[1].item<double>(), tensor[2].item<double>(),
                     tensor[3].item<double>(), tensor[4].item<double>(), tensor[5].item<double>());
    }

    Point Point::from_eigen_point(const EigenPoint &eigen_point)
    {
        return Point(eigen_point(0), eigen_point(1), eigen_point(2), eigen_point(3), eigen_point(4), eigen_point(5));
    }

    Point::PclPoint Point::to_pcl_point() const
    {
        PclPoint pcl_point;
        pcl_point.x = x;
        pcl_point.y = y;
        pcl_point.z = z;
        pcl_point.normal_x = nx;
        pcl_point.normal_y = ny;
        pcl_point.normal_z = nz;
        return pcl_point;
    }

    torch::Tensor Point::to_torch_tensor() const
    {
        return torch::tensor({x, y, z, nx, ny, nz}, torch::dtype(torch::kDouble));
    }

    Point::EigenPoint Point::to_eigen_point() const
    {
        EigenPoint eigen_point;
        eigen_point(0) = x;
        eigen_point(1) = y;
        eigen_point(2) = z;
        eigen_point(3) = nx;
        eigen_point(4) = ny;
        eigen_point(5) = nz;
        return eigen_point;
    }
} // namespace common
