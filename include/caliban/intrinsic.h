#include <array>
#include <map>
#include <vector>

namespace caliban {
    constexpr size_t n_r2       = 2;  // dimensionality of R^2 space
    constexpr size_t n_r3       = 3;  // dimensionality of R^3 space
    constexpr size_t n_so3      = 3;  // dimensionality of SO(3) space
    constexpr size_t n_se3      = 6;  // dimensionality of SE(3) = SO(3) x R^3 space
    constexpr size_t n_quat_se3 = 7;  // dimensionality of SE(3) with quaternion representation (1 dof is redundant)
    constexpr size_t n_cam      = 4;  // the number of camera matrix parameters (fx, cx, fy, cy)
    constexpr size_t n_dist     = 5;  // the number of distortion coefficients (k1, k2, p1, p2, k3)

    using CameraMatrix = std::array<double, n_cam>;
    using Point3D = std::array<double, n_r3>;
    using Point2D = std::array<double, n_r2>;
    using DistortionCoefficients = std::array<double, n_dist>;
    using QuatSE3 = std::array<double, n_quat_se3>;
    void calibrate(
        std::vector<Point3D>& target_points,
        const std::vector<std::map<size_t, Point2D>>& image_points,
        CameraMatrix& camera_matrix,
        DistortionCoefficients& distortion_coefficients,
        std::vector<QuatSE3> obj_2_cams
    );
}