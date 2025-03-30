#include "jaco.h"

namespace caliban {

using namespace Eigen;

Matrix<double, n_se3 + n_r3, n_r3> get_transform_jaco(std::array<double, n_se3> transform,
                                                      std::array<double, n_r3> point) {
    Matrix<double, n_se3 + n_r3, n_r3> result{};
    result.setZero();

    auto p = Map<Vector3d>(point.data());
    Matrix3d rotation = AngleAxisd(transform[0], Vector3d::UnitX()).toRotationMatrix() *
                        AngleAxisd(transform[1], Vector3d::UnitY()).toRotationMatrix() *
                        AngleAxisd(transform[2], Vector3d::UnitZ()).toRotationMatrix();

    // computing the derivatives of the position with respect to Euler angles
    Matrix3d M = Matrix3d::Identity();
    M(0, 2) = std::sin(transform[1]);                            // sin(beta)
    M(1, 1) = std::cos(transform[0]);                            // cos(alpha)
    M(2, 1) = std::sin(transform[0]);                            // sin(alpha)
    M(1, 2) = -std::sin(transform[0]) * std::cos(transform[1]);  // -sin(alpha) * cos(beta)
    M(2, 2) = std::cos(transform[0]) * std::sin(transform[1]);   // cos(alpha) * sin(beta)

    const Vector3d rp = -rotation * p;
    const auto rp_skew = Matrix3d{
        {0, -rp(2), rp(1)},
        {rp(2), 0, -rp(0)},
        {-rp(1), rp(0), 0},
    };

    result.block<3, 3>(0, 0) = (-rp_skew * M).transpose();

    // computing the derivatives of the position with respect to translation
    result.block<3, 3>(3, 0) = Matrix3d::Identity();

    // computing the derivatives of the position with respect to the point
    result.block<3, 3>(6, 0) = rotation.transpose();

    return result;
}

}  // namespace caliban
