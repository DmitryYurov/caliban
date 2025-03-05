#ifndef CALIBAN_CONSTANTS_H
#define CALIBAN_CONSTANTS_H

#include <cstddef>

namespace caliban {

constexpr size_t n_r2 = 2;        // dimensionality of R^2 space
constexpr size_t n_r3 = 3;        // dimensionality of R^3 space
constexpr size_t n_so3 = 3;       // dimensionality of SO(3) space
constexpr size_t n_quat_so3 = 4;  // dimensionality of SO(3) space with quaternion representation (1 dof is redundant)
constexpr size_t n_se3 = 6;       // dimensionality of SE(3) = SO(3) x R^3 space
constexpr size_t n_quat_se3 = 7;  // dimensionality of SE(3) with quaternion representation (1 dof is redundant)
constexpr size_t n_cam = 4;       // the number of camera matrix parameters (fx, cx, fy, cy)
constexpr size_t n_dist = 5;      // the number of distortion coefficients (k1, k2, p1, p2, k3)

}  // namespace caliban

#endif  // CALIBAN_CONSTANTS_H
