#ifndef CALIBAN_TYPES_H
#define CALIBAN_TYPES_H

#include <array>

#include "constants.h"

namespace caliban {

using CameraMatrix = std::array<double, n_cam>;
using Point3D = std::array<double, n_r3>;
using Point2D = std::array<double, n_r2>;
using DistortionCoefficients = std::array<double, n_dist>;
using QuatSE3 = std::array<double, n_quat_se3>;

}  // namespace caliban

#endif  // CALIBAN_TYPES_H
