#ifndef CALIBAN_INTRINSIC_H
#define CALIBAN_INTRINSIC_H

#include <array>
#include <map>
#include <vector>

#include "caliban/constants.h"
#include "caliban/types.h"

namespace caliban {

double calibrate_intrinsics(
    std::vector<Point3D>& target_points,
    const std::vector<std::map<size_t, Point2D>>& image_points,
    CameraMatrix& camera_matrix,
    DistortionCoefficients& distortion_coefficients,
    std::vector<QuatSE3> obj_2_cams
);

}  // namespace caliban

#endif // CALIBAN_INTRINSIC_H
