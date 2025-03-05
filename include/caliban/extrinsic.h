#ifndef CALIBAN_EXTRINSIC_H
#define CALIBAN_EXTRINSIC_H

#include "caliban/types.h"

#include <map>
#include <vector>

namespace caliban {

enum ExtrinsicCalibType { EyeInHand = 0, EyeToHand = 1 };

enum ExtrinsicFlags {
    None = 0,
    OptimizeScale = 1,
};

/**
 * @brief Refine the extrinsic calibration of the camera with respect to the robot
 *
 * Input parameters correspond to the AX = BZ problem. Depending on the extrinsic calibration type,
 * camera is either mounted in a constant position wrt to the robot base (EyeToHand) or it's mounted
 * on the robot end-effector (EyeInHand). The equation solved stays the same, but the computation of the
 * minimized reprojection error is different.
 *
 * @param type The type of the extrinsic calibration (EyeInHand or EyeToHand)
 * @param target_points The 3D points in the world coordinate system. In case the flag OptimizeScale is set,
 *                      the scale of the target points will be optimized as well.
 * @param image_points The 2D points in the image coordinate system. The map key corresponds to the index of the target
 * point.
 * @param Bs The base-to-flange SE3 transformations
 * @param X Input/Output parameter for the base-to-target SE3 transformation (eye in hand) or the base-to camera SE3
 * transformation (eye to hand)
 * @param Z Input/Output parameter for the flange-to-camera SE3 transformation (eye in hand) or to the target (eye to
 * hand)
 * @param camera_matrix The camera matrix
 * @param distortion_coefficients The distortion coefficients
 * @param flags Flags for the optimization
 * @return A tuple with the optimized base-to-target and flange-to-camera transformations
 */
double calibrate_extrinsics(ExtrinsicCalibType type,
                            std::vector<Point3D>& target_points,
                            const std::vector<std::map<size_t, Point2D>>& image_points,
                            const std::vector<QuatSE3>& Bs,
                            QuatSE3& X,
                            QuatSE3& Z,
                            const CameraMatrix& camera_matrix,
                            const DistortionCoefficients& distortion_coefficients,
                            int flags = ExtrinsicFlags::None);

}  // namespace caliban

#endif  // CALIBAN_EXTRINSIC_H
