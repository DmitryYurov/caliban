#ifndef CALIBAN_EXTRINSIC_H
#define CALIBAN_EXTRINSIC_H

#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>

#include <map>
#include <vector>

namespace caliban {

enum ExtrinsicCalibType { EyeInHand = 0, EyeToHand = 1 };

enum ExtrinsicFlags {
    None = 0,
    OptimizeScale = 1,
};

/**
 * @brief The result of the extrinsic calibration
 *
 * Notation corresponds to the equation AX = BZ solved during the optimization. X denotes the base-to-target
 * transformation (eye in hand) or the base-to-camera transformation (eye to hand). Z denotes the transformation from
 * the flange to the camera (eye in hand) or to the target (eye to hand).
 */
struct ExtrinsicResult {
    ExtrinsicCalibType calib_type;  ///< The type of the extrinsic calibration (just for user convenience)
    double rms_repro;               ///< The root mean square reprojection error
    double scale;                   ///< The scale of the target points
    cv::Quat<double> X_rot;         ///< Rotational part of X transformation
    cv::Vec<double, 3> X_tvec;      ///< Translational part of X transformation
    cv::Quat<double> Z_rot;         ///< Rotational part of Z transformation
    cv::Vec<double, 3> Z_tvec;      ///< Translational part of Z transformation
};

/**
 * @brief Refine the extrinsic calibration of the camera with respect to the robot
 *
 * Input parameters correspond to the AX = BZ problem. Depending on the extrinsic calibration type,
 * camera is either mounted in a constant position wrt to the robot base (EyeToHand) or it's mounted
 * on the robot end-effector (EyeInHand). The equation solved stays the same, but the computation of the
 * minimized reprojection error is slightly different.
 *
 * @param calib_type The type of the extrinsic calibration (EyeInHand or EyeToHand)
 * @param target_points_cv The 3D points in the world coordinate system. In case the flag OptimizeScale is set,
 *                         the scale of the target points will be optimized.
 * @param image_points_cv Detected 2D points. The map key corresponds to the index of the target point.
 * @param B_rvecs The base-to-flange SE3 transformations (rotational part)
 * @param B_tvecs The base-to-flange SE3 transformations (translational part)
 * @param tar_2_cam_rvecs The target-to-camera SE3 transformations (rotational part)
 * @param tar_2_cam_tvecs The target-to-camera SE3 transformations (translational part)
 * @param camera_matrix_cv The camera matrix
 * @param distort_coeffs_cv The distortion coefficients
 * @param scale The scale of the target points, is optimized if the OptimizeScale flag is set
 * @param flags Flags for the optimization
 * @return A tuple with the optimized base-to-target and flange-to-camera transformations
 */
ExtrinsicResult calibrate_extrinsics(ExtrinsicCalibType calib_type,
                                     const std::vector<cv::Point3f>& target_points_cv,
                                     const std::vector<std::map<size_t, cv::Point2f>>& image_points_cv,
                                     const std::vector<cv::Quat<double>>& B_rquats,
                                     const std::vector<cv::Vec<double, 3>>& B_tvecs,
                                     const std::vector<cv::Quat<double>>& tar_2_cam_rquats,
                                     const std::vector<cv::Vec<double, 3>>& tar_2_cam_tvecs,
                                     const cv::Matx<double, 3, 3>& camera_matrix_cv,
                                     const cv::Vec<double, 5>& distort_coeffs_cv,
                                     double scale = 1.0,
                                     int flags = ExtrinsicFlags::None);

}  // namespace caliban

#endif  // CALIBAN_EXTRINSIC_H
