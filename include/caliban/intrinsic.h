#ifndef CALIBAN_INTRINSIC_H
#define CALIBAN_INTRINSIC_H

#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>

#include <array>
#include <map>
#include <vector>

namespace caliban {

/**
 * @brief The result of the intrinsic calibration
 */
struct IntrinsicsResult {
    double rms_repro;                              ///< The root mean square reprojection error
    cv::Matx<double, 3, 3> camera_matrix;          ///< The camera matrix (3 x 3)
    cv::Vec<double, 5> dist_coeffs;                ///< The distortion coefficients (5 coefficients: k1, k2, p1, p2, k3)
    std::vector<cv::Point3f> target_points;        ///< The updated target points
    std::vector<cv::Quat<double>> rotations;       ///< The rotation quaternions (one per image)
    std::vector<cv::Vec<double, 3>> translations;  ///< The translation vectors (one per image)
};

/**
 * @brief Calibrate the intrinsic parameters of the camera
 *
 * @param target_points_cv The 3D points in the world coordinate system
 * @param image_points_cv The 2D points in the image. The map key corresponds to the index of the target point
 * @param image_size The size of the image (used for camera matrix initialization)
 *
 * @return The intrinsic calibration result
 */
IntrinsicsResult calibrate_intrinsics(const std::vector<cv::Point3f>& target_points_cv,
                                      const std::vector<std::map<size_t, cv::Point2f>>& image_points_cv,
                                      const cv::Size& image_size);

}  // namespace caliban

#endif  // CALIBAN_INTRINSIC_H
