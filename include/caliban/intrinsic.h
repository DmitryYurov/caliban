#ifndef CALIBAN_INTRINSIC_H
#define CALIBAN_INTRINSIC_H

#include <opencv2/core.hpp>

#include <array>
#include <map>
#include <vector>

namespace caliban {

struct IntrinsicsResult {
    double rms_repro;
    cv::Matx<double, 3, 3> camera_matrix;
    cv::Matx<double, 5, 1> dist_coeffs;
    std::vector<cv::Point3f> target_points;
    std::vector<cv::Vec<double, 3>> rvecs;
    std::vector<cv::Vec<double, 3>> tvecs;
};

IntrinsicsResult calibrate_intrinsics(const std::vector<cv::Point3f>& target_points_cv,
                                      const std::vector<std::map<size_t, cv::Point2f>>& image_points_cv,
                                      const cv::Size& image_size);

}  // namespace caliban

#endif  // CALIBAN_INTRINSIC_H
