#ifndef CALIBAN_UTILS_H
#define CALIBAN_UTILS_H

#include "constants.h"

#include <opencv2/core.hpp>
#include <opencv2/core/quaternion.hpp>

#include <ceres/rotation.h>

namespace caliban {
/**
 * @brief   Computes the product of two SE3 transformations.
 *
 * a and b should not have overlapping memory regions.
 *
 * @param a     The first transformation (array of n_quat_se3 size, quaternion + Cartesian vector representation)
 * @param b     The second transformation (array of n_quat_se3 size, quaternion + Cartesian vector representation)
 * @param out   The output transformation (array of n_quat_se3 size, quaternion + Cartesian vector representation)
 */
template <typename T>
void MultiplySE3(const T* const a, const T* const b, T* out) {
    ceres::QuaternionProduct(a, b, out);
    ceres::QuaternionRotatePoint(a, b + n_quat_so3, out + n_quat_so3);

    out[n_quat_so3] += a[n_quat_so3];
    out[n_quat_so3 + 1] += a[n_quat_so3 + 1];
    out[n_quat_so3 + 2] += a[n_quat_so3 + 2];
}

/**
 * @brief   Computes the inverse of the SE3 transformation.
 *
 * in and out should not have overlapping memory regions.
 *
 * @param in    The input transformation (array of n_quat_se3 size, quaternion + Cartesian vector representation)
 * @param out   The output transformation (array of n_quat_se3 size, quaternion + Cartesian vector representation)
 */
template <typename T>
void InverseSE3(const T* const in, T* out) {
    out[0] = -in[0];
    out[1] = in[1];
    out[2] = in[2];
    out[3] = in[3];

    ceres::QuaternionRotatePoint(out, in + n_quat_so3, out + n_quat_so3);

    out[n_quat_so3] = -out[n_quat_so3];
    out[n_quat_so3 + 1] = -out[n_quat_so3 + 1];
    out[n_quat_so3 + 2] = -out[n_quat_so3 + 2];
}

/**
 * @brief Provides the inverse of the input SE(3) transformation, a wrapper around InverseSE3.
 */
inline QuatSE3 invert(const QuatSE3& q) {
    QuatSE3 result;
    InverseSE3(q.data(), result.data());
    return result;
}

inline std::vector<Point3D> convert(const std::vector<cv::Point3f>& points_cv) {
    std::vector<Point3D> points;
    for (const auto& op : points_cv) {
        points.push_back({op.x, op.y, op.z});
    }

    return points;
}

inline std::vector<cv::Point3f> convert(const std::vector<Point3D>& points) {
    std::vector<cv::Point3f> points_cv;
    for (const auto& op : points) {
        points_cv.push_back({
            static_cast<float>(op[0]),
            static_cast<float>(op[1]),
            static_cast<float>(op[2]),
        });
    }

    return points_cv;
}

inline std::vector<std::map<size_t, Point2D>> convert(const std::vector<std::map<size_t, cv::Point2f>>& points_cv) {
    std::vector<std::map<size_t, Point2D>> points;
    for (const auto& cp : points_cv) {
        std::map<size_t, Point2D> points_loc;
        for (auto [index, point] : cp) {
            points_loc[index] = {point.x, point.y};
        }
        points.push_back(std::move(points_loc));
    }

    return points;
}

inline std::tuple<cv::Vec<double, 3>, cv::Vec<double, 3>> convert(const QuatSE3& quat_se3) {
    cv::Quat<double> q{quat_se3[0], quat_se3[1], quat_se3[2], quat_se3[3]};
    auto rvec = q.toRotVec();
    cv::Vec<double, 3> tvec{quat_se3[4], quat_se3[5], quat_se3[6]};
    return {rvec, tvec};
}

inline QuatSE3 convert(const cv::Vec<double, 3>& rvec, const cv::Vec<double, 3>& tvec) {
    auto q = cv::Quat<double>::createFromRvec(rvec);
    return {q.w, q.x, q.y, q.z, tvec(0), tvec(1), tvec(2)};
}
}  // namespace caliban

#endif  // CALIBAN_UTILS_H
