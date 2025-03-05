#include "caliban/extrinsic.h"

#include "caliban/types.h"
#include "utils.h"

#include <ceres/ceres.h>

#include <memory>

namespace caliban {

struct ReprojectionError {
    ReprojectionError(ExtrinsicCalibType calib_type, QuatSE3 B, Point2D point_2d, Point3D point_3d, Point3D base)
        : m_calib_type(calib_type)
        , m_B(B)
        , m_point_2d(point_2d)
        , m_point_3d(point_3d)
        , m_base(base) {}

    template <typename T>
    bool operator()(const T* const X,
                    const T* const cam_mat,
                    const T* const distort,
                    const T* const Z,
                    const T* const scale,
                    T* residuals) const {
        // convert B_ to the required type T
        T B[n_quat_se3];
        for (size_t i = 0; i < n_quat_se3; ++i) {
            B[i] = T(m_B[i]);
        }

        // computing transform from object to camera
        // eye-in-hand: ob_2_cam = Z * B * X^{-1}
        // eye-to-hand: ob_2_cam = X * B^{-1} * Z^{-1}
        // they are inverse of each other
        T obj_2_cam[n_quat_se3];
        T ZB[n_quat_se3];
        MultiplySE3(Z, B, ZB);
        if (m_calib_type == ExtrinsicCalibType::EyeInHand) {
            T inv_X[n_quat_se3];
            InverseSE3(X, inv_X);
            MultiplySE3(ZB, inv_X, obj_2_cam);
        } else if (m_calib_type == ExtrinsicCalibType::EyeToHand) {
            T inv_ZB[n_quat_se3];
            InverseSE3(ZB, inv_ZB);
            MultiplySE3(X, inv_ZB, obj_2_cam);
        }

        T P[n_r3];
        P[0] = T(m_point_3d[0] - m_base[0]) * scale[0] + T(m_base[0]);
        P[1] = T(m_point_3d[1] - m_base[1]) * scale[0] + T(m_base[1]);
        P[2] = T(m_point_3d[2] - m_base[2]) * scale[0] + T(m_base[2]);

        T p[n_r3];
        ceres::QuaternionRotatePoint(obj_2_cam, P, p);
        p[0] += obj_2_cam[n_quat_so3];
        p[1] += obj_2_cam[n_quat_so3 + 1];
        p[2] += obj_2_cam[n_quat_so3 + 2];

        p[0] = p[0] / p[2];
        p[1] = p[1] / p[2];

        T r_2 = p[0] * p[0] + p[1] * p[1];
        T r_4 = r_2 * r_2;
        T r_6 = r_4 * r_2;

        T r_dist = T(1) + distort[0] * r_2 + distort[1] * r_4 + distort[4] * r_6;

        p[0] = p[0] * r_dist + T(2) * distort[2] * p[0] * p[1] + distort[3] * (r_2 + T(2) * p[0] * p[0]);
        p[1] = p[1] * r_dist + T(2) * distort[3] * p[0] * p[1] + distort[2] * (r_2 + T(2) * p[1] * p[1]);

        T predicted_x = p[0] * cam_mat[0] + cam_mat[1];  // fx, cx
        T predicted_y = p[1] * cam_mat[2] + cam_mat[3];  // fy, cy

        residuals[0] = predicted_x - T(m_point_2d[0]);
        residuals[1] = predicted_y - T(m_point_2d[1]);

        return true;
    }

    static auto Create(ExtrinsicCalibType calib_type, QuatSE3 B, Point2D point_2d, Point3D point_3d, Point3D base)
        -> std::unique_ptr<ceres::CostFunction> {
        constexpr auto n_residuals = n_r2;
        constexpr size_t n_scale = 1U;
        using AutoDiffCF =
            ceres::AutoDiffCostFunction<ReprojectionError, n_residuals, n_quat_se3, n_cam, n_dist, n_quat_se3, n_scale>;
        auto estimator = std::make_unique<ReprojectionError>(calib_type, B, point_2d, point_3d, base);
        auto result = std::make_unique<AutoDiffCF>(estimator.get());
        estimator.release();

        return result;
    }

    const ExtrinsicCalibType m_calib_type;
    const QuatSE3 m_B{};
    const Point2D m_point_2d{};
    const Point3D m_point_3d{};
    const Point3D m_base{};
};

double calibrate_extrinsics(ExtrinsicCalibType type,
                            std::vector<Point3D>& target_points,
                            const std::vector<std::map<size_t, Point2D>>& image_points,
                            const std::vector<QuatSE3>& Bs,
                            QuatSE3& X,
                            QuatSE3& Z,
                            const CameraMatrix& camera_matrix,
                            const DistortionCoefficients& distortion_coefficients,
                            int flags) {
    return 0.0;
}
}  // namespace caliban
