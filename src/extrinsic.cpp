#include "caliban/extrinsic.h"

#include "types.h"
#include "utils.h"

#include <ceres/ceres.h>
#include <opencv2/calib3d.hpp>

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

double calibrate(ExtrinsicCalibType calib_type,
                 std::vector<Point3D>& target_points,
                 const std::vector<std::map<size_t, Point2D>>& image_points,
                 const std::vector<QuatSE3>& Bs,
                 QuatSE3& X,
                 QuatSE3& Z,
                 CameraMatrix& camera_matrix,
                 DistortionCoefficients& distortion_coefficients,
                 double scale,
                 int flags) {
    // define the SE3 manifold
    auto se3 = ceres::ProductManifold{ceres::QuaternionManifold{}, ceres::EuclideanManifold<3>{}};

    ceres::Problem::Options problem_options;
    problem_options.cost_function_ownership = ceres::TAKE_OWNERSHIP;
    problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    auto problem = ceres::Problem(problem_options);

    // adding parameter blocks
    problem.AddParameterBlock(camera_matrix.data(), camera_matrix.size());
    problem.SetParameterBlockConstant(camera_matrix.data());  // camera matrix is fixed
    problem.AddParameterBlock(distortion_coefficients.data(), distortion_coefficients.size());
    problem.SetParameterBlockConstant(distortion_coefficients.data());  // distortion coefficients are fixed
    problem.AddParameterBlock(X.data(), X.size(), &se3);
    problem.AddParameterBlock(Z.data(), Z.size(), &se3);

    // add scaling parameter
    problem.AddParameterBlock(&scale, 1);
    if (flags & ExtrinsicFlags::OptimizeScale == 0) {
        problem.SetParameterBlockConstant(&scale);
    }

    // take the very first target point as the base for rescaling
    const auto base_point = target_points[0];

    // Add residual blocks
    for (size_t i_im = 0; i_im < image_points.size(); ++i_im) {
        const auto& points_per_im = image_points[i_im];
        const auto& B = Bs[i_im];
        for (const auto& [tp_index, point_2d] : points_per_im) {
            const auto& point_3d = target_points[tp_index];
            auto cost_function = ReprojectionError::Create(calib_type, B, point_2d, point_3d, base_point);
            problem.AddResidualBlock(cost_function.release(),
                                     nullptr,  // loss function (nullptr = default)
                                     X.data(), camera_matrix.data(), distortion_coefficients.data(), Z.data(), &scale);
        }
    }

    // Run the solver
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable()) {
        throw std::runtime_error("Ceres solver failed, reason: " + summary.message);
    }

    const size_t n_points =
        std::accumulate(image_points.begin(), image_points.end(), size_t(0),
                        [](size_t acc, const auto& points_2d) { return acc + n_r2 * points_2d.size(); });

    const size_t bessel = flags & ExtrinsicFlags::OptimizeScale == 0 ? n_se3 * 2 : n_se3 * 2 + 1;

    return std::sqrt(2. * summary.final_cost / (n_points - bessel));  // 2. is due to the way Ceres computes the cost
}

ExtrinsicResult calibrate_extrinsics(ExtrinsicCalibType calib_type,
                                     const std::vector<cv::Point3f>& target_points_cv,
                                     const std::vector<std::map<size_t, cv::Point2f>>& image_points_cv,
                                     const std::vector<cv::Vec<double, 3>>& B_rvecs,
                                     const std::vector<cv::Vec<double, 3>>& B_tvecs,
                                     const std::vector<cv::Vec<double, 3>>& tar_2_cam_rvecs,
                                     const std::vector<cv::Vec<double, 3>>& tar_2_cam_tvecs,
                                     const cv::Matx<double, 3, 3>& camera_matrix_cv,
                                     const cv::Vec<double, 5>& distort_coeffs_cv,
                                     double scale,
                                     int flags) {
    // computing the initial guess with opencv built-in functions
    // depending on the calibration type, A transform is either target-to-camera (eye-in-hand) or camera-to-target
    // (eye-to-hand)
    std::vector<cv::Mat> A_rvecs;
    std::vector<cv::Mat> A_tvecs;
    {
        std::vector<QuatSE3> As;
        for (size_t i = 0; i < tar_2_cam_rvecs.size(); ++i) {
            As.push_back(convert(tar_2_cam_rvecs[i], tar_2_cam_tvecs[i]));
        }
        if (calib_type == ExtrinsicCalibType::EyeToHand) {
            std::transform(As.begin(), As.end(), As.begin(), [](const auto& A) { return invert(A); });
        }

        for (const auto& A : As) {
            const auto& [rvec, tvec] = convert(A);
            A_rvecs.push_back(cv::Mat(rvec));
            A_tvecs.push_back(cv::Mat(tvec));
        }
    }

    QuatSE3 X{};
    QuatSE3 Z{};
    {
        std::vector<cv::Mat> B_rvecs_mat;
        std::vector<cv::Mat> B_tvecs_mat;
        for (size_t i = 0; i < B_rvecs.size(); ++i) {
            B_rvecs_mat.push_back(cv::Mat(B_rvecs[i]));
            B_tvecs_mat.push_back(cv::Mat(B_tvecs[i]));
        }

        cv::Mat_<double> X_rvec_mat;
        cv::Mat_<double> X_tvec_mat;
        cv::Mat_<double> Z_rvec_mat;
        cv::Mat_<double> Z_tvec_mat;
        cv::calibrateRobotWorldHandEye(A_rvecs, A_tvecs, B_rvecs_mat, B_tvecs_mat, X_rvec_mat, X_tvec_mat, Z_rvec_mat,
                                       Z_tvec_mat);

        const auto X_rvec = cv::Vec<double, 3>{X_rvec_mat(0), X_rvec_mat(1), X_rvec_mat(2)};
        const auto X_tvec = cv::Vec<double, 3>{X_tvec_mat(0), X_tvec_mat(1), X_tvec_mat(2)};
        const auto Z_rvec = cv::Vec<double, 3>{Z_rvec_mat(0), Z_rvec_mat(1), Z_rvec_mat(2)};
        const auto Z_tvec = cv::Vec<double, 3>{Z_tvec_mat(0), Z_tvec_mat(1), Z_tvec_mat(2)};
        X = convert(X_rvec, X_tvec);
        Z = convert(Z_rvec, Z_tvec);
    }

    std::vector<QuatSE3> Bs;
    for (size_t i = 0; i < B_rvecs.size(); ++i) {
        Bs.push_back(convert(B_rvecs[i], B_tvecs[i]));
    }

    // now perform the refinement
    auto target_points = convert(target_points_cv);
    auto image_points = convert(image_points_cv);
    CameraMatrix camera_matrix{camera_matrix_cv(0, 0), camera_matrix_cv(0, 2), camera_matrix_cv(1, 1),
                               camera_matrix_cv(1, 2)};
    DistortionCoefficients distortion_coefficients{distort_coeffs_cv[0], distort_coeffs_cv[1], distort_coeffs_cv[2],
                                                   distort_coeffs_cv[3], distort_coeffs_cv[4]};

    ExtrinsicResult result;
    result.rms_repro = calibrate(calib_type, target_points, image_points, Bs, X, Z, camera_matrix,
                                 distortion_coefficients, scale, flags);

    return result;
}
}  // namespace caliban
