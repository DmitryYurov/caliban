#include "caliban/intrinsic.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace caliban {
struct ReprojectionError {
    ReprojectionError(const Point2D& point_2d)
        : point_2d_(point_2d) {}

    template <typename T>
    bool operator()(const T* const cam_mat,
                    const T* const distort,
                    const T* const obj_2_cam,
                    const T* const point_3d,
                    T* residuals) const {
        T fx = cam_mat[0];
        T cx = cam_mat[1];
        T fy = cam_mat[2];
        T cy = cam_mat[3];

        T k1 = distort[0];
        T k2 = distort[1];
        T p1 = distort[2];
        T p2 = distort[3];
        T k3 = distort[4];

        T P[n_r3];
        P[0] = T(point_3d[0]);
        P[1] = T(point_3d[1]);
        P[2] = T(point_3d[2]);

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

        T r_dist = T(1) + k1 * r_2 + k2 * r_4 + k3 * r_6;

        p[0] = p[0] * r_dist + T(2) * p1 * p[0] * p[1] + p2 * (r_2 + T(2) * p[0] * p[0]);
        p[1] = p[1] * r_dist + T(2) * p2 * p[0] * p[1] + p1 * (r_2 + T(2) * p[1] * p[1]);

        T predicted_x = p[0] * fx + cx;
        T predicted_y = p[1] * fy + cy;

        residuals[0] = predicted_x - T(point_2d_[0]);
        residuals[1] = predicted_y - T(point_2d_[1]);

        return true;
    }

    static auto Create(const Point2D point_2d) -> std::unique_ptr<ceres::CostFunction> {
        constexpr auto n_residuals = n_r2;
        using AutoDiffCF = ceres::AutoDiffCostFunction<ReprojectionError, n_residuals, n_cam, n_dist, n_quat_se3, n_r3>;

        auto estimator = std::make_unique<ReprojectionError>(point_2d);
        auto result = std::make_unique<AutoDiffCF>(estimator.get());
        estimator.release();

        return result;
    }

    Point2D point_2d_;
};

double norm2(const Point3D& point) {
    return point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
}

Point3D crossProduct(const Point3D& p1, const Point3D& p2) {
    return {p1[1] * p2[2] - p1[2] * p2[1], p1[2] * p2[0] - p1[0] * p2[2], p1[0] * p2[1] - p1[1] * p2[0]};
}

// We assume for this selection, that all points are in the same plane defined by the condition z = 0
std::map<size_t, std::vector<int>> selectBase(std::vector<Point3D> target_points) {
    if (target_points.size() < 3) {
        throw std::runtime_error("Not enough points to select from");
    }

    // we always take the very first and very last points to define the base
    const auto p0_iter = target_points.begin();
    const size_t p0_i = std::distance(target_points.begin(), p0_iter);
    std::transform(target_points.begin(), target_points.end(), target_points.begin(), [p0 = *p0_iter](const auto& p) {
        return Point3D{p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]};
    });
    const auto p0 = *p0_iter;

    const auto p2_iter = std::prev(target_points.end());
    const size_t p2_i = std::distance(target_points.begin(), p2_iter);
    const auto p2 = *p2_iter;

    // now selecting the third point to form the base of the coordinate system
    // doing it by selecting the point that produces the biggest cross product with the p2
    const auto p1_iter = std::max_element(std::next(p0_iter), p2_iter, [p2](const auto& lhs, const auto& rhs) {
        return norm2(crossProduct(lhs, p2)) < norm2(crossProduct(rhs, p2));
    });
    if (norm2(crossProduct(*p1_iter, p2)) < std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("Search for the base point failed");
    }
    const size_t p1_i = std::distance(target_points.begin(), p1_iter);

    return {{p0_i, {0, 1, 2}}, {p1_i, {0, 1, 2}}, {p2_i, {2}}};
}

double calibrate_intrinsics(std::vector<Point3D>& target_points,
                            const std::vector<std::map<size_t, Point2D>>& image_points,
                            CameraMatrix& camera_matrix,
                            DistortionCoefficients& distortion_coefficients,
                            std::vector<QuatSE3> obj_2_cams) {
    // define the manifold for the SE3 transformation
    auto se3 = ceres::ProductManifold{ceres::QuaternionManifold{}, ceres::EuclideanManifold<3>{}};

    // manifolds (constraints) for the base points
    const auto base_points = selectBase(target_points);
    std::map<size_t, ceres::SubsetManifold> base_manifolds;
    for (const auto& [i, dims] : base_points) {
        base_manifolds.emplace(i, ceres::SubsetManifold{n_r3, dims});
    }

    ceres::Problem::Options problem_options;
    problem_options.cost_function_ownership = ceres::TAKE_OWNERSHIP;
    problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    auto problem = ceres::Problem(problem_options);

    // adding parameter blocks
    problem.AddParameterBlock(camera_matrix.data(), camera_matrix.size());
    problem.AddParameterBlock(distortion_coefficients.data(), distortion_coefficients.size());
    for (auto& obj_2_cam : obj_2_cams) {
        problem.AddParameterBlock(obj_2_cam.data(), obj_2_cam.size(), &se3);
    }

    for (size_t i = 0; i < target_points.size(); ++i) {
        auto& target_point = target_points[i];
        if (base_points.contains(i)) {
            problem.AddParameterBlock(target_point.data(), target_point.size(), &base_manifolds.at(i));
        } else {
            problem.AddParameterBlock(target_point.data(), target_point.size());
        }
    }

    // Add residual blocks
    for (size_t i_im = 0; i_im < image_points.size(); ++i_im) {
        const auto& points_per_im = image_points[i_im];
        for (const auto& [tp_index, point_2d] : points_per_im) {
            auto& point_3d = target_points[tp_index];
            auto cost_function = ReprojectionError::Create(point_2d);
            problem.AddResidualBlock(cost_function.release(),
                                     nullptr,  // loss function (nullptr = default)
                                     camera_matrix.data(), distortion_coefficients.data(), obj_2_cams[i_im].data(),
                                     point_3d.data());
        }
    }

    // Run the solver
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
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
    // 7 is for the 7 dofs fixed by the base points
    const size_t bessel = n_r3 * target_points.size() + n_cam + n_dist + n_se3 * obj_2_cams.size() - 7;

    return std::sqrt(2. * summary.final_cost / (n_points - bessel));  // 2. is due to the way Ceres computes the cost
}
}  // namespace caliban
