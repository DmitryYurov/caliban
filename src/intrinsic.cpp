#include "caliban/intrinsic.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace caliban {
struct ReprojectionError {
    ReprojectionError(const Point2D& point_2d)
        : point_2d_(point_2d)
    {
    }
    
    template <typename T>
    bool operator()(const T* const cam_mat, const T* const distort, const T* const obj_2_cam, const T* const point_3d, T* residuals) const
    {
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
        p[0] += obj_2_cam[n_so3];
        p[1] += obj_2_cam[n_so3 + 1];
        p[2] += obj_2_cam[n_so3 + 2];

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

    static auto Create(const Point2D point_2d)
        -> std::unique_ptr<ceres::CostFunction>
    {
        constexpr auto n_residuals = n_r2;
        using AutoDiffCF = ceres::AutoDiffCostFunction<ReprojectionError, n_residuals, n_cam, n_dist, n_se3, n_r3>;

        auto estimator   = std::make_unique<ReprojectionError>(point_2d);
        auto result      = std::make_unique<AutoDiffCF>(estimator.get());
        estimator.release();

        return result;
    }

    Point2D point_2d_;
};

double norm2(const Point3D& point) {
    return point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
}

Point3D crossProduct(const Point3D& p1, const Point3D& p2) {
    return {
        p1[1] * p2[2] - p1[2] * p2[1],
        p1[2] * p2[0] - p1[0] * p2[2],
        p1[0] * p2[1] - p1[1] * p2[0]
    };
}

std::vector<size_t> selectFixedDims(const Point3D& p1, const Point3D& p2) {
    const auto cp = crossProduct(p1, p2);
    const auto iter = std::max_element(cp.begin(), cp.end(), [](const auto& lhs, const auto& rhs) {
        return std::abs(lhs) < std::abs(rhs);
    });
    const size_t i_max = std::distance(cp.begin(), iter);

    return i_max == 0 ? std::vector<size_t>{1, 2} : i_max == 1 ? std::vector<size_t>{0, 2} : std::vector<size_t>{0, 1};
}

std::map<size_t, std::vector<size_t>> selectBase(std::vector<Point3D> target_points) {
    if (target_points.size() < 3) {
        throw std::runtime_error("Not enough points to select from");
    }

    // first selected point is the target's center of origin - it makes sense to select the point with the smallest distance from (0, 0, 0)
    const auto base_iter = std::min_element(target_points.begin(), target_points.end(), [](const auto& lhs, const auto& rhs) {
        return norm2(lhs) < norm2(rhs);
    });

    const size_t i_base = std::distance(target_points.begin(), base_iter);
    auto p0 = target_points[i_base];
    std::transform(target_points.begin(), target_points.end(), target_points.begin(), [&p0](const auto& p) {
        return Point3D{p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]};
    });

    // now selecting two points to form the base of the coordinate system
    // doing it by selecting the least collinear points except from the base point
    // "least collinear" means the points that have the biggest cross product
    std::map<std::pair<size_t, size_t>, double> cross_products;
    for (size_t i = 0; i < target_points.size(); ++i) {
        for (size_t j = i + 1; j < target_points.size(); ++j) {
            if (i == i_base || j == i_base) {
                continue;
            }

            const auto& p1 = target_points[i];
            const auto& p2 = target_points[j];
            cross_products[{i, j}] = norm2(crossProduct(p1, p2));
        }
    }

    const auto max_iter = std::max_element(cross_products.begin(), cross_products.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second < rhs.second;
    });

    if (max_iter->second < std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("All points in the target are collinear");
    }

    const auto& p1 = target_points[max_iter->first.first];
    const auto& p2 = target_points[max_iter->first.second];
    const auto fixed_dims = selectFixedDims(p1, p2);

    return {
        { i_base, {0, 1, 2}},
        { max_iter->first.first, fixed_dims },
        { max_iter->first.second, fixed_dims }
    };
}

void calibrate(
    std::vector<Point3D>& target_points,
    const std::vector<std::vector<Point2D>>& image_points,
    CameraMatrix& camera_matrix,
    DistortionCoefficients& distortion_coefficients,
    std::vector<QuatSE3> obj_2_cams
) {
    // define the manifold for the R^3 space
    auto r3 = ceres::EuclideanManifold<3>{};
    // define the manifold for the SE3 transformation
    auto se3 = ceres::ProductManifold{ceres::QuaternionManifold{}, ceres::EuclideanManifold<3>{}};

    ceres::Problem::Options problem_options;
    problem_options.cost_function_ownership = ceres::TAKE_OWNERSHIP;
    problem_options.manifold_ownership      = ceres::DO_NOT_TAKE_OWNERSHIP;
    auto problem                            = ceres::Problem(problem_options);

    // adding parameter blocks
    problem.AddParameterBlock(camera_matrix.data(), camera_matrix.size());
    problem.AddParameterBlock(distortion_coefficients.data(), distortion_coefficients.size());
    for (auto& obj_2_cam : obj_2_cams) {
        problem.AddParameterBlock(obj_2_cam.data(), obj_2_cam.size(), &se3);
    }
    for (auto& target_point : target_points) {
        problem.AddParameterBlock(target_point.data(), target_point.size(), &r3);
    }

    //   Add residual blocks
    for (size_t i_im = 0; i_im < image_points.size(); ++i_im) {
        const auto& points_per_im = image_points[i_im];
        for (size_t j = 0; j < points_per_im.size(); ++j) {
            auto& point_2d = points_per_im[j];
            auto& point_3d = target_points[j];
            auto cost_function                = ReprojectionError::Create(point_2d);
            problem.AddResidualBlock(cost_function.release(),
                                     nullptr,  // loss function (nullptr = default)
                                     camera_matrix.data(),
                                     distortion_coefficients.data(),
                                     obj_2_cams[i_im].data(),
                                     point_3d.data());
        }
    }

    // Run the solver
    ceres::Solver::Options options;
    options.trust_region_strategy_type   = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type           = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.logging_type                 = ceres::SILENT;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}
}  // namespace caliban
