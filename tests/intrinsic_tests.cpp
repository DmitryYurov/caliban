#include <gtest/gtest.h>

#include <caliban/intrinsic.h>
#include <filesystem>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

auto collect_images(std::filesystem::path subpath) -> std::vector<cv::Mat> {
    auto full_path = std::filesystem::path(TEST_DATA_PATH) / subpath;

    std::vector<std::filesystem::path> path_list;
    for (const auto& entry : std::filesystem::directory_iterator(full_path)) {
        if (entry.path().extension() == ".png") {
            path_list.push_back(entry.path());
        }
    }

    std::vector<cv::Mat> images;
    images.reserve(path_list.size());
    for (const auto& path : path_list) {
        images.push_back(cv::imread(path.string(), cv::IMREAD_GRAYSCALE));
    }

    return images;
}

std::vector<std::vector<cv::Point2f>> detect(std::vector<cv::Mat> images, cv::Size pattern_size, bool with_radon) {
    std::vector<std::vector<cv::Point2f>> corners{};
    corners.reserve(images.size());

    constexpr int default_flags = cv::CALIB_CB_ACCURACY | cv::CALIB_CB_EXHAUSTIVE;
    const int detection_flags = with_radon ? int(cv::CALIB_CB_MARKER | default_flags) : default_flags;

    for (size_t i = 0; i < images.size(); ++i) {
        std::vector<cv::Point2f> corners_per_image;
        if (!cv::findChessboardCornersSB(images[i], pattern_size, corners_per_image, detection_flags)) {
            continue;
        }
        corners.push_back(std::move(corners_per_image));
    }

    return corners;
}

std::vector<cv::Point3f> fill_obj_points(cv::Size pattern_size, float tile_size) {
    std::vector<cv::Point3f> object_points;
    object_points.reserve(pattern_size.width * pattern_size.height);

    for (int j = 0; j < pattern_size.height; ++j) {
        for (int k = 0; k < pattern_size.width; ++k) {
            object_points.push_back(cv::Point3f(k * tile_size, j * tile_size, 0));
        }
    }

    return object_points;
}

auto calibrate(const std::vector<cv::Point3f>& target_points_cv,
               const std::vector<std::vector<cv::Point2f>>& corners_cv,
               cv::Size image_size)
    -> std::tuple<double, caliban::CameraMatrix, caliban::DistortionCoefficients, std::vector<caliban::Point3D>> {
    std::vector<caliban::Point3D> target_points;
    for (const auto& op : target_points_cv) {
        target_points.push_back({op.x, op.y, op.z});
    }

    std::vector<std::map<size_t, caliban::Point2D>> image_points;
    for (const auto& cp : corners_cv) {
        std::map<size_t, caliban::Point2D> image_points_per_image;
        for (size_t i = 0; i < cp.size(); ++i) {
            image_points_per_image[i] = {cp[i].x, cp[i].y};
        }
        image_points.push_back(std::move(image_points_per_image));
    }

    caliban::CameraMatrix camera_matrix{};
    std::vector<caliban::QuatSE3> obj_2_cams;
    {
        std::vector<std::vector<cv::Point3f>> target_points_cv_vec(corners_cv.size(), target_points_cv);
        auto camera_matrix_cv = cv::initCameraMatrix2D(target_points_cv_vec, corners_cv, image_size);
        camera_matrix = {camera_matrix_cv.at<double>(0, 0), camera_matrix_cv.at<double>(0, 2),
                         camera_matrix_cv.at<double>(1, 1), camera_matrix_cv.at<double>(1, 2)};

        for (size_t i = 0; i < corners_cv.size(); ++i) {
            cv::Mat rvec, tvec;
            cv::solvePnP(target_points_cv, corners_cv[i], camera_matrix_cv, cv::noArray(), rvec, tvec, false,
                         cv::SOLVEPNP_ITERATIVE);
            auto q = cv::Quat<double>::createFromRvec(rvec);
            obj_2_cams.push_back(
                {q.w, q.x, q.y, q.z, tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)});
        }
    }

    caliban::DistortionCoefficients dist_coeffs{0, 0, 0, 0, 0};
    const double rms =
        caliban::calibrate_intrinsics(target_points, image_points, camera_matrix, dist_coeffs, obj_2_cams);

    return {rms, camera_matrix, dist_coeffs, target_points};
}

TEST(Intrinsic, planar) {
    auto images = collect_images("no_curv/data");
    ASSERT_TRUE(images.size() > 0);

    const auto pattern_size = cv::Size(14, 9);
    const auto corners = detect(images, pattern_size, true);
    ASSERT_EQ(corners.size(), images.size());

    const auto object_points = fill_obj_points(pattern_size, 12.1f);
    ASSERT_EQ(object_points.size(), pattern_size.width * pattern_size.height);

    const auto [rms, cam_mat, dist_coeffs, target_points] = calibrate(object_points, corners, images[0].size());

    EXPECT_LT(rms, 1e-1);
    EXPECT_NEAR(cam_mat[0], 1400.0, 1e-1);        // fx
    EXPECT_NEAR(cam_mat[1], 960.0, 2.0);          // cx
    EXPECT_NEAR(cam_mat[2], 1400.0, 1e-1);        // fy
    EXPECT_NEAR(cam_mat[3], 540.0, 2.0);          // cy
    EXPECT_NEAR(dist_coeffs[0], -1.73e-1, 1e-2);  // k1
    EXPECT_NEAR(dist_coeffs[1], 2.49e-2, 1e-2);   // k2
}

TEST(Intrinsic, curved) {
    auto images = collect_images("curv/data");
    ASSERT_TRUE(images.size() > 0);

    const auto pattern_size = cv::Size(14, 9);
    const auto corners = detect(images, pattern_size, true);
    ASSERT_EQ(corners.size(), images.size());

    const auto object_points = fill_obj_points(pattern_size, 12.1f);
    ASSERT_EQ(object_points.size(), pattern_size.width * pattern_size.height);

    const auto [rms, cam_mat, dist_coeffs, target_points] = calibrate(object_points, corners, images[0].size());

    EXPECT_LT(rms, 1e-1);
    EXPECT_NEAR(cam_mat[0], 1400.0, 5e-1);        // fx
    EXPECT_NEAR(cam_mat[1], 960.0, 2.0);          // cx
    EXPECT_NEAR(cam_mat[2], 1400.0, 5e-1);        // fy
    EXPECT_NEAR(cam_mat[3], 540.0, 2.0);          // cy
    EXPECT_NEAR(dist_coeffs[0], -1.73e-1, 1e-2);  // k1
    EXPECT_NEAR(dist_coeffs[1], 2.49e-2, 1e-2);   // k2
}

TEST(Intrinsic, real) {
    auto images = collect_images("1");
    ASSERT_TRUE(images.size() > 0);

    const auto pattern_size = cv::Size(6, 8);
    const auto corners = detect(images, pattern_size, false);
    ASSERT_EQ(corners.size(), images.size());

    const auto object_points = fill_obj_points(pattern_size, 28.5f);
    ASSERT_EQ(object_points.size(), pattern_size.width * pattern_size.height);

    const auto [rms, cam_mat, dist_coeffs, target_points] = calibrate(object_points, corners, images[0].size());

    EXPECT_LT(rms, 1e-1);
    EXPECT_NEAR(cam_mat[0], 1084.75, 1e-2);    // fx
    EXPECT_NEAR(cam_mat[1], 309.0, 1e-1);      // cx
    EXPECT_NEAR(cam_mat[2], 1083.64, 1e-1);    // fy
    EXPECT_NEAR(cam_mat[3], 235.0, 2e-1);      // cy
    EXPECT_NEAR(dist_coeffs[0], -0.23, 1e-2);  // k1
    EXPECT_NEAR(dist_coeffs[1], 0.21, 1e-2);   // k2
}
