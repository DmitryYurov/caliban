#include <gtest/gtest.h>

#include "test_utils.h"

using namespace test_utils;

TEST(Intrinsic, planar) {
    auto images = collect_images("no_curv/data");
    ASSERT_TRUE(images.size() > 0);

    const auto pattern_size = cv::Size(14, 9);
    const auto corners = detect(images, pattern_size, true);
    ASSERT_EQ(corners.size(), images.size());

    const auto object_points = fill_obj_points(pattern_size, 12.1f);
    ASSERT_EQ(object_points.size(), pattern_size.width * pattern_size.height);

    const auto result = caliban::calibrate_intrinsics(object_points, corners, images[0].size());

    EXPECT_LT(result.rms_repro, 1e-1);
    EXPECT_NEAR(result.camera_matrix(0, 0), 1400.0, 1e-1);  // fx
    EXPECT_NEAR(result.camera_matrix(0, 2), 960.0, 2.0);    // cx
    EXPECT_NEAR(result.camera_matrix(1, 1), 1400.0, 1e-1);  // fy
    EXPECT_NEAR(result.camera_matrix(1, 2), 540.0, 2.0);    // cy
    EXPECT_NEAR(result.dist_coeffs(0, 0), -1.73e-1, 1e-2);  // k1
    EXPECT_NEAR(result.dist_coeffs(1, 0), 2.49e-2, 1e-2);   // k2
}

TEST(Intrinsic, curved) {
    auto images = collect_images("curv/data");
    ASSERT_TRUE(images.size() > 0);

    const auto pattern_size = cv::Size(14, 9);
    const auto corners = detect(images, pattern_size, true);
    ASSERT_EQ(corners.size(), images.size());

    const auto object_points = fill_obj_points(pattern_size, 12.1f);
    ASSERT_EQ(object_points.size(), pattern_size.width * pattern_size.height);

    const auto result = caliban::calibrate_intrinsics(object_points, corners, images[0].size());

    EXPECT_LT(result.rms_repro, 1e-1);
    EXPECT_NEAR(result.camera_matrix(0, 0), 1400.0, 5e-1);  // fx
    EXPECT_NEAR(result.camera_matrix(0, 2), 960.0, 2.0);    // cx
    EXPECT_NEAR(result.camera_matrix(1, 1), 1400.0, 5e-1);  // fy
    EXPECT_NEAR(result.camera_matrix(1, 2), 540.0, 2.0);    // cy
    EXPECT_NEAR(result.dist_coeffs(0, 0), -1.73e-1, 1e-2);  // k1
    EXPECT_NEAR(result.dist_coeffs(1, 0), 2.49e-2, 1e-2);   // k2
}

TEST(Intrinsic, real) {
    auto images = collect_images("1");
    ASSERT_TRUE(images.size() > 0);

    const auto pattern_size = cv::Size(6, 8);
    const auto corners = detect(images, pattern_size, false);
    ASSERT_EQ(corners.size(), images.size());

    const auto object_points = fill_obj_points(pattern_size, 28.5f);
    ASSERT_EQ(object_points.size(), pattern_size.width * pattern_size.height);

    const auto result = caliban::calibrate_intrinsics(object_points, corners, images[0].size());

    EXPECT_LT(result.rms_repro, 1e-1);
    EXPECT_NEAR(result.camera_matrix(0, 0), 1084.75, 1e-2);  // fx
    EXPECT_NEAR(result.camera_matrix(0, 2), 309.0, 1e-1);    // cx
    EXPECT_NEAR(result.camera_matrix(1, 1), 1083.64, 1e-1);  // fy
    EXPECT_NEAR(result.camera_matrix(1, 2), 235.0, 2e-1);    // cy
    EXPECT_NEAR(result.dist_coeffs(0, 0), -0.23, 1e-2);      // k1
    EXPECT_NEAR(result.dist_coeffs(1, 0), 0.21, 1e-2);       // k2
}
