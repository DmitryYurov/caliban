#include <gtest/gtest.h>

#include <caliban/extrinsic.h>
#include <caliban/intrinsic.h>
#include "test_utils.h"

using namespace test_utils;

TEST(Intrinsic, real) {
    auto images = collect_images("1");
    ASSERT_TRUE(images.size() > 0);

    const auto [B_rvecs, B_tvecs] = collect_transforms(std::filesystem::path("1") / "robot_cali.txt");
    ASSERT_TRUE(B_rvecs.size() > 0);
    ASSERT_TRUE(B_tvecs.size() == B_rvecs.size());

    const auto pattern_size = cv::Size(8, 6);
    const auto corners = detect(images, pattern_size, false);
    ASSERT_EQ(corners.size(), images.size());

    const auto object_points = fill_obj_points(pattern_size, 28.5f);
    ASSERT_EQ(object_points.size(), pattern_size.width * pattern_size.height);

    const auto intrinsics = caliban::calibrate_intrinsics(object_points, corners, images[0].size());
    const auto extrinsics = caliban::calibrate_extrinsics(
        caliban::ExtrinsicCalibType::EyeInHand, intrinsics.target_points, corners, B_rvecs, B_tvecs, intrinsics.rvecs,
        intrinsics.tvecs, intrinsics.camera_matrix, intrinsics.dist_coeffs);

    EXPECT_LT(extrinsics.rms_repro, 1e-1);
}
