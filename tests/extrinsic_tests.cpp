#include <gtest/gtest.h>

#include <caliban/extrinsic.h>
#include <caliban/intrinsic.h>
#include "test_utils.h"

using namespace test_utils;

TEST(Extrinsic, planar) {
    auto images = collect_images(std::filesystem::path("no_curv") / "data");
    ASSERT_TRUE(images.size() > 0);

    const auto [B_rvecs, B_tvecs] = collect_transforms(std::filesystem::path("no_curv") / "robot_cali.txt");
    ASSERT_TRUE(B_rvecs.size() > 0);
    ASSERT_TRUE(B_tvecs.size() == B_rvecs.size());

    const auto pattern_size = cv::Size(14, 9);
    const auto corners = detect(images, pattern_size, true);
    ASSERT_EQ(corners.size(), images.size());

    const auto object_points = fill_obj_points(pattern_size, 12.1f);
    ASSERT_EQ(object_points.size(), pattern_size.width * pattern_size.height);

    const auto intrinsics = caliban::calibrate_intrinsics(object_points, corners, images[0].size());
    const auto extrinsics = caliban::calibrate_extrinsics(
        caliban::ExtrinsicCalibType::EyeInHand, intrinsics.target_points, corners, B_rvecs, B_tvecs,
        intrinsics.rotations, intrinsics.translations, intrinsics.camera_matrix, intrinsics.dist_coeffs, 1.0,
        caliban::ExtrinsicFlags::OptimizeScale);

    EXPECT_LT(extrinsics.rms_repro, 0.05);
    EXPECT_NEAR(extrinsics.scale, 1.0, 1e-3);
    EXPECT_NEAR(extrinsics.Z_rot.w, 0.0, 1e-1);
    EXPECT_NEAR(extrinsics.Z_rot.x, -1.0, 1e-1);
    EXPECT_NEAR(extrinsics.Z_rot.y, 0.0, 1e-1);
    EXPECT_NEAR(extrinsics.Z_rot.z, 0.0, 1e-1);
    EXPECT_NEAR(extrinsics.Z_tvec[0], -100.0, 1e-1);
    EXPECT_NEAR(extrinsics.Z_tvec[1], -200.0, 1e-1);
    EXPECT_NEAR(extrinsics.Z_tvec[2], 0.0, 1e-1);
}

TEST(Extrinsic, curved) {
    auto images = collect_images(std::filesystem::path("curv") / "data");
    ASSERT_TRUE(images.size() > 0);

    const auto [B_rvecs, B_tvecs] = collect_transforms(std::filesystem::path("curv") / "robot_cali.txt");
    ASSERT_TRUE(B_rvecs.size() > 0);
    ASSERT_TRUE(B_tvecs.size() == B_rvecs.size());

    const auto pattern_size = cv::Size(14, 9);
    const auto corners = detect(images, pattern_size, true);
    ASSERT_EQ(corners.size(), images.size());

    const auto object_points = fill_obj_points(pattern_size, 12.1f);
    ASSERT_EQ(object_points.size(), pattern_size.width * pattern_size.height);

    const auto intrinsics = caliban::calibrate_intrinsics(object_points, corners, images[0].size());
    const auto extrinsics = caliban::calibrate_extrinsics(
        caliban::ExtrinsicCalibType::EyeInHand, intrinsics.target_points, corners, B_rvecs, B_tvecs,
        intrinsics.rotations, intrinsics.translations, intrinsics.camera_matrix, intrinsics.dist_coeffs, 1.0,
        caliban::ExtrinsicFlags::OptimizeScale);

    EXPECT_LT(extrinsics.rms_repro, 0.05);
    EXPECT_NEAR(extrinsics.scale, 0.998, 1e-3);
    EXPECT_NEAR(extrinsics.Z_rot.w, 0.0, 1e-1);
    EXPECT_NEAR(extrinsics.Z_rot.x, -1.0, 1e-1);
    EXPECT_NEAR(extrinsics.Z_rot.y, 0.0, 1e-1);
    EXPECT_NEAR(extrinsics.Z_rot.z, 0.0, 1e-1);
    EXPECT_NEAR(extrinsics.Z_tvec[0], -100.0, 1e-1);
    EXPECT_NEAR(extrinsics.Z_tvec[1], -200.0, 1e-1);
    EXPECT_NEAR(extrinsics.Z_tvec[2], 0.0, 1e-1);
}

TEST(Extrinsic, real) {
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
        caliban::ExtrinsicCalibType::EyeInHand, intrinsics.target_points, corners, B_rvecs, B_tvecs,
        intrinsics.rotations, intrinsics.translations, intrinsics.camera_matrix, intrinsics.dist_coeffs, 1.0,
        caliban::ExtrinsicFlags::OptimizeScale);

    EXPECT_LT(extrinsics.rms_repro, 1.45);
    EXPECT_NEAR(extrinsics.scale, 1.009, 1e-3);
    // FIXME: find the reason for instability of the translation vector (Z_tvec windows vs linux)
}
