#include <gtest/gtest.h>

#include <caliban/intrinsic.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>

auto collect_images(std::filesystem::path subpath) -> std::vector<cv::Mat> {
    auto full_path = std::filesystem::path(TEST_DATA_PATH) / subpath;

    std::vector<std::filesystem::path> path_list;
    for (const auto &entry : std::filesystem::directory_iterator(full_path)) {
        if (entry.path().extension() == ".png") {
            path_list.push_back(entry.path());
        }
    }

    std::vector<cv::Mat> images;
    images.reserve(path_list.size());
    for (const auto &path : path_list) {
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

TEST(Intrinsic, planar){
    auto images = collect_images("no_curv/data");
    ASSERT_TRUE(images.size() > 0);

    const auto pattern_size = cv::Size(14, 9);
    const auto corners = detect(images, pattern_size, true);
    ASSERT_EQ(corners.size(), images.size());

    const auto object_points = fill_obj_points(pattern_size, 12.1f);
    ASSERT_EQ(object_points.size(), pattern_size.width * pattern_size.height);


}

TEST(Intrinsic, curved){
    auto images = collect_images("curv/data");
    ASSERT_TRUE(images.size() > 0);

    const auto pattern_size = cv::Size(14, 9);
    const auto corners = detect(images, pattern_size, true);
    ASSERT_EQ(corners.size(), images.size());
}

TEST(Intrinsic, real){
    auto images = collect_images("1");
    ASSERT_TRUE(images.size() > 0);

    const auto pattern_size = cv::Size(6, 8);
    const auto corners = detect(images, pattern_size, false);
    ASSERT_EQ(corners.size(), images.size());
}
