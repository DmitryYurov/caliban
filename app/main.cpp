#include <caliban/intrinsic.h>
#include "app_utils.h"

#include <argparse/argparse.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <random>

static void object_point_statistics(const std::vector<cv::Point3f>& lhs, const std::vector<cv::Point3f>& rhs) {
    if (lhs.size() != rhs.size()) {
        std::cerr << "Sizes of the point sets are different" << std::endl;
        return;
    }
    std::list<double> diffs;
    std::transform(lhs.begin(), lhs.end(), rhs.begin(), std::back_inserter(diffs),
                   [](const cv::Point3f& l, const cv::Point3f& r) { return (l - r).ddot(l - r); });
    auto max_it = std::max_element(diffs.begin(), diffs.end());
    const auto avr_diff = std::sqrt(std::accumulate(diffs.begin(), diffs.end(), 0.0) / diffs.size());

    std::cout << "Max difference from expected: " << std::sqrt(*max_it) << " at index "
              << std::distance(diffs.begin(), max_it) << std::endl;
    std::cout << "Average difference from expected: " << avr_diff << std::endl;
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("calib_app", "0.0.0");
    program.add_argument("-p", "--path")
        .help("Path to the folder with calibration images. Only .png files will be picked.")
        .required()
        .action([](const std::string& value) { return std::filesystem::absolute(std::filesystem::path(value)); });
    program.add_argument("-n", "--number_of_images")
        .help("Number of images to use for calibration. Default is 10.")
        .default_value(10)
        .action([](const std::string& value) { return std::stoi(value); });
    program.add_argument("-s", "--seed")
        .help("Random seed for image selection. Defaults to 1234")
        .default_value(1234)
        .action([](const std::string& value) { return std::stoi(value); });
    program.add_argument("-o", "--output")
        .help(
            "Output path for calibration intermediate and final data. Stays empty by default, which means no data will "
            "be saved on disk.")
        .default_value(std::optional<std::filesystem::path>{})
        .action([](const std::string& value) { return std::optional{std::filesystem::path(value)}; });
    program.add_argument("-c", "--checkerboard_dimensions")
        .help("Dimensions of the checkerboard pattern.")
        .default_value(cv::Size{14, 9})
        .action([](const std::string& value) {
            cv::Size pattern_size;
            std::stringstream ss(value);
            ss >> pattern_size.width;
            ss.ignore(1);
            ss >> pattern_size.height;
            return pattern_size;
        });
    program.add_argument("-r", "--no-radon")
        .help("Use a standard checkerboard instead of the one with radon markers.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("-h", "--help").help("Print this help message.").default_value(false).implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        std::cout << program;
        return 1;
    }

    // step 0: prepare the output directory
    const auto output_dir = program.get<std::optional<std::filesystem::path>>("--output");
    if (output_dir.has_value()) {
        if (!std::filesystem::exists(*output_dir)) {
            std::filesystem::create_directories(*output_dir);
        }
    }

    const auto pattern_size = program.get<cv::Size>("--checkerboard_dimensions");
    std::cout << "Checkerboard dimensions: " << pattern_size << std::endl;

    // step 1: loading the image list and taking n_images random images
    std::vector<cv::Mat> images;
    std::vector<std::filesystem::path> image_list;
    {
        const auto image_dir = program.get<std::filesystem::path>("--path");
        for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
            if (entry.path().extension() == ".png") {
                image_list.push_back(entry.path());
            }
        }

        if (image_list.empty()) {
            std::cerr << "No .png files found in the folder " << image_dir << std::endl;
            return 1;
        }
        std::sort(image_list.begin(), image_list.end());

        const auto n_images = program.get<int>("--number_of_images");
        if (n_images > image_list.size()) {
            std::cerr << "Number of images to use for calibration is greater than the number of images in the folder."
                      << std::endl;
            return 1;
        }
        if (n_images < 4) {
            std::cerr << "Number of images to use for calibration should be at least 4." << std::endl;
            return 1;
        }

        const auto seed = program.get<int>("--seed");
        std::mt19937 rng(seed);
        std::shuffle(image_list.begin(), image_list.end(), rng);

        for (size_t i = 0; i < n_images; ++i) {
            images.push_back(cv::imread(image_list[i].string(), cv::IMREAD_GRAYSCALE));
        }

        if (output_dir.has_value()) {
            for (size_t i = 0; i < images.size(); ++i) {
                cv::imwrite(output_dir->string() + "/image_" + std::to_string(i) + ".png", images[i]);
            }
        }
    }

    // step 2: detect chessboard corners in the images and drop the images in case of failure
    std::vector<std::vector<cv::Point2f>> corners{};
    {
        constexpr int default_flags = cv::CALIB_CB_ACCURACY | cv::CALIB_CB_EXHAUSTIVE;
        const int detection_flags =
            program.get<bool>("--no-radon") ? default_flags : int(cv::CALIB_CB_MARKER | default_flags);
        std::set<size_t> images_to_remove;
        for (size_t i = 0; i < images.size(); ++i) {
            std::vector<cv::Point2f> corners_per_image;
            if (!cv::findChessboardCornersSB(images[i], pattern_size, corners_per_image, detection_flags)) {
                std::cerr << "Chessboard pattern for image " << image_list[i].string() << " not found" << std::endl;
                images_to_remove.insert(i);
                continue;
            }

            if (output_dir.has_value()) {
                cv::Mat image_with_corners;
                cv::cvtColor(images[i], image_with_corners, cv::COLOR_GRAY2BGR);
                cv::drawChessboardCorners(image_with_corners, pattern_size, corners_per_image, true);
                cv::imwrite((*output_dir / ("image_" + std::to_string(i) + "_corners.png")).string(),
                            image_with_corners);
            }

            corners.push_back(std::move(corners_per_image));
        }

        // cleaning up the image list
        std::vector<cv::Mat> filtered_images;
        for (size_t i = 0; i < images.size(); ++i) {
            if (images_to_remove.contains(i)) {
                continue;
            }
            filtered_images.push_back(images[i]);
        }

        images = std::move(filtered_images);
    }

    if (images.size() < 4) {  // FIXME: condition on the number of point measurements
        std::cerr << "Not enough data for calibration" << std::endl;
        return 1;
    }

    // step 3: fill the object points
    // absolute scale is not important, only the relative positions
    std::vector<cv::Point3f> target_points{};
    {
        for (int i = 0; i < pattern_size.height; ++i) {
            for (int j = 0; j < pattern_size.width; ++j) {
                target_points.push_back(cv::Point3f(j, i, 0));
            }
        }
    }

    // step 4.0: calibrate with opencv
    {
        std::cout << std::endl << "Calibration with cv::calibrateCameraRO" << std::endl;

        std::vector<std::vector<cv::Point3f>> object_points(images.size(), target_points);

        const int fixed_index = pattern_size.width * (pattern_size.height - 1);
        auto camera_matrix = cv::Matx<double, 3, 3>();
        auto dist_coeffs = cv::Vec<double, 5>();
        std::vector<cv::Mat> rvecs, tvecs;
        std::vector<cv::Point3f> new_points;

        const auto rms = cv::calibrateCameraRO(object_points, corners, images[0].size(), fixed_index, camera_matrix,
                                               dist_coeffs, rvecs, tvecs, new_points);

        std::cout << "RMS Reprojection: " << rms << std::endl;
        std::cout << "Camera matrix: " << camera_matrix << std::endl;
        std::cout << "Distortion coefficients: " << dist_coeffs << std::endl;

        object_point_statistics(object_points[0], new_points);

        if (output_dir.has_value()) {
            cv::FileStorage fs((*output_dir / "calibration_data_opencv_ro.yml").string(), cv::FileStorage::WRITE);
            fs << "rms" << rms;
            fs << "camera_matrix" << camera_matrix;
            fs << "dist_coeffs" << dist_coeffs;
            fs.release();
        }
    }

    // step 4: calibrate the camera with ceres-based solver
    {
        std::cout << std::endl << "Calibration with ceres-based solver" << std::endl;

        std::vector<std::map<size_t, cv::Point2f>> image_points;
        for (const auto& cp : corners) {
            std::map<size_t, cv::Point2f> image_points_per_image;
            for (size_t i = 0; i < cp.size(); ++i) {
                image_points_per_image[i] = cp[i];
            }
            image_points.push_back(std::move(image_points_per_image));
        }

        const auto intrinsic_result = caliban::calibrate_intrinsics(target_points, image_points, images[0].size());

        std::cout << "RMS Reprojection: " << intrinsic_result.rms_repro << std::endl;
        std::cout << "Camera matrix: " << intrinsic_result.camera_matrix << std::endl;
        std::cout << "Distortion coefficients: " << intrinsic_result.dist_coeffs << std::endl;

        object_point_statistics(target_points, intrinsic_result.target_points);

        if (output_dir.has_value()) {
            cv::FileStorage fs((*output_dir / "calibration_data_ceres.yml").string(), cv::FileStorage::WRITE);
            fs << "rms" << intrinsic_result.rms_repro;
            fs << "camera_matrix" << intrinsic_result.camera_matrix;
            fs << "dist_coeffs" << intrinsic_result.dist_coeffs;
            fs.release();

            // ply file for checherboard visualization
            auto ply_file = std::ofstream((*output_dir / "checkerboard.ply").string());
            ply_file << caliban::make_ply(intrinsic_result.target_points, pattern_size);
        }
    }

    return 0;
}