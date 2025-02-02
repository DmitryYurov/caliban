#include <filesystem>
#include <random>

#include <argparse/argparse.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>

#include <caliban/intrinsic.h>

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("calib_app", "0.0.0");
    program.add_argument("-p", "--path")
        .help("Path to the folder with calibration images. Only .png files will be picked.")
        .required()
        .action([](const std::string &value) { return std::filesystem::absolute(std::filesystem::path(value)); });
    program.add_argument("-n", "--number_of_images")
        .help("Number of images to use for calibration. Default is 10.")
        .default_value(10)
        .action([](const std::string &value) { return std::stoi(value); });
    program.add_argument("-s", "--seed")
        .help("Random seed for image selection. Defaults to 1234")
        .default_value(1234)
        .action([](const std::string &value) { return std::stoi(value); });
    program.add_argument("-o", "--output")
        .help("Output path for calibration intermediate and final data. Stays empty by default, which means no data will be saved on disk.")
        .action([](const std::string &value) {
            return value.empty()
                ? std::optional<std::filesystem::path>{}
                : std::optional{std::filesystem::path(value)};
            });
    program.add_argument("-c", "--checkerboard_dimensions")
        .help("Dimensions of the checkerboard pattern. Default is 14x9.")
        .default_value(cv::Size{14, 9})
        .action([](const std::string &value) {
            cv::Size pattern_size;
            std::stringstream ss(value);
            ss >> pattern_size.width;
            ss.ignore(1);
            ss >> pattern_size.height;
            return pattern_size;
        });
    program.add_argument("-h", "--help")
        .help("Print this help message.")
        .default_value(false)
        .implicit_value(true);

    try {
        program.parse_args(argc, argv);
    }
    catch(const std::exception& ex) {
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
    {
        const auto image_dir = program.get<std::filesystem::path>("--path");
        std::vector<std::filesystem::path> image_list;
        for (const auto &entry : std::filesystem::directory_iterator(image_dir)) {
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
            std::cerr << "Number of images to use for calibration is greater than the number of images in the folder." << std::endl;
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

    // step 2: detect chessboard corners in the images
    std::vector<std::vector<cv::Point2f>> corners{};
    {
        constexpr int detection_flags = cv::CALIB_CB_MARKER | cv::CALIB_CB_ACCURACY;
        for (size_t i = 0; i < images.size(); ++i) {
            std::vector<cv::Point2f> corners_per_image;
            if (!cv::findChessboardCornersSB(images[i], pattern_size, corners_per_image, detection_flags)) {
                std::cerr << "Chessboard pattern for image " << i << " not found" << std::endl;
                continue;
            }

            if (output_dir.has_value()) {
                cv::Mat image_with_corners;
                cv::cvtColor(images[i], image_with_corners, cv::COLOR_GRAY2BGR);
                cv::drawChessboardCorners(image_with_corners, pattern_size, corners_per_image, true);
                cv::imwrite((*output_dir / ("image_" + std::to_string(i) + "_corners.png")).string(), image_with_corners);
            }

            corners.push_back(std::move(corners_per_image));
        }
    }

    if (corners.size() != images.size()) {
        std::cerr << "Not all images have the chessboard pattern detected." << std::endl;
        return 1;
    }

    // step 3: fill the object points
    // absolute scale is not important, only relative positions are important
    std::vector<std::vector<cv::Point3f>> object_points{};
    {
        std::vector<cv::Point3f> op_local{};
        for (int i = 0; i < pattern_size.height; ++i) {
            for (int j = 0; j < pattern_size.width; ++j) {
                op_local.push_back(cv::Point3f(j, i, 0));
            }
        }

        object_points.resize(images.size(), op_local);
    }

    // step 3: calibrate the camera with OpenCV standard calibration method
    {
        auto camera_matrix = cv::Matx<double, 3, 3>();
        auto dist_coeffs = cv::Vec<double, 5>();
        std::vector<cv::Mat> rvecs, tvecs;
        const auto rms = cv::calibrateCamera(object_points, corners, images[0].size(), camera_matrix, dist_coeffs, rvecs, tvecs);

        std::cout << std::endl << "Calibration with cv::calibrateCamera" << std::endl;
        std::cout << "RMS Reprojection: " << rms << std::endl;
        std::cout << "Camera matrix: " << camera_matrix << std::endl;
        std::cout << "Distortion coefficients: " << dist_coeffs << std::endl;

        if (output_dir.has_value()) {
            cv::FileStorage fs((*output_dir / "calibration_data_opencv.yml").string(), cv::FileStorage::WRITE);
            fs << "rms" << rms;
            fs << "camera_matrix" << camera_matrix;
            fs << "dist_coeffs" << dist_coeffs;
            fs.release();
        }
    }

    // step 4: calibrate the camera with OpenCV release object method
    {
        const int fixed_index = pattern_size.width * (pattern_size.height - 1);
        auto camera_matrix = cv::Matx<double, 3, 3>();
        auto dist_coeffs = cv::Vec<double, 5>();
        std::vector<cv::Mat> rvecs, tvecs;
        std::vector<cv::Point3f> new_points;
        const auto rms = cv::calibrateCameraRO(object_points, corners, images[0].size(), fixed_index, camera_matrix, dist_coeffs, rvecs, tvecs, new_points);

        std::cout << std::endl << "Calibration with cv::calibrateCameraRO" << std::endl;
        std::cout << "RMS Reprojection: " << rms << std::endl;
        std::cout << "Camera matrix: " << camera_matrix << std::endl;
        std::cout << "Distortion coefficients: " << dist_coeffs << std::endl;

        if (output_dir.has_value()) {
            cv::FileStorage fs((*output_dir / "calibration_data_opencv_ro.yml").string(), cv::FileStorage::WRITE);
            fs << "rms" << rms;
            fs << "camera_matrix" << camera_matrix;
            fs << "dist_coeffs" << dist_coeffs;
            fs.release();
        }
    }

    // step 5: calibrate the camera with ceres-based solver
    {
        std::vector<caliban::Point3D> target_points;
        for (const auto &op : object_points[0]) {
            target_points.push_back({op.x, op.y, op.z});
        }

        std::vector<std::map<size_t, caliban::Point2D> > image_points;
        for (const auto &cp : corners) {
            std::map<size_t, caliban::Point2D> image_points_per_image;
            for (size_t i = 0; i < cp.size(); ++i) {
                image_points_per_image[i] = {cp[i].x, cp[i].y};
            }
            image_points.push_back(std::move(image_points_per_image));
        }

        auto camera_matrix = cv::initCameraMatrix2D(object_points, corners, images[0].size(), 1.0);
        caliban::CameraMatrix camera_matrix_{camera_matrix.at<double>(0, 0), camera_matrix.at<double>(0, 2), camera_matrix.at<double>(1, 1), camera_matrix.at<double>(1, 2)};
        caliban::DistortionCoefficients distortion_coefficients_{0, 0, 0, 0, 0};
        std::vector<caliban::QuatSE3> obj_2_cams;
        for (size_t i = 0; i < images.size(); ++i) {
            cv::Mat rvec, tvec;
            cv::solvePnP(object_points[i], corners[i], camera_matrix, cv::noArray(), rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
            auto q = cv::Quat<double>::createFromRvec(rvec);
            obj_2_cams.push_back({q.w, q.x, q.y, q.z, tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0)});
        }
        caliban::calibrate(target_points, image_points, camera_matrix_, distortion_coefficients_, obj_2_cams);
    }

    return 0;
}