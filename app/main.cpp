#include <filesystem>

#include <argparse/argparse.hpp>
#include <opencv2/opencv.hpp>

#include <caliban/intrinsic.h>

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("caliba_app", "0.0.0");
    program.add_argument("-p", "--path")
        .help("Path to the folder with calibration images. Only .png files will be picked.")
        .required()
        .action([](const std::string &value) { return std::filesystem::path(value); });
    program.add_argument("-n", "--number_of_images")
        .help("Number of images to use for calibration. Default is 10.")
        .default_value(10)
        .action([](const std::string &value) { return std::stoi(value); });
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

    return 0;
}