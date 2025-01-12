#include <array>
#include <vector>

namespace caliban {
    using CameraMatrix = std::array<double, 9>;
    using Point3D = std::array<double, 3>;
    using Point2D = std::array<double, 2>;
    void calibrate(
        const std::vector<Point3D>& target_points,
        const std::vector<Point2D>& image_points,
        const CameraMatrix& camera_matrix,
        const std::vector<double>& distortion_coefficients
    );
}