namespace caliban {
inline std::string make_ply(std::vector<cv::Point3f> target_points, cv::Size check_size) {
    constexpr std::string_view color = "0 255 0 175";  // RGBA

    const int w = check_size.width - 1;   // for computing n_faces only
    const int h = check_size.height - 1;  // for computing n_faces only
    const int n_faces = (h / 2) * w + (h % 2) * ((w + 1) / 2);

    std::stringstream result;
    result << "ply\n";
    result << "format ascii 1.0\n";
    result << "element vertex " << check_size.area() << std::endl;
    result << "property float x\n";
    result << "property float y\n";
    result << "property float z\n";
    result << "property uchar red\n";
    result << "property uchar green\n";
    result << "property uchar blue\n";
    result << "property uchar alpha\n";
    result << "element face " << n_faces << std::endl;
    result << "property list uchar int vertex_indices\n";
    result << "end_header" << std::endl;

    for (const auto& p : target_points) {
        result << p.x << " " << p.y << " " << p.z << " " << color << std::endl;
    }

    for (int i = 0; i + 1 < check_size.height; ++i) {
        const int offset = i % 2 == 0 ? 0 : 1;
        for (int j = 0; 2 * j + offset + 1 < check_size.width; ++j) {
            int i0 = i * check_size.width + 2 * j + offset;
            int i1 = i0 + 1;
            int i3 = (i + 1) * check_size.width + 2 * j + offset;
            int i2 = i3 + 1;
            result << "4 " << i0 << " " << i1 << " " << i2 << " " << i3 << std::endl;
        }
    }

    result << std::endl;
    return result.str();
}
}  // namespace caliban