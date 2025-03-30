#include "constants.h"

#include <Eigen/Dense>
#include <array>

namespace caliban {
/**
 * @brief Provides a Jacobian for a point in one coordinate system wrt to a point in another coordinate system.
 *
 * The result contains the first derivatives with respect to the transform parameters (Euler angles and translation) as
 * well as the derivatives with respect to the point itself.
 *
 */
Eigen::Matrix<double, n_se3 + n_r3, n_r3> get_transform_jaco(std::array<double, n_se3> transform,
                                                             std::array<double, n_r3> point);
}  // namespace caliban
