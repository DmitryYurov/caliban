#ifndef CALIBAN_UTILS_H
#define CALIBAN_UTILS_H

#include <ceres/rotation.h>

#include "caliban/constants.h"

namespace caliban {
/**
 * @brief   Computes the product of two SE3 transformations.
 *
 * a and b should not have overlapping memory regions.
 *
 * @param a     The first transformation (array of n_quat_se3 size, quaternion + Cartesian vector representation)
 * @param b     The second transformation (array of n_quat_se3 size, quaternion + Cartesian vector representation)
 * @param out   The output transformation (array of n_quat_se3 size, quaternion + Cartesian vector representation)
 */
template <typename T>
void MultiplySE3(const T* const a, const T* const b, T* out)
{
    ceres::QuaternionProduct(a, b, out);
    ceres::QuaternionRotatePoint(a, b + n_quat_so3, out + n_quat_so3);

    out[n_quat_so3] += a[n_quat_so3];
    out[n_quat_so3 + 1] += a[n_quat_so3 + 1];
    out[n_quat_so3 + 2] += a[n_quat_so3 + 2];
}

/**
 * @brief   Computes the inverse of the SE3 transformation.
 *
 * in and out should not have overlapping memory regions.
 *
 * @param in    The input transformation (array of n_quat_se3 size, quaternion + Cartesian vector representation)
 * @param out   The output transformation (array of n_quat_se3 size, quaternion + Cartesian vector representation)
 */
template <typename T>
void InverseSE3(const T* const in, T* out)
{
    out[0] = -in[0];
    out[1] = in[1];
    out[2] = in[2];
    out[3] = in[3];

    ceres::QuaternionRotatePoint(out, in + n_quat_so3, out + n_quat_so3);

    out[n_quat_so3]     = -out[n_quat_so3];
    out[n_quat_so3 + 1] = -out[n_quat_so3 + 1];
    out[n_quat_so3 + 2] = -out[n_quat_so3 + 2];
}
}

#endif // CALIBAN_UTILS_H
