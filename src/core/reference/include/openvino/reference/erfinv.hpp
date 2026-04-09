// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {
namespace func {

// Mike Giles' rational polynomial approximation for erfinv.
// Reference: "Approximating the erfinv function", GPU Computing Gems, Chapter 10.
// Accuracy: ~1e-7 relative error for float32.
template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T erfinv(const T v) {
    float x = static_cast<float>(v);
    float ax = std::fabs(x);

    if (ax >= 1.0f) {
        if (ax == 1.0f) {
            return static_cast<T>(std::copysign(std::numeric_limits<float>::infinity(), x));
        }
        return static_cast<T>(std::numeric_limits<float>::quiet_NaN());
    }

    float w, p;
    w = -std::log(1.0f - x * x);
    if (w < 5.0f) {
        w = w - 2.5f;
        p = ((((((((2.81022636e-08f * w + 3.43273939e-07f) * w
            + (-3.5233877e-06f)) * w + (-4.39150654e-06f)) * w
            + 0.00021858087f) * w + (-0.00125372503f)) * w
            + (-0.00417768164f)) * w + 0.246640727f) * w
            + 1.50140941f) * x;
    } else {
        w = std::sqrt(w) - 3.0f;
        p = ((((((((-0.000200214257f * w
            + 0.000100950558f) * w + 0.00134934322f) * w
            + (-0.00367342844f)) * w + 0.00573950773f) * w
            + (-0.0076224613f)) * w + 0.00943887047f) * w
            + 1.00167406f) * w + 2.83297682f) * x;
    }
    return static_cast<T>(p);
}

template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T erfinv(const T v) {
    return static_cast<T>(std::round(erfinv(static_cast<float>(v))));
}

}  // namespace func

/**
 * @brief Reference implementation of ErfInv operator.
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param count  Number of elements in input buffer.
 */
template <class T>
void erfinv(const T* arg, T* out, const size_t count) {
    std::transform(arg, arg + count, out, func::erfinv<T>);
}
}  // namespace reference
}  // namespace ov
