// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cinttypes>
#include <sstream>
#include <string>
#include <utility>

#include "openvino/core/except.hpp"

namespace ov::intel_gpu {

template <typename T>
std::string to_code_string(T val) {
    std::stringstream ss;
    ss.imbue(std::locale("C"));
    ss << val;
    return ss.str();
}

// 18 - Representation of a double of maximum length in hexadecimal notation
// 11 - as_double()
// 17 - Commented representation of the maximum double in scientific .6e notation /*1.797693e+308*/
static thread_local char buf[18 + 11 + 17] = "";

inline std::string to_code_string(const std::string& val) {
    return val;
}

inline std::string to_code_string(const char* val) {
    return val;
}

inline std::string to_code_string(bool val) {
    return val ? "1" : "0";
}

inline std::string to_code_string(size_t val) {
    snprintf(buf, sizeof(buf), "%zu", val);
    return buf;
}

inline std::string to_code_string(uint8_t val) {
    snprintf(buf, sizeof(buf), "%d", static_cast<int>(val));
    return buf;
}

inline std::string to_code_string(int8_t val) {
    snprintf(buf, sizeof(buf), "%d", static_cast<int>(val));
    return buf;
}

inline std::string to_code_string(float val) {
    if (std::isinf(val)) {
        return std::signbit(val) ? "-INFINITY" : "INFINITY";
    }
    // Workaround GCC compiler/STL bug
#ifdef GPU_DEBUG_CONFIG
    snprintf(buf, sizeof(buf), "as_float(0x%" PRIx32 ")/*%.6e*/", *reinterpret_cast<uint32_t*>(&val), val);
#else
    snprintf(buf, sizeof(buf), "as_float(0x%" PRIx32 ")", *reinterpret_cast<uint32_t*>(&val));
#endif
    return buf;
}

inline std::string to_code_string(double val) {
    if (std::isinf(val)) {
        return std::signbit(val) ? "-INFINITY" : "INFINITY";
    }
    // Workaround GCC compiler/STL bug
#ifdef GPU_DEBUG_CONFIG
    snprintf(buf, sizeof(buf), "as_double(0x%" PRIx64 ")/*%.6e*/", *reinterpret_cast<uint64_t*>(&val), val);
#else
    snprintf(buf, sizeof(buf), "as_double(0x%" PRIx64 ")", *reinterpret_cast<uint64_t*>(&val));
#endif
    return buf;
}

class JitTerm {
public:
    JitTerm() = default;
    template <typename T,
              std::enable_if_t<!std::is_same_v<T, std::string> && !std::is_same_v<T, std::string_view> && !std::is_same_v<T, const char*>, bool> = true>
    explicit JitTerm(const T& v) : text(to_code_string(v)) {}

    explicit JitTerm(std::string v) : text(std::move(v)) {}

    [[nodiscard]] const std::string& str() const {
        return text;
    }
    [[nodiscard]] JitTerm gt(const JitTerm& rhs) const {
        return JitTerm{"(" + text + ">" + rhs.str() + ")"};
    }
    [[nodiscard]] JitTerm ge(const JitTerm& rhs) const {
        return JitTerm{"(" + text + ">=" + rhs.str() + ")"};
    }
    [[nodiscard]] JitTerm le(const JitTerm& rhs) const {
        return JitTerm{"(" + text + "<=" + rhs.str() + ")"};
    }
    [[nodiscard]] JitTerm lt(const JitTerm& rhs) const {
        return JitTerm{"(" + text + "<" + rhs.str() + ")"};
    }
    [[nodiscard]] JitTerm eq(const JitTerm& rhs) const {
        return JitTerm{"(" + text + "==" + rhs.str() + ")"};
    }
    [[nodiscard]] JitTerm ne(const JitTerm& rhs) const {
        return JitTerm{"(" + text + "!=" + rhs.str() + ")"};
    }
    [[nodiscard]] JitTerm assign(const JitTerm& rhs) const {
        return JitTerm{text + " = " + rhs.str()};
    }
    [[nodiscard]] JitTerm body(const JitTerm& rhs) const {
        return JitTerm{text + "{\n" + rhs.str() + "\n}"};
    }

    template <typename... Args>
    JitTerm operator()(Args&&... args) const {
        return JitTerm{text + "(" + concat(",", std::forward<Args>(args)...).str() + ")"};
    }

    JitTerm operator[](const JitTerm& idx) const {
        return JitTerm{text + "[" + idx.str() + "]"};
    }
    JitTerm operator[](size_t idx) const {
        return JitTerm{text + "[" + to_code_string(idx) + "]"};
    }

    template <typename T1, typename... Args>
    [[nodiscard]] static JitTerm concat(const std::string& separator, const T1& first, const Args&... args) {
        std::ostringstream oss;
        oss << first;
        ((oss << separator << args), ...);
        return JitTerm{oss.str()};
    }

private:
    std::string text;
};

template <typename... Args>
inline JitTerm concat(Args&&... args) {
    return JitTerm::concat("", std::forward<Args>(args)...);
}

inline std::ostream& operator<<(std::ostream& os, const JitTerm& t) {
    return os << t.str();
}

inline bool is_number(const JitTerm& s) {
    return !s.str().empty() && std::all_of(s.str().begin(), s.str().end(), ::isdigit);
}
template <typename T>
inline T as_number(const JitTerm& s) {
    T val;
    std::stringstream ss(s.str());
    ss >> val;
    return val;
}

inline JitTerm neg(const JitTerm& arg) {
    return JitTerm{"(-" + arg.str() + ")"};
}
inline JitTerm operator+(const JitTerm& lhs, const JitTerm& rhs) {
    if (lhs.str() == "0") {
        return rhs;
    }
    if (rhs.str() == "0") {
        return lhs;
    }
    if (is_number(lhs) && is_number(rhs)) {
        return JitTerm{std::to_string(as_number<int64_t>(lhs) + as_number<int64_t>(rhs))};
    }

    return JitTerm{"(" + lhs.str() + " + " + rhs.str() + ")"};
}

inline JitTerm operator-(const JitTerm& lhs, const JitTerm& rhs) {
    if (lhs.str() == "0") {
        return neg(rhs);
    }
    if (rhs.str() == "0") {
        return lhs;
    }
    if (is_number(lhs) && is_number(rhs)) {
        return JitTerm{std::to_string(as_number<int64_t>(lhs) - as_number<int64_t>(rhs))};
    }

    return JitTerm{"(" + lhs.str() + " - " + rhs.str() + ")"};
}

inline JitTerm operator*(const JitTerm& lhs, const JitTerm& rhs) {
    if (lhs.str() == "0" || rhs.str() == "0") {
        return JitTerm{"0"};
    }
    if (lhs.str() == "1") {
        return rhs;
    }
    if (rhs.str() == "1") {
        return lhs;
    }
    if (is_number(lhs) && is_number(rhs)) {
        return JitTerm{std::to_string(as_number<int64_t>(lhs) * as_number<int64_t>(rhs))};
    }
    return JitTerm{"(" + lhs.str() + " * " + rhs.str() + ")"};
}
inline JitTerm operator/(const JitTerm& lhs, const JitTerm& rhs) {
    OPENVINO_ASSERT(rhs.str() != "0");
    if (rhs.str() == "1") {
        return lhs;
    }
    if (is_number(lhs) && is_number(rhs)) {
        auto rhs_val = as_number<int64_t>(rhs);
        OPENVINO_ASSERT(rhs_val != 0, "Division by zero detected in operator/");
        return JitTerm{std::to_string(as_number<int64_t>(lhs) / rhs_val)};
    }
    return JitTerm{"(" + lhs.str() + " / " + rhs.str() + ")"};
}
inline JitTerm operator%(const JitTerm& lhs, const JitTerm& rhs) {
    if (is_number(rhs)) {
        auto rhs_val = as_number<int64_t>(rhs);
        OPENVINO_ASSERT(rhs_val != 0, "Modulo by zero detected in operator%");
        if (rhs_val == 1 || rhs_val == -1) {
            return JitTerm{"0"};
        }
        if (is_number(lhs)) {
            return JitTerm{std::to_string(as_number<int64_t>(lhs) % rhs_val)};
        }
    }
    return JitTerm{"(" + lhs.str() + " % " + rhs.str() + ")"};
}
inline JitTerm operator++(JitTerm& t, int) {
    return JitTerm{t.str() + "++"};
}
inline JitTerm operator--(JitTerm& t, int) {
    return JitTerm{t.str() + "--"};
}
inline JitTerm operator-=(const JitTerm& a, const JitTerm& b) {
    return concat(a, " -= ", b);
}

inline JitTerm ternary(const JitTerm& condition, const JitTerm& true_expr, const JitTerm& false_expr) {
    return JitTerm{"(" + condition.str() + " ? " + true_expr.str() + " : " + false_expr.str() + ")"};
}
inline JitTerm isinf(const JitTerm& arg) {
    return JitTerm{"isinf(" + arg.str() + ")"};
}
inline JitTerm exp(const JitTerm& arg) {
    return JitTerm{"exp(" + arg.str() + ")"};
}
inline JitTerm erf(const JitTerm& arg) {
    return JitTerm{"erf(" + arg.str() + ")"};
}
inline JitTerm erfinv(const JitTerm& input, const std::string& type_suffix) {
    // Inverse error function using rational polynomial approximation
    // with 2 Newton-Raphson refinement steps (matching PyTorch calc_erfinv)
    const JitTerm zero = JitTerm{"0.0" + type_suffix};
    const JitTerm one = JitTerm{"1.0" + type_suffix};

    // Central range coefficients (|y| <= 0.7)
    const JitTerm a0{"0.886226899" + type_suffix};
    const JitTerm a1{"-1.645349621" + type_suffix};
    const JitTerm a2{"0.914624893" + type_suffix};
    const JitTerm a3{"-0.140543331" + type_suffix};
    const JitTerm b0{"-2.118377725" + type_suffix};
    const JitTerm b1{"1.442710462" + type_suffix};
    const JitTerm b2{"-0.329097515" + type_suffix};
    const JitTerm b3{"0.012229801" + type_suffix};

    // Tail range coefficients (0.7 < |y| < 1.0)
    const JitTerm c0{"-1.970840454" + type_suffix};
    const JitTerm c1{"-1.624906493" + type_suffix};
    const JitTerm c2{"3.429567803" + type_suffix};
    const JitTerm c3{"1.641345311" + type_suffix};
    const JitTerm d0{"3.543889200" + type_suffix};
    const JitTerm d1{"1.637067800" + type_suffix};

    const JitTerm central_range{"0.7" + type_suffix};
    const JitTerm two{"2.0" + type_suffix};
    const JitTerm neg_two{"-2.0" + type_suffix};
    const JitTerm two_over_sqrtpi{"1.1283791670955126" + type_suffix};
    const JitTerm inf_val{"INFINITY"};
    const JitTerm nan_val{"NAN"};

    // |y|
    const JitTerm abs_y = fabs(input);

    // Central range: z = y*y, Horner evaluation
    const JitTerm z_c = input * input;
    const JitTerm num_c = ((a3 * z_c + a2) * z_c + a1) * z_c + a0;
    const JitTerm dem_c = (((b3 * z_c + b2) * z_c + b1) * z_c + b0) * z_c + one;
    const JitTerm x_central = input * num_c / dem_c;

    // Tail range: z = sqrt(-2 * log((1 - |y|) / 2))
    const JitTerm z_t = sqrt(neg_two * log((one - fabs(input)) / two));
    const JitTerm num_t = ((c3 * z_t + c2) * z_t + c1) * z_t + c0;
    const JitTerm dem_t = (d1 * z_t + d0) * z_t + one;
    // copysign emulated via ternary
    const JitTerm x_tail_abs = num_t / dem_t;
    const JitTerm x_tail = ternary(input.ge(zero), x_tail_abs, neg(x_tail_abs));

    // Select central or tail region
    const JitTerm x0 = ternary(abs_y.le(central_range), x_central, x_tail);

    // Newton-Raphson refinement step 1
    const JitTerm x1 = x0 - (erf(x0) - input) / (two_over_sqrtpi * exp(neg(x0 * x0)));

    // Newton-Raphson refinement step 2
    const JitTerm x2 = x1 - (erf(x1) - input) / (two_over_sqrtpi * exp(neg(x1 * x1)));

    // Edge cases: |y| > 1 -> NaN, |y| == 1 -> copysign(inf, y)
    const JitTerm copysign_inf = ternary(input.ge(zero), inf_val, neg(inf_val));
    return ternary(abs_y.gt(one), nan_val,
                   ternary(abs_y.eq(one), copysign_inf, x2));
}
inline JitTerm sin(const JitTerm& arg) {
    return JitTerm{"sin(" + arg.str() + ")"};
}
inline JitTerm asin(const JitTerm& arg) {
    return JitTerm{"asin(" + arg.str() + ")"};
}
inline JitTerm sinh(const JitTerm& arg) {
    return JitTerm{"sinh(" + arg.str() + ")"};
}
inline JitTerm asinh(const JitTerm& arg) {
    return JitTerm{"asinh(" + arg.str() + ")"};
}
inline JitTerm cos(const JitTerm& arg) {
    return JitTerm{"cos(" + arg.str() + ")"};
}
inline JitTerm acos(const JitTerm& arg) {
    return JitTerm{"acos(" + arg.str() + ")"};
}
inline JitTerm cosh(const JitTerm& arg) {
    return JitTerm{"cosh(" + arg.str() + ")"};
}
inline JitTerm acosh(const JitTerm& arg) {
    return JitTerm{"acosh(" + arg.str() + ")"};
}
inline JitTerm tan(const JitTerm& arg) {
    return JitTerm{"tan(" + arg.str() + ")"};
}
inline JitTerm atan(const JitTerm& arg) {
    return JitTerm{"atan(" + arg.str() + ")"};
}
inline JitTerm tanh(const JitTerm& arg) {
    return JitTerm{"tanh(" + arg.str() + ")"};
}
inline JitTerm atanh(const JitTerm& arg) {
    return JitTerm{"atanh(" + arg.str() + ")"};
}
inline JitTerm log(const JitTerm& arg) {
    return JitTerm{"log(" + arg.str() + ")"};
}
inline JitTerm log2(const JitTerm& arg) {
    return JitTerm{"log2(" + arg.str() + ")"};
}
inline JitTerm round(const JitTerm& arg) {
    return JitTerm{"round(" + arg.str() + ")"};
}
inline JitTerm rint(const JitTerm& arg) {
    return JitTerm{"rint(" + arg.str() + ")"};
}
inline JitTerm floor(const JitTerm& arg) {
    return JitTerm{"floor(" + arg.str() + ")"};
}
inline JitTerm ceil(const JitTerm& arg) {
    return JitTerm{"ceil(" + arg.str() + ")"};
}
inline JitTerm sqrt(const JitTerm& arg) {
    return JitTerm{"sqrt(" + arg.str() + ")"};
}
inline JitTerm abs(const JitTerm& arg) {
    return JitTerm{"abs(" + arg.str() + ")"};
}
inline JitTerm fabs(const JitTerm& arg) {
    return JitTerm{"fabs(" + arg.str() + ")"};
}
inline JitTerm pow(const JitTerm& arg, const JitTerm& power) {
    return JitTerm{"pow(" + arg.str() + "," + power.str() + ")"};
}
inline JitTerm logical_and(const JitTerm& lhs, const JitTerm& rhs) {
    return JitTerm{"(" + lhs.str() + " && " + rhs.str() + ")"};
}
inline JitTerm logical_or(const JitTerm& lhs, const JitTerm& rhs) {
    return JitTerm{"(" + lhs.str() + " || " + rhs.str() + ")"};
}
inline JitTerm max(const JitTerm& lhs, const JitTerm& rhs) {
    return JitTerm{"max(" + lhs.str() + ", " + rhs.str() + ")"};
}
inline JitTerm min(const JitTerm& lhs, const JitTerm& rhs) {
    return JitTerm{"min(" + lhs.str() + ", " + rhs.str() + ")"};
}
inline JitTerm clamp(const JitTerm& val, const JitTerm& low, const JitTerm& high) {
    return JitTerm{"clamp(" + val.str() + ", " + low.str() + ", " + high.str() + ")"};
}
inline JitTerm for_loop(const JitTerm& init, const JitTerm& condition, const JitTerm& expression) {
    const JitTerm _for("for");
    return _for(JitTerm::concat("; ", init, condition, expression));
}
inline JitTerm operator"" _jit(const char* str, size_t /*unused*/) {
    return JitTerm{str};
}

}  // namespace ov::intel_gpu
