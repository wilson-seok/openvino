// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/aarch64/cpu_isa_traits.hpp>
#include <cpu/aarch64/jit_generator.hpp>
#include <cstddef>
#include <memory>
#include <vector>

#include "jit_emitter.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::aarch64 {

class jit_convert_emitter : public jit_emitter {
public:
    jit_convert_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                        dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                        const std::shared_ptr<ov::Node>& n,
                        ov::element::Type exec_prc = ov::element::f32);

    size_t get_inputs_count() const override;

protected:
    void emit_data() const override;
    void validate_types() const;
    template <typename TReg>
    void jit_convert_process(const TReg& src,
                             const TReg& dst,
                             ov::element::Type input_type,
                             ov::element::Type output_type,
                             bool is_saturated) const;

    ov::element::Type input_type;
    ov::element::Type output_type;

private:
    template <typename TReg>
    inline void cvt_f16_to_f32(const TReg& src, const TReg& dst) const;
    template <typename TReg>
    inline void cvt_f32_to_f16(const TReg& src, const TReg& dst) const;
    template <typename TReg>
    inline void cvt_f32_to_i32(const TReg& src, const TReg& dst) const;
    template <typename TReg>
    inline void cvt_i32_to_f32(const TReg& src, const TReg& dst) const;
    template <typename TReg>
    inline void cvt_i32_to_i16(const TReg& src, const TReg& dst, bool is_saturated) const;
    template <typename TReg>
    inline void cvt_i16_to_i32(const TReg& src, const TReg& dst) const;
    template <typename TReg>
    inline void cvt_f16_to_i16(const TReg& src, const TReg& dst) const;
    template <typename TReg>
    inline void cvt_i16_to_f16(const TReg& src, const TReg& dst) const;
    template <typename TReg>
    inline void cvt_i16_to_byte(const TReg& src, const TReg& dst, bool is_signed, bool is_saturated) const;
    template <typename TReg>
    inline void cvt_byte_to_i16(const TReg& src, const TReg& dst, bool is_signed) const;
};

// This emitter is covered by specification of "Convert" operation. The implementation uses a "warp-around" conversion.
// Example:
//  int32_t -> int8_t
//   129   -> -127
class jit_convert_truncation_emitter : public jit_convert_emitter {
public:
    jit_convert_truncation_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& n,
                                   ov::element::Type exec_prc = ov::element::f32);

private:
    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const;
};

// This emitter is covered by the common dnnl behavior. The implementation uses a "saturation" conversion.
// Example:
//  int32_t -> int8_t
//   129   -> 127
class jit_convert_saturation_emitter : public jit_convert_emitter {
public:
    jit_convert_saturation_emitter(dnnl::impl::cpu::aarch64::jit_generator* host,
                                   dnnl::impl::cpu::aarch64::cpu_isa_t host_isa,
                                   const std::shared_ptr<ov::Node>& n,
                                   ov::element::Type exec_prc = ov::element::f32);

private:
    void emit_impl(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const override;
    template <dnnl::impl::cpu::aarch64::cpu_isa_t isa>
    void emit_isa(const std::vector<size_t>& in_idxs, const std::vector<size_t>& out_idxs) const;
};

}  // namespace ov::intel_cpu::aarch64
