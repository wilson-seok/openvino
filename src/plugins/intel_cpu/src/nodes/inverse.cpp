// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "inverse.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "cpu_types.h"
#include "graph_context.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/common/cpu_memcpy.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/inverse.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

namespace ov::intel_cpu::node {

Inverse::Inverse(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    auto inverse_op = as_type_ptr<op::v14::Inverse>(op);
    m_adjoint = inverse_op->get_adjoint();

    constant = ConstantType::StrictNoConst;

    m_const_input = is_type<op::v0::Constant>(op->get_input_node_ptr(INPUT_PORT));
}

bool Inverse::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v14::Inverse::get_type_info_static()) {
            errorMessage = "Only Inverse operation from the opset14 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void Inverse::getSupportedDescriptors() {
    CPU_NODE_ASSERT(getParentEdges().size() == 1, "has incorrect number of input edges.");
    CPU_NODE_ASSERT(!getChildEdges().empty(), "has incorrect number of output edges.");
}

void Inverse::initSupportedPrimitiveDescriptors() {
    m_input_precision = getOriginalInputPrecisionAtPort(INPUT_PORT);
    if (m_input_precision != ov::element::f32) {
        m_input_precision = ov::element::f32;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, m_input_precision, m_const_input}},
                         {{LayoutType::ncsp, m_input_precision}},
                         ref_any);
}

void Inverse::prepareParams() {
    const auto& input_shape = getParentEdgeAt(INPUT_PORT)->getMemory().getStaticDims();

    CPU_NODE_ASSERT(input_shape.size() >= 2,
                    "has incompatible 'data' shape ",
                    PartialShape(input_shape),
                    ". Only tensors of rank at least 2 are allowed.");

    m_side = input_shape.back();
    m_side_squared = m_side * m_side;
    m_batches_count = 1;

    for (size_t i = 0; i < input_shape.size() - 2; ++i) {
        m_batches_count = m_batches_count * input_shape[i];
    }
}

bool Inverse::created() const {
    return getType() == Type::Inverse;
}

void Inverse::execute([[maybe_unused]] const dnnl::stream& strm) {
    inverse();
}

void Inverse::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void Inverse::inverse() {
    const auto* data = getSrcDataAtPortAs<const float>(INPUT_PORT);
    auto* output = getDstDataAtPortAs<float>(OUTPUT_PORT);

    std::vector<float> L(m_side_squared);
    std::vector<float> U(m_side_squared);
    std::vector<size_t> P(m_side);

    for (size_t b = 0; b < m_batches_count; ++b) {
        lu_decomposition(data, L, U, P, b);
        lu_solve(output, L, U, P, b);
    }
}

void Inverse::lu_decomposition(const float* data,
                               std::vector<float>& L,
                               std::vector<float>& U,
                               std::vector<size_t>& P,
                               size_t b) const {
    // Make L identity, U a copy of data and P a range(0, side)
    const auto batch_idx = b * m_side_squared;

    std::fill(L.begin(), L.end(), 0.0F);
    if (!m_adjoint) {
        cpu_parallel_memcpy(U.data(), &data[batch_idx], sizeof(float) * m_side_squared);
    } else {
        parallel_for2d(m_side, m_side, [&](size_t i, size_t j) {
            U[j * m_side + i] = data[batch_idx + i * m_side + j];
        });
    }

    parallel_for(m_side, [&](size_t i) {
        L[i * m_side + i] = 1.0F;
        P[i] = i;
    });

    for (size_t k = 0; k < m_side; ++k) {
        // Partial Pivoting
        auto pivot_row = k;
        auto pivot_idx = pivot_row * m_side;
        const auto k_idx = k * m_side;

        // Find maximum value pivot - non-parallel
        for (size_t i = (k + 1) * m_side, j = k + 1; i < m_side_squared; i += m_side, ++j) {
            if (std::abs(U[i + k]) > std::abs(U[pivot_idx + k])) {
                pivot_row = j;
                pivot_idx = pivot_row * m_side;
            }
        }

        if (pivot_row != k) {
            // Swap rows in L, U and P
            std::swap(P[k], P[pivot_row]);
            parallel_for(m_side, [&](size_t i) {
                std::swap(L[k_idx + i], L[pivot_idx + i]);
                std::swap(U[k_idx + i], U[pivot_idx + i]);
            });
        }

        const auto remaining_columns = m_side - k;
        const auto remaining_rows = remaining_columns - 1;

        parallel_for(remaining_rows, [&](size_t i) {
            const auto i_idx = (i + k + 1) * m_side;
            L[i_idx + k] = U[i_idx + k] / U[k_idx + k];
        });

        parallel_for(remaining_rows * remaining_columns, [&](size_t i) {
            const auto i_idx = (i / remaining_columns + k + 1) * m_side;
            const auto j_idx = i % remaining_columns + k;
            U[i_idx + j_idx] = U[i_idx + j_idx] - L[i_idx + k] * U[k_idx + j_idx];
        });
    }
}

void Inverse::lu_solve(float* output,
                       std::vector<float>& L,
                       std::vector<float>& U,
                       std::vector<size_t>& P,
                       size_t b) const {
    parallel_for(m_side, [&](size_t column) {
        std::vector<float> X(m_side, 0.0F);
        std::vector<float> Y(m_side, 0.0F);

        // Forward substitution: Ly = Pb
        for (size_t i = 0; i < m_side; ++i) {
            if (P[i] == column) {
                Y[i] = 1.0F;
            }
            const auto i_idx = i * m_side;
            for (size_t j = 0; j < i; ++j) {
                Y[i] = Y[i] - L[i_idx + j] * Y[j];
            }
        }

        // Backward substitution: Ux = y
        for (size_t i = 0; i < m_side; ++i) {
            size_t i_adj = m_side - i - 1;
            size_t i_idx = i_adj * m_side;
            X[i_adj] = Y[i_adj];
            for (size_t j = i_adj + 1; j < m_side; ++j) {
                X[i_adj] = X[i_adj] - U[i_idx + j] * X[j];
            }
            X[i_adj] = X[i_adj] / U[i_idx + i_adj];
        }

        // Substitute back to get result
        const auto batch_column_idx = b * m_side_squared + column;
        for (size_t row = 0; row < m_side; ++row) {
            output[batch_column_idx + row * m_side] = X[row];
        }
    });
}

}  // namespace ov::intel_cpu::node
