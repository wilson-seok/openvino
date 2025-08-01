// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace npuw {

// Model optimization patterns. Triggered by the plugin at the very top
namespace patterns {
namespace opt {

struct Context {
    std::string pmm_dims;
    bool is_spatial = false;
    bool mm_dq_full = true;

    using PPtr = std::shared_ptr<ov::op::v0::Parameter>;
    using NPtr = std::shared_ptr<ov::Node>;
    using CPtr = std::shared_ptr<ov::op::v0::Constant>;

    using Axes = std::vector<std::size_t>;
    std::map<PPtr, Axes> closures_to_permute;
    void permute(const PPtr& orig_param, const Axes& order);

    std::set<PPtr> closures_to_f16;
    void to_f16(const PPtr& orig_param);

    using O = ov::Output<ov::Node>;
    struct DQParMM {
        PPtr w, s;
        NPtr mm;
    };
    using DQParMMs = std::vector<DQParMM>;
    std::map<std::pair<O, std::size_t>, DQParMMs> par_dq_mms;
    void register_parallel_matmul(const O& multiply, std::size_t axis, DQParMM&& mm);

    std::map<PPtr, std::pair<ov::ParameterVector, std::size_t>> params_to_concat;
    PPtr concat(ov::ParameterVector&& v, std::size_t dim);

    struct DQUnpack {
        PPtr w, z, s;
    };
    std::map<PPtr, DQUnpack> params_to_unpack;
    PPtr unpack(const PPtr& w, const PPtr& z, const PPtr& s, ov::element::Type type);
    PPtr unpack(const PPtr& w, const PPtr& s, ov::element::Type type);

    struct Gather {
        PPtr pnew, pold, pids;
    };
    std::optional<Gather> params_to_gather;
    PPtr host_gather(const PPtr& w, const PPtr& ids);

    struct QuantizedGather {
        // New param -> orig params
        std::map<PPtr, DQUnpack> params_to_runtime_unpack_gather;
        PPtr pids;
    };
    std::optional<QuantizedGather> params_to_quant_gather_unpack;
    PPtr host_gather_unpack_quant(const PPtr& ids, const PPtr& w, const PPtr& z, const PPtr& s, ov::element::Type type);

    using Ref = std::reference_wrapper<Context>;
};

class DQMatMulCWi : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQMatMulCWi");
    explicit DQMatMulCWi(Context::Ref ctx);
};

class DQMatMulCWi_Transpose : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQMatMulCWi_Transpose");
    explicit DQMatMulCWi_Transpose(Context::Ref ctx);
};

class DQMatMulGQi : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQMatMulGQi");
    explicit DQMatMulGQi(Context::Ref ctx);
};

class DQMatMulGQ2i : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQMatMulGQ2i");
    explicit DQMatMulGQ2i(Context::Ref ctx);
};

class DQMatMulGQiP : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQMatMulGQiP");
    explicit DQMatMulGQiP(Context::Ref ctx);
};

class DQMatMulGQ2iP : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQMatMulGQ2iP");
    explicit DQMatMulGQ2iP(Context::Ref ctx);
};

class DQParMMGQ : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQParMMGQ");
    explicit DQParMMGQ(Context::Ref ctx);
};

void mergeParallelMatMuls(const std::shared_ptr<ov::Model>& m, Context& ctx);

// Gather-related passes

class DQLiftGatherAsymCW : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQLiftGatherAsymCW");
    DQLiftGatherAsymCW();
};

class DQLiftGatherSymCW : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQLiftGatherSymCW");
    DQLiftGatherSymCW();
};

class DQLiftGatherSymGQ : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQLiftGatherSymGQ");
    DQLiftGatherSymGQ();
};

class DQLiftGatherCW : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQLiftGatherCW");
    DQLiftGatherCW();
};

// Head vocab unpacks

class DQUnpackDictGatheru : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQUnpackDictGatheru");
    DQUnpackDictGatheru(Context::Ref ctx);
};

class DQUnpackDictGatherGQi : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQUnpackDictGatherGQi");
    DQUnpackDictGatherGQi(Context::Ref ctx);
};

class HostGatherQuantAsymm : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::HostGatherQuantAsymm");
    HostGatherQuantAsymm(Context::Ref ctx);
};

class HostGatherQuantSymm : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::HostGatherQuantSymm");
    HostGatherQuantSymm(Context::Ref ctx);
};

class HostGatherQuant : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::HostGatherQuant");
    HostGatherQuant(Context::Ref ctx);
};

class HostGather : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::HostGather");
    HostGather(Context::Ref ctx);
};

class HostGatherDQ : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::HostGatherDQ");
    HostGatherDQ(Context::Ref ctx);
};

// Tail vocab unpacks

class DQUnpackDictMatMulCWu : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQUnpackDictMatMulCWu");
    DQUnpackDictMatMulCWu(Context::Ref ctx);
};

class DQUnpackDictMatMulCWf8 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQUnpackDictMatMulCWf8");
    DQUnpackDictMatMulCWf8(Context::Ref ctx);
};

class DQUnpackDictMatMulGQi : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::DQUnpackDictMatMulGQi");
    DQUnpackDictMatMulGQi(Context::Ref ctx);
};

class CompressDictMatMulf32 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::CompressDictMatMulf32");
    CompressDictMatMulf32(Context::Ref ctx);
};

// Tail vocab transformations
class PreserveConstDictMatMulCWu : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::PreserveConstDictMatMulCWu");

    using CPtr = std::shared_ptr<ov::op::v0::Constant>;
    using Results = std::reference_wrapper<std::vector<CPtr>>;

    PreserveConstDictMatMulCWu(Results to_keep);
};

class PreserveConstDictMatMulCWf8 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::PreserveConstDictMatMulCWf8");

    using CPtr = std::shared_ptr<ov::op::v0::Constant>;
    using Results = std::reference_wrapper<std::vector<CPtr>>;

    PreserveConstDictMatMulCWf8(Results to_keep);
};

// Slice last Matmul
class SliceLastMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::SliceLastMatmul");
    SliceLastMatmul();
};

class SliceLastMatmulAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::SliceLastMatmulAdd");
    SliceLastMatmulAdd();
};

class SliceLastMatmulTranspose : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::SliceLastMatmulTranspose");
    SliceLastMatmulTranspose();
};

class SliceLastMatmulMultiply : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::SliceLastMatmulMultiply");
    SliceLastMatmulMultiply();
};

// Convolution to MatMul
class ConvToMatmul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("npuw::patterns::opt::ConvToMatmul");
    ConvToMatmul(Context::Ref ctx);
};

// UntangleConst
void untangleConst(std::shared_ptr<ov::Model> model);

}  // namespace opt
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
