// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "debug_helper.hpp"
#include <regex>
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/util/file_util.hpp"

#ifdef GPU_DEBUG_CONFIG

#include "to_string_utils.h"
#include "loop_inst.h"
#include "condition_inst.h"
#include "program_dump_graph.h"

#include <iomanip>
#include <fstream>
#include <sys/stat.h>

namespace cldnn {

namespace {

float convert_element(int64_t i) { return static_cast<float>(i); }
float convert_element(int32_t i) { return static_cast<float>(i); }

float convert_element(float f) { return f; }

float convert_element(ov::float16 h) { return static_cast<float>(h); }

size_t get_x_pitch(const layout& layout) {
    try {
        auto tensor_x0 = tensor(batch(0), feature(0), spatial(0, 0, 0, 0));
        auto tensor_x1 = tensor(batch(0), feature(0), spatial(1, 0, 0, 0));
        auto x0 = layout.get_linear_offset(tensor_x0);
        auto x1 = layout.get_linear_offset(tensor_x1);
        return (x1 - x0);
    } catch (...) {
        // When spatial size of x=0, x_pitch is meaningless
        return 0;
    }
}

template <class T>
void dump(memory::ptr mem, stream& stream, std::ofstream& file_stream, bool dump_raw) {
    auto&& size = mem->get_layout().get_tensor();

    auto batch_size = std::max<ov::Dimension::value_type>(std::min<ov::Dimension::value_type>(ExecutionConfig::get_dump_batch_limit(), size.batch[0]), 1);
    tensor tmp_size(size);
    tmp_size.batch[0] = batch_size;
    if (tmp_size == size) {
        file_stream << "shape: " << size.to_string() << " ";
        file_stream << "(count: " << size.count()
                    << ", addr: " << mem->buffer_ptr()
                    << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format) << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    } else {
        file_stream << "shape: " << tmp_size.to_string() << " ";
        file_stream << "(count: " << tmp_size.count()
                    << ", addr: " << mem->buffer_ptr()
                    << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format)
                    << ", original shape: " << size.to_string() << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    }

    if (size.count() == 0) {
        file_stream << "Empty buffer" << std::endl;
        return;
    }

    mem_lock<T, mem_lock_type::read> lock(mem, stream);
    auto mem_ptr = lock.data();
    auto x_pitch = get_x_pitch(mem->get_layout());
    std::stringstream buffer;

    if (!dump_raw) {
        for (ov::Dimension::value_type g = 0; g < size.group[0]; ++g) {
            for (ov::Dimension::value_type b = 0; b < batch_size; ++b) {
                for (ov::Dimension::value_type f = 0; f < size.feature[0]; ++f) {
                    for (ov::Dimension::value_type w = 0; w < size.spatial[3]; ++w) {
                        for (ov::Dimension::value_type z = 0; z < size.spatial[2]; ++z) {
                            for (ov::Dimension::value_type y = 0; y < size.spatial[1]; ++y) {
                                cldnn::tensor t(cldnn::group(g), cldnn::batch(b), cldnn::feature(f), cldnn::spatial(0, y, z, w));
                                size_t input_it = mem->get_layout().get_linear_offset(t);

                                for (ov::Dimension::value_type x = 0; x < size.spatial[0]; ++x, input_it += x_pitch) {
                                    buffer << std::fixed << std::setprecision(6) << convert_element(mem_ptr[input_it]) << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        for (size_t i = 0; i < lock.size(); ++i) {
            buffer << std::fixed << std::setprecision(6) << convert_element(mem_ptr[i]) << std::endl;
        }
    }
    file_stream << buffer.str();
}

void unpack(cldnn::data_types type, uint8_t input, int8_t &v0, int8_t &v1) {
    if (type == cldnn::data_types::i4) {
        char s_bit = (input & 0x08);
        char mask = s_bit > 0 ? 0xF0 : 0x00;
        v0 = (input & 0x0F) | mask;

        input >>= 4;
        s_bit = (input & 0x08);
        mask = s_bit > 0 ? 0xF0 : 0x00;
        v1 = (input & 0x0F) | mask;
    } else if (type == cldnn::data_types::u4) {
        v0 = input & 0x0F;
        v1 = input >> 4;
    } else {
        OPENVINO_ASSERT(false, "not supported unpacking");
    }
}

void dump_i4u4(cldnn::data_types type, memory::ptr mem, stream& stream, std::ofstream& file_stream, bool dump_raw) {
    auto&& size = mem->get_layout().get_tensor();

    auto batch_size = std::max<ov::Dimension::value_type>(std::min<ov::Dimension::value_type>(ExecutionConfig::get_dump_batch_limit(), size.batch[0]), 1);
    tensor tmp_size(size);
    tmp_size.batch[0] = batch_size;
    if (tmp_size == size) {
        file_stream << "shape: " << size.to_string() << " ";
        file_stream << "(count: " << size.count()
                    << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format) << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    } else {
        file_stream << "shape: " << tmp_size.to_string() << " ";
        file_stream << "(count: " << tmp_size.count()
                    << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format)
                    << ", original shape: " << size.to_string() << ")"
                    << (dump_raw ? " raw data" : "") << std::endl;
    }

    if (size.count() == 0) {
        file_stream << "Empty buffer" << std::endl;
        return;
    }

    mem_lock<uint8_t, mem_lock_type::read> lock(mem, stream);
    auto mem_ptr = lock.data();
    std::stringstream buffer;

    if (dump_raw) {
        for (size_t i = 0; i < lock.size(); ++i) {
            int8_t v0, v1;
            unpack(type, mem_ptr[i], v0, v1);
            buffer << std::fixed << std::setprecision(6) << static_cast<int>(v0) << std::endl;
            buffer << std::fixed << std::setprecision(6) << static_cast<int>(v1) << std::endl;
        }
    } else {
        std::cout << __func__ << " supports raw dump only" << std::endl;
    }
    file_stream << buffer.str();
}

std::string get_name_for_dump(const std::string& file_name) {
    std::string filename = file_name;
    std::replace(filename.begin(), filename.end(), '\\', '_');
    std::replace(filename.begin(), filename.end(), '/', '_');
    std::replace(filename.begin(), filename.end(), ' ', '_');
    std::replace(filename.begin(), filename.end(), ':', '_');
    return filename;
}

void log_memory_to_file(memory::ptr mem, layout data_layout, stream& stream, std::string filename, bool dump_raw) {
    std::ofstream file_stream(filename);
    if (!mem) {
        file_stream << "Empty" << std::endl;
        return;
    }

    // Reinterpret buffer to represent actual data layout
    auto actual_mem = mem->get_engine()->reinterpret_buffer(*mem, data_layout);

    auto mem_dt = actual_mem->get_layout().data_type;
    if (mem_dt == cldnn::data_types::f32)
        dump<float>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::f16)
        dump<ov::float16>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i64)
        dump<int64_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i32)
        dump<int32_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i8)
        dump<int8_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::u8)
        dump<uint8_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::u8)
        dump<uint8_t>(actual_mem, stream, file_stream, dump_raw);
    else if (mem_dt == cldnn::data_types::i4 || mem_dt == cldnn::data_types::u4)
        dump_i4u4(mem_dt, actual_mem, stream, file_stream, dump_raw);
    else
        std::cout << "Dump for this data type is not supported: " << dt_to_str(mem_dt) << std::endl;
}

std::string get_file_path_for_binary_dump(cldnn::layout layout, const std::string& name, const std::string& dump_layers_path) {
    std::string filename;
    std::string data_type = ov::element::Type(layout.data_type).get_type_name();
    std::string format = layout.format.to_string();
    std::string tensor;
    auto dims = layout.get_dims();
    for (size_t r = 0 ; r < layout.get_rank() ; r++) {
        tensor += ("_" + to_string(dims[r]));
    }

    std::string layer_name = get_name_for_dump(name);
    filename = dump_layers_path + layer_name + "__" + data_type + "_" + tensor + "__" + format + ".bin";
    return filename;
}

bool is_target_iteration(int64_t iteration, const std::set<int64_t> dump_iteration) {
    if (iteration < 0)
        return true;

    if (dump_iteration.empty())
        return true;

    if (dump_iteration.find(iteration) == std::end(dump_iteration))
        return false;

    return true;
}

std::string get_matched_from_filelist(const std::vector<std::string>& file_names, std::string pattern) {
    for (const auto& file : file_names) {
        auto found = file.find(pattern);
        if (found != std::string::npos) {
            return file;
        }
    }

    return std::string();
}

bool is_layer_name_matched(const std::string& layer_name, const std::string& pattern) {
    auto upper_layer_name = std::string(layer_name.length(), '\0');
    std::transform(layer_name.begin(), layer_name.end(), upper_layer_name.begin(), ::toupper);
    auto upper_pattern = std::string(pattern.length(), '\0');
    std::transform(pattern.begin(), pattern.end(), upper_pattern.begin(), ::toupper);

    // Check pattern from exec_graph
    size_t pos = upper_layer_name.find(':');
    auto upper_exec_graph_name = upper_layer_name.substr(pos + 1, upper_layer_name.size());
    if (upper_exec_graph_name.compare(upper_pattern) == 0) {
        return true;
    }

    // Check pattern with regular expression
    std::regex re(upper_pattern);
    return std::regex_match(upper_layer_name, re);
}

bool is_layer_for_dumping(const ExecutionConfig& config, const std::string& layer_name) {
    const auto& dump_layers = config.get_dump_layer_names();
    if (dump_layers.empty())
        return true;

    auto iter = std::find_if(dump_layers.begin(), dump_layers.end(), [&](const std::string& dl){
        return is_layer_name_matched(layer_name, dl);
    });
    return (iter != dump_layers.end());
}

std::vector<std::string> get_filenames_for_matched_layer_loading_binaries(const ExecutionConfig& config, const std::string& id) {
    std::vector<std::string> file_names;
    if (config.get_load_dump_raw_binary().empty())
        return file_names;

    for (const auto& load_layer : config.get_load_dump_raw_binary()) {
        size_t file = load_layer.rfind(":");
        if (file != std::string::npos) {
            if (id == load_layer.substr(0, file)) {
                auto file_name_str = load_layer.substr(file + 1);
                size_t head = 0;
                size_t found = 0;
                do {
                    found = file_name_str.find(",", head);
                    if (found != std::string::npos)
                        file_names.push_back(file_name_str.substr(head, (found - head)));
                    else
                        file_names.push_back(file_name_str.substr(head));

                    head = found+1;
                    GPU_DEBUG_LOG << " Layer name loading raw dump : " << load_layer.substr(0, file) << " / the dump file : "
                                << file_names.back() << std::endl;
                } while (found != std::string::npos);

                return file_names;
            }
        }
    }

    return file_names;
}

}  // namespace

NodeDebugHelper::NodeDebugHelper(const primitive_inst& inst)
    : m_inst(inst)
    , m_stream(inst.get_network().get_stream())
    , m_network(inst.get_network())
    , m_program(inst.get_network().get_program().get())
    , m_iter(m_network.iteration) {
    const auto& config = m_network.get_config();
    // Load binary dump for input layers
    if (!config.get_load_dump_raw_binary().empty()) {
        const std::string layer_name = m_inst.id();
        auto files = get_filenames_for_matched_layer_loading_binaries(config, layer_name);
        if (!files.empty()) {
            m_stream.finish(); // Wait for stream completion before buffer assignment
            if (m_inst.is_input()) {
                // Loading binary dumps for output tensors of input-layers : only one output exists or index(dstN) exists
                auto dump_file = get_matched_from_filelist(files, "_dst0__");
                OPENVINO_ASSERT((files.size() == 1 || dump_file.length() != 0), "Unexpected binary dump for input layer");

                OPENVINO_ASSERT(files.size() == m_inst.outputs_memory_count(), "Mismatch dump file count");

                for (size_t i = 0; i < m_inst.outputs_memory_count(); i++) {
                    auto dump_file = files[0];
                    if (files.size() > 1 || m_inst.outputs_memory_count() != 1) {
                        std::string pattern = "_dst" + std::to_string(i) + "__";
                        dump_file = get_matched_from_filelist(files, pattern);
                    }
                    OPENVINO_ASSERT((dump_file.length() > 0), "Could not find expected pattern '_dst[N]__' for binary dump");
                    GPU_DEBUG_COUT << " Load binary dump : " << dump_file << " for " << layer_name << std::endl;

                    std::vector<uint8_t> bin = ov::util::load_binary(dump_file);
                    OPENVINO_ASSERT(!bin.empty(), "Failure loading binary from OV_LOAD_DUMP_RAW_BINARY : " + dump_file);

                    auto output_mem = m_inst.output_memory_ptr(i);
                    OPENVINO_ASSERT(output_mem->size() == bin.size(), "memory size mis-match for OV_LOAD_DUMP_RAW_BINARY : " + layer_name
                                    + "\n Expected size : " + to_string(output_mem->size()) + ", Binary : " + to_string(bin.size()));

                    output_mem->copy_from(m_stream, static_cast<void *>(&bin[0]), true);
                }
            } else {
                auto check_dst = get_matched_from_filelist(files, "_dst0__");
                OPENVINO_ASSERT(check_dst.length() == 0, "Expected to load binaries for inputs of " + layer_name);

                // Loading input tensors for any layer
                auto dump_file = get_matched_from_filelist(files, "_src0__");
                OPENVINO_ASSERT(dump_file.length() != 0, "Could not find expected pattern '_src[N]__' for binary dump input : " + layer_name);

                for (size_t i = 0; i < m_inst.dependencies().size(); i++) {
                    auto dump_file = files[0];
                    if (files.size() > 1 || m_inst.dependencies().size() != 1) {
                        std::string pattern = "_src" + std::to_string(i) + "__";
                        dump_file = get_matched_from_filelist(files, pattern);
                    }
                    if (dump_file.length() == 0) {
                        GPU_DEBUG_COUT  << " Skip loading for  input(" << i << ") of " << layer_name << std::endl;
                        continue;
                    }
                    OPENVINO_ASSERT((dump_file.length() > 0), "Could not find expected pattern '_src[N]__' for binary dump input");
                    GPU_DEBUG_COUT  << " Load binary dump : " << dump_file << " for input(" << i << ") of " << layer_name << std::endl;

                    std::vector<uint8_t> bin = ov::util::load_binary(dump_file);
                    OPENVINO_ASSERT(!bin.empty(), "Failure loading binary from OV_LOAD_DUMP_RAW_BINARY : " + dump_file);

                    auto input_mem = m_inst.dep_memory_ptr(i);
                    if (input_mem->size() != bin.size()) {
                        std::cout << "WARNING: memory size mis-match for OV_LOAD_DUMP_RAW_BINARY : " + layer_name
                                    << "  " << input_mem->size() << " / " << bin.size() << std::endl;
                        bin.resize(input_mem->size());
                    }

                    input_mem->copy_from(m_stream, static_cast<void *>(&bin[0]), true);
                }
            }
        }
    }

    // Dump input buffers of 'inst'
    if (config.get_dump_tensors_path().length() > 0) {
        const std::string& layer_name = inst.id();

        if (is_target_iteration(m_iter, config.get_dump_iterations()) &&
            config.get_dump_tensors() != ov::intel_gpu::DumpTensors::out && is_layer_for_dumping(config, layer_name)) {
            m_stream.finish(); // Wait for stream completion before dumping input buffers
            std::string debug_str_for_bin_load = " Command for loading : OV_LOAD_DUMP_RAW_BINARY=\"" + layer_name + ":";
            for (size_t i = 0; i < m_inst.dependencies().size(); i++) {
                std::string name = get_file_prefix() + "_src" + std::to_string(i);
                auto input_mem = m_inst.dep_memory_ptr(i);
                if (input_mem == nullptr) {
                    GPU_DEBUG_COUT  << " input_mem_" << i << " is nullptr. Nothing to dump." << std::endl;
                    continue;
                }

                auto dep = m_inst.dependencies().at(i);
                auto input_layout = dep.first->get_output_layout(dep.second);
                if (config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::binary) {
                    // Binary dump : raw
                    auto filename = get_file_path_for_binary_dump(input_layout, name, config.get_dump_tensors_path());

                    mem_lock<char, mem_lock_type::read> lock(input_mem, m_stream);
                    ov::util::save_binary(filename, lock.data(), input_mem->size());
                    GPU_DEBUG_COUT << " Dump layer src : " << layer_name << " to " << filename << std::endl;
                    debug_str_for_bin_load += (filename + ",");
                } else {
                    const bool dump_raw = config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::text_raw;
                    GPU_DEBUG_COUT << " Dump " << (dump_raw ? "raw " : "") << name << std::endl;
                    auto filename = config.get_dump_tensors_path() + get_name_for_dump(name) + ".txt";
                    log_memory_to_file(input_mem,
                                       input_layout,
                                       m_stream,
                                       filename,
                                       dump_raw);
                }
            }

            if (config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::binary && !inst.is_input()) {
                debug_str_for_bin_load[debug_str_for_bin_load.size()-1] = '\"';
                GPU_DEBUG_COUT << debug_str_for_bin_load << std::endl;
            }
        }
    }
}


NodeDebugHelper::~NodeDebugHelper() {
    const auto& config = m_network.get_config();
    // Dump output buffers of 'inst'
    if (config.get_dump_tensors_path().length() > 0) {
        const std::string layer_name = m_inst.id();

        if (is_target_iteration(m_iter, config.get_dump_iterations()) &&
            config.get_dump_tensors() != ov::intel_gpu::DumpTensors::in &&
            is_layer_for_dumping(config, layer_name)) {
            m_stream.finish(); // Wait for stream completion before dumping output buffers
            std::string debug_str_for_bin_load = " Command for loading : OV_LOAD_DUMP_RAW_BINARY=\""
                                                    + layer_name + ":";
            for (size_t i = 0; i < m_inst.outputs_memory_count(); i++) {
                std::string name = get_file_prefix() + "_dst" + std::to_string(i);
                auto output_mem = m_inst.output_memory_ptr(i);
                if (output_mem == nullptr) {
                    GPU_DEBUG_COUT  << " output_mem is nullptr. Nothing to dump." << std::endl;
                    continue;
                }

                if (config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::binary) {
                    // Binary dump : raw
                    auto output_layout = m_inst.get_output_layout(i);
                    auto filename = get_file_path_for_binary_dump(output_layout, name, config.get_dump_tensors_path());

                    mem_lock<char, mem_lock_type::read> lock(output_mem, m_stream);
                    ov::util::save_binary(filename, lock.data(), output_mem->size());
                    GPU_DEBUG_COUT  << " Dump layer dst : " << layer_name << " to " << filename << std::endl;
                    debug_str_for_bin_load += (filename + ",");
                } else {
                    const bool dump_raw = config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::text_raw;
                    GPU_DEBUG_COUT << " Dump " << (dump_raw ? "raw " : "") << name << std::endl;
                    auto filename = config.get_dump_tensors_path() + get_name_for_dump(name) + ".txt";
                    // Text dump
                    log_memory_to_file(output_mem,
                                       m_inst.get_output_layout(i),
                                       m_stream,
                                       filename,
                                       dump_raw);
                }
            }

            if (config.get_dump_tensors_format() == ov::intel_gpu::DumpFormat::binary && m_inst.is_input()) {
                debug_str_for_bin_load[debug_str_for_bin_load.size()-1] = '\"';
                GPU_DEBUG_COUT << debug_str_for_bin_load << std::endl;;
            }
        }
    }
}

NetworkDebugHelper::NetworkDebugHelper(const network& net)
    : m_network(net)
    , m_iter(net.iteration) {
    auto net_id = m_network.get_id();
    const auto& config = m_network.get_config();
    if (config.get_dump_memory_pool()) {
        auto& iters = config.get_dump_iterations();
        if (iters.empty() || iters.find(m_iter) != iters.end()) {
            GPU_DEBUG_COUT << "============================================================================" << std::endl;
            GPU_DEBUG_COUT << "Start network execution (net_id : " << net_id << ", iter :" << m_iter << ")" << std::endl;
            if (m_iter == 0 && net_id > 0) {
                dump_memory_pool(config.get_dump_memory_pool_path(), m_iter);
                GPU_DEBUG_COUT << "============================================================================" << std::endl;
            }
        }
    } else {
        GPU_DEBUG_TRACE << "============================================================================" << std::endl;
        GPU_DEBUG_TRACE << "Start network execution (net_id : " << net_id << ", iter :" << m_iter << ")" << std::endl;
    }
}

NetworkDebugHelper::~NetworkDebugHelper() {
    auto prog = m_network.get_program().get();
    auto net_id = m_network.get_id();
    const auto& config = prog->get_config();
    // print '-data_shape' option for benchmark_app
    if (config.get_verbose() >= 4) {
        std::stringstream data_shape_str;
        auto add_string = [&data_shape_str](std::string str) {
            data_shape_str << ((data_shape_str.rdbuf()->in_avail() == 0) ? " -data_shape " : ",") << str;
        };

        for (auto& inst : m_network._exec_order) {
            auto name = inst->id();
            auto pos = name.find(':');
            auto type = name.substr(0, pos);
            name.erase(0, pos + 1);
            if (inst->is_input() && type == "parameter") {
                add_string(name + inst->get_output_layout().get_partial_shape().to_string());
            }
        }

        GPU_DEBUG_COUT << "[program:" << std::setw(2) << ((prog != nullptr) ? prog->get_id() : 0)
                       << "|network:" << std::setw(2) << net_id << "|iter:" << std::setw(4) << m_iter <<  "] benchmark_app cmd: "
                       << data_shape_str.str() << std::endl;
    }

    if (!config.get_dump_graphs_path().empty() && is_target_iteration(m_iter, config.get_dump_iterations())) {
        auto get_fixed_str = [](int value, int length = 2) -> std::string {
            std::ostringstream ss;
            ss << std::setw(length) << std::setfill('0') << std::to_string(value);
            return ss.str();
        };
        std::string path = get_dir_path(m_network.get_config());
        if (!path.empty()) {
            std::ofstream ofs(path + "cldnn_program_exec_p" + get_fixed_str(prog->get_id()) + "_n" + get_fixed_str(net_id)
                              + "_" + get_fixed_str(m_iter, 5) + ".graph");
            dump_graph_init(ofs, *prog, [this](const primitive_id& id) -> std::shared_ptr<const primitive_inst> {
                return m_network.get_primitive(id);
            });
        }
    }

    if (config.get_dump_memory_pool()) {
        auto& iters = config.get_dump_iterations();
        if (iters.empty() || iters.find(m_iter) != iters.end()) {
            dump_memory_pool(config.get_dump_memory_pool_path(), m_iter);
            GPU_DEBUG_COUT << "============================================================================" << std::endl;
        }
    }

    m_network.iteration++;
}

void NetworkDebugHelper::dump_memory_pool(std::string dump_path, int64_t curr_iter) const {
    m_network.get_memory_pool().dump(m_network.get_id(), curr_iter, dump_path);
    auto get_constants_mem_size = [&](allocation_type type) -> size_t {
        size_t mem_size = 0;
        for (auto& prim : m_network._primitives) {
            if (prim.second->get_node().is_constant()) {
                for (size_t i = 0; i < prim.second->outputs_memory_count(); i++) {
                    if (prim.second->output_memory_ptr(i)->get_allocation_type() == type)
                        mem_size += prim.second->output_memory_ptr(i)->size();
                }
            }
        }
        return mem_size;
    };
    auto get_variables_mem_size = [&](allocation_type type) -> size_t {
        size_t mem_size = 0;
        for (auto& var : m_network.get_variables()) {
            if (var.second->get_memory() && var.second->get_memory()->get_allocation_type() == type)
                mem_size += var.second->get_actual_mem_size();
        }
        return mem_size;
    };
    auto get_mb_size = [&](int64_t size) -> std::string {
        if (size == 0) return "0 MB";
        return std::to_string(static_cast<float>(size) / (1024 * 1024)) + " MB";
    };
    int64_t usm_host_const_mem_size     = get_constants_mem_size(allocation_type::usm_host);
    int64_t usm_device_const_mem_size   = get_constants_mem_size(allocation_type::usm_device);
    int64_t usm_host_var_mem_size       = get_variables_mem_size(allocation_type::usm_host);
    int64_t usm_device_var_mem_size     = get_variables_mem_size(allocation_type::usm_device);
    int64_t host_mem_size               = m_network.get_engine().get_used_device_memory(allocation_type::usm_host);
    int64_t device_mem_size             = m_network.get_engine().get_used_device_memory(allocation_type::usm_device);
    int64_t usm_host_mem_pool_size      = m_network.get_memory_pool().get_total_mem_pool_size(allocation_type::usm_host);
    int64_t usm_host_etc_size           = host_mem_size - usm_host_mem_pool_size
                                            - usm_host_const_mem_size - usm_host_var_mem_size;
    int64_t usm_device_mem_pool_size    = m_network.get_memory_pool().get_total_mem_pool_size(allocation_type::usm_device);
    int64_t usm_device_etc_size         = device_mem_size - usm_device_mem_pool_size
                                            - usm_device_const_mem_size - usm_device_var_mem_size;
    GPU_DEBUG_COUT << "------------------------------------------------------------------------" << std::endl;
    GPU_DEBUG_COUT << "Memory statistics for (net_id:" << m_network.get_id() << ", iter:" << curr_iter << ")" << std::endl;
    GPU_DEBUG_COUT << " Total host mem size     : " << get_mb_size(host_mem_size)               << std::endl;
    GPU_DEBUG_COUT << " * Memory pool           : " << get_mb_size(usm_host_mem_pool_size)      << std::endl;
    GPU_DEBUG_COUT << " * Constant              : " << get_mb_size(usm_host_const_mem_size)     << std::endl;
    GPU_DEBUG_COUT << " * Variable              : " << get_mb_size(usm_host_var_mem_size)       << std::endl;
    GPU_DEBUG_COUT << " * ETC                   : " << get_mb_size(usm_host_etc_size)           << std::endl;
    GPU_DEBUG_COUT << " Total device mem size   : " << get_mb_size(device_mem_size)             << std::endl;
    GPU_DEBUG_COUT << " * Memory pool           : " << get_mb_size(usm_device_mem_pool_size)    << std::endl;
    GPU_DEBUG_COUT << " * Constant              : " << get_mb_size(usm_device_const_mem_size)   << std::endl;
    GPU_DEBUG_COUT << " * Variable              : " << get_mb_size(usm_device_var_mem_size)     << std::endl;
    GPU_DEBUG_COUT << " * ETC                   : " << get_mb_size(usm_device_etc_size)         << std::endl;
    GPU_DEBUG_COUT << "------------------------------------------------------------------------" << std::endl;
}

}  // namespace cldnn

#endif // GPU_DEBUG_CONFIG
