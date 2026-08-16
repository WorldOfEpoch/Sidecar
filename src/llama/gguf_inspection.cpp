#include "sidecar/llama/observation.hpp"

#include "sidecar/core/sha256.hpp"

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#if SIDECAR_LLAMA_ENABLED
#include <gguf.h>
#include <llama.h>
#endif

namespace sidecar::llama {
namespace {

std::string EscapeJson(std::string_view text) {
    std::ostringstream output;
    for (const unsigned char value : text) {
        if (value == '\"') output << "\\\"";
        else if (value == '\\') output << "\\\\";
        else if (value == '\n') output << "\\n";
        else if (value == '\r') output << "\\r";
        else if (value == '\t') output << "\\t";
        else if (value < 0x20U) output << "\\u" << std::hex << std::setw(4)
                                      << std::setfill('0') << static_cast<unsigned>(value)
                                      << std::dec;
        else output << static_cast<char>(value);
    }
    return output.str();
}

#if SIDECAR_LLAMA_ENABLED
std::optional<std::string> StringValue(const gguf_context* context, const char* key) {
    const auto id = gguf_find_key(context, key);
    if (id < 0 || gguf_get_kv_type(context, id) != GGUF_TYPE_STRING) return std::nullopt;
    return std::string(gguf_get_val_str(context, id));
}

std::optional<std::uint64_t> UnsignedValue(const gguf_context* context,
                                           const std::string& key) {
    const auto id = gguf_find_key(context, key.c_str());
    if (id < 0) return std::nullopt;
    switch (gguf_get_kv_type(context, id)) {
        case GGUF_TYPE_UINT8: return gguf_get_val_u8(context, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(context, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(context, id);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(context, id);
        case GGUF_TYPE_INT8: {
            const auto value = gguf_get_val_i8(context, id); return value < 0 ? std::nullopt : std::optional<std::uint64_t>(value);
        }
        case GGUF_TYPE_INT16: {
            const auto value = gguf_get_val_i16(context, id); return value < 0 ? std::nullopt : std::optional<std::uint64_t>(value);
        }
        case GGUF_TYPE_INT32: {
            const auto value = gguf_get_val_i32(context, id); return value < 0 ? std::nullopt : std::optional<std::uint64_t>(value);
        }
        case GGUF_TYPE_INT64: {
            const auto value = gguf_get_val_i64(context, id); return value < 0 ? std::nullopt : std::optional<std::uint64_t>(value);
        }
        default: return std::nullopt;
    }
}

std::string MetadataJson(const gguf_context* context) {
    std::ostringstream output;
    output << '{';
    const auto count = gguf_get_n_kv(context);
    for (std::int64_t index = 0; index < count; ++index) {
        if (index != 0) output << ',';
        output << '"' << EscapeJson(gguf_get_key(context, index)) << "\":";
        const auto type = gguf_get_kv_type(context, index);
        switch (type) {
            case GGUF_TYPE_UINT8: output << static_cast<unsigned>(gguf_get_val_u8(context, index)); break;
            case GGUF_TYPE_INT8: output << static_cast<int>(gguf_get_val_i8(context, index)); break;
            case GGUF_TYPE_UINT16: output << gguf_get_val_u16(context, index); break;
            case GGUF_TYPE_INT16: output << gguf_get_val_i16(context, index); break;
            case GGUF_TYPE_UINT32: output << gguf_get_val_u32(context, index); break;
            case GGUF_TYPE_INT32: output << gguf_get_val_i32(context, index); break;
            case GGUF_TYPE_UINT64: output << gguf_get_val_u64(context, index); break;
            case GGUF_TYPE_INT64: output << gguf_get_val_i64(context, index); break;
            case GGUF_TYPE_FLOAT32: output << std::setprecision(9) << gguf_get_val_f32(context, index); break;
            case GGUF_TYPE_FLOAT64: output << std::setprecision(17) << gguf_get_val_f64(context, index); break;
            case GGUF_TYPE_BOOL: output << (gguf_get_val_bool(context, index) ? "true" : "false"); break;
            case GGUF_TYPE_STRING:
                output << '"' << EscapeJson(gguf_get_val_str(context, index)) << '"'; break;
            case GGUF_TYPE_ARRAY:
                output << "{\"array_type\":\"" << gguf_type_name(gguf_get_arr_type(context, index))
                       << "\",\"count\":" << gguf_get_arr_n(context, index) << '}'; break;
            default: output << "null"; break;
        }
    }
    output << '}';
    return output.str();
}
#endif

}  // namespace

ModelIndex InspectGguf(const std::filesystem::path& path,
                       const ModelProvenance& provenance,
                       std::string llama_cpp_commit,
                       bool compute_sha256) {
#if !SIDECAR_LLAMA_ENABLED
    (void)path; (void)provenance; (void)llama_cpp_commit; (void)compute_sha256;
    throw std::runtime_error(
        "GGUF inspection requires SIDECAR_ENABLE_LLAMA=ON and a pinned llama.cpp build");
#else
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("GGUF path is not a regular file: " + path.string());
    }
    const auto file_size = std::filesystem::file_size(path);
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;
    gguf_context* raw = gguf_init_from_file(path.string().c_str(), params);
    if (raw == nullptr) throw std::runtime_error("pinned ggml rejected GGUF metadata");
    struct ContextOwner final {
        gguf_context* value;
        ~ContextOwner() { gguf_free(value); }
    } context{raw};

    ModelIndex model;
    model.path = std::filesystem::absolute(path);
    model.provenance = provenance;
    model.storage = ResolveModelStorage(model.path,
        provenance.source == "OLLAMA_LOCAL_BLOB" ? "OLLAMA_LOCAL_BLOB_INSPECTION" :
                                                    "MODEL_INSPECTION",
        "DIRECT_READ_ONLY_INSPECTION");
    model.llama_cpp_commit = std::move(llama_cpp_commit);
    model.file_size_bytes = file_size;
    model.sha256 = compute_sha256 ? core::Sha256File(path) : "NOT_COMPUTED";
    model.gguf_version = gguf_get_version(raw);
    model.alignment_bytes = static_cast<std::uint32_t>(gguf_get_alignment(raw));
    model.data_offset = gguf_get_data_offset(raw);
    model.architecture = StringValue(raw, "general.architecture").value_or("unknown");
    model.model_name = StringValue(raw, "general.name").value_or(path.filename().string());
    if (const auto file_type = UnsignedValue(raw, "general.file_type")) {
        model.quantization = llama_ftype_name(static_cast<llama_ftype>(*file_type));
    } else {
        model.quantization = "unknown";
    }
    if (const auto blocks = UnsignedValue(raw, model.architecture + ".block_count"))
        model.layer_count = static_cast<std::uint32_t>(*blocks);
    for (const auto* suffix : {".expert_count", ".expert_used_count", ".n_expert"}) {
        if (const auto experts = UnsignedValue(raw, model.architecture + suffix)) {
            model.expert_count = static_cast<std::uint32_t>(*experts);
            break;
        }
    }
    model.metadata_json = MetadataJson(raw);
    if (const auto split_count = UnsignedValue(raw, "split.count");
        split_count && *split_count > 1) {
        throw std::runtime_error(
            "split GGUF models are explicitly unsupported by WU9 range indexing");
    }

    const auto tensor_count = gguf_get_n_tensors(raw);
    if (tensor_count < 0 || tensor_count > static_cast<std::int64_t>((std::numeric_limits<std::uint32_t>::max)()))
        throw std::runtime_error("GGUF tensor count is out of supported range");
    model.tensors.reserve(static_cast<std::size_t>(tensor_count));
    std::uint64_t parameter_count = 0;
    bool valid_ranges = true;
    for (std::int64_t id = 0; id < tensor_count; ++id) {
        TensorIndexEntry tensor;
        tensor.tensor_id = static_cast<std::uint32_t>(id);
        tensor.name = gguf_get_tensor_name(raw, id);
        tensor.type = ggml_type_name(gguf_get_tensor_type(raw, id));
        const auto* dimensions = gguf_get_tensor_ne(raw, id);
        std::size_t dimension_count = GGML_MAX_DIMS;
        while (dimension_count > 1U && dimensions[dimension_count - 1U] == 1) --dimension_count;
        std::uint64_t elements = 1;
        for (std::size_t dimension = 0; dimension < dimension_count; ++dimension) {
            tensor.dimensions.push_back(dimensions[dimension]);
            if (dimensions[dimension] > 0 &&
                elements <= (std::numeric_limits<std::uint64_t>::max)() /
                            static_cast<std::uint64_t>(dimensions[dimension])) {
                elements *= static_cast<std::uint64_t>(dimensions[dimension]);
            }
        }
        if (parameter_count <= (std::numeric_limits<std::uint64_t>::max)() - elements)
            parameter_count += elements;
        tensor.bytes = gguf_get_tensor_size(raw, id);
        const auto relative_offset = gguf_get_tensor_offset(raw, id);
        tensor.file_offset = model.data_offset + relative_offset;
        const std::uint64_t alignment = (std::max)(std::uint64_t{1},
                                                   static_cast<std::uint64_t>(model.alignment_bytes));
        tensor.file_span_bytes = ((tensor.bytes + alignment - 1U) / alignment) * alignment;
        tensor.alignment_bytes = model.alignment_bytes;
        tensor.layer = ParseLayerIndex(tensor.name);
        tensor.expert = ParseExpertIndex(tensor.name);
        tensor.role = ClassifyTensorRole(tensor.name);
        model.is_moe = model.is_moe || IsMoeTensor(tensor.name);
        if (tensor.file_offset < model.data_offset ||
            relative_offset % alignment != 0 ||
            tensor.file_offset > file_size || tensor.bytes > file_size - tensor.file_offset)
            valid_ranges = false;
        model.tensors.push_back(std::move(tensor));
    }
    std::vector<const TensorIndexEntry*> by_offset;
    by_offset.reserve(model.tensors.size());
    for (const auto& tensor : model.tensors) by_offset.push_back(&tensor);
    std::sort(by_offset.begin(), by_offset.end(), [](const auto* left, const auto* right) {
        return left->file_offset < right->file_offset;
    });
    for (std::size_t index = 1; index < by_offset.size(); ++index) {
        const auto* previous = by_offset[index - 1U];
        const auto* current = by_offset[index];
        if (previous->file_offset > (std::numeric_limits<std::uint64_t>::max)() - previous->bytes ||
            previous->file_offset + previous->bytes > current->file_offset) {
            valid_ranges = false;
            break;
        }
    }
    model.parameter_count = parameter_count;
    model.verification_status = valid_ranges ? "GGUF_METADATA_AND_RANGES_VALID" : "FAILED";
    model.verification_detail = valid_ranges
        ? "pinned ggml metadata parser accepted aligned, non-overlapping tensor byte ranges within one GGUF file"
        : "one or more tensor ranges are unaligned, overlapping, or outside the file";
    if (!valid_ranges) throw std::runtime_error(model.verification_detail);
    return model;
#endif
}

}  // namespace sidecar::llama
