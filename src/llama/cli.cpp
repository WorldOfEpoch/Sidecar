#include "sidecar/llama/cli.hpp"

#include "sidecar/core/sha256.hpp"
#include "sidecar/database/database.hpp"
#include "sidecar/hardware/hardware.hpp"
#include "sidecar/hardware/format.hpp"
#include "sidecar/hardware/persistence.hpp"
#include "sidecar/llama/observation.hpp"
#include "sidecar/version.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#if SIDECAR_LLAMA_ENABLED
#include <ggml.h>
#include <llama.h>
#endif

namespace sidecar::llama {
namespace {

struct Options {
    std::filesystem::path model;
    std::optional<std::filesystem::path> database;
    std::optional<std::filesystem::path> trace;
    ModelProvenance provenance;
    std::string storage_role;
    std::string copy_relationship;
    bool json{false};
    bool tensors{false};
    bool hash{true};
    bool authoritative{false};
    std::optional<std::string> supplied_sha256;
    std::uint32_t prompt{128};
    std::uint32_t generate{32};
    std::uint32_t context{4096};
    std::uint32_t warmups{1};
    std::uint32_t repetitions{1};
    std::int32_t gpu_layers{-1};
    ObserverMode observer{ObserverMode::None};
    std::uint64_t causal_lead_ns{0};
    std::uint64_t oracle_lead_ns{0};
    std::uint64_t pipeline_ns{0};
    std::uint64_t block_size_bytes{kBlock64MiB};
};

std::uint64_t ParseUnsigned(std::string_view text, std::string_view option) {
    std::size_t used = 0;
    const auto value = std::stoull(std::string(text), &used, 10);
    if (used != text.size()) throw std::invalid_argument(std::string(option) + " requires an integer");
    return value;
}

Options ParseOptions(int argc, char** argv, int begin) {
    Options options;
    for (int index = begin; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto next = [&]() -> std::string {
            if (++index >= argc) throw std::invalid_argument(std::string(argument) + " requires a value");
            return argv[index];
        };
        if (argument == "--model") options.model = next();
        else if (argument == "--database") options.database = std::filesystem::path(next());
        else if (argument == "--trace") options.trace = std::filesystem::path(next());
        else if (argument == "--json") options.json = true;
        else if (argument == "--tensors") options.tensors = true;
        else if (argument == "--no-hash") options.hash = false;
        else if (argument == "--authoritative") options.authoritative = true;
        else if (argument == "--sha256") { options.supplied_sha256 = next(); options.hash = false; }
        else if (argument == "--source") options.provenance.source = next();
        else if (argument == "--source-repository") options.provenance.source_repository = next();
        else if (argument == "--source-revision") options.provenance.source_revision = next();
        else if (argument == "--ollama-tag") options.provenance.ollama_model_tag = next();
        else if (argument == "--ollama-digest") options.provenance.ollama_digest = next();
        else if (argument == "--original-blob") options.provenance.original_blob_path = std::filesystem::path(next());
        else if (argument == "--alias") options.provenance.alias_path = std::filesystem::path(next());
        else if (argument == "--storage-role") options.storage_role = next();
        else if (argument == "--copy-relationship") options.copy_relationship = next();
        else if (argument == "--prompt-tokens") options.prompt = static_cast<std::uint32_t>(ParseUnsigned(next(), argument));
        else if (argument == "--generate-tokens") options.generate = static_cast<std::uint32_t>(ParseUnsigned(next(), argument));
        else if (argument == "--context") options.context = static_cast<std::uint32_t>(ParseUnsigned(next(), argument));
        else if (argument == "--warmups") options.warmups = static_cast<std::uint32_t>(ParseUnsigned(next(), argument));
        else if (argument == "--repetitions") options.repetitions = static_cast<std::uint32_t>(ParseUnsigned(next(), argument));
        else if (argument == "--gpu-layers") options.gpu_layers = std::stoi(next());
        else if (argument == "--observer") {
            const auto value = ParseObserverMode(next());
            if (!value) throw std::invalid_argument("unknown observer mode");
            options.observer = *value;
        } else if (argument == "--causal-lead-ns") options.causal_lead_ns = ParseUnsigned(next(), argument);
        else if (argument == "--oracle-lead-ns") options.oracle_lead_ns = ParseUnsigned(next(), argument);
        else if (argument == "--pipeline-ns") options.pipeline_ns = ParseUnsigned(next(), argument);
        else if (argument == "--block-bytes") options.block_size_bytes = ParseUnsigned(next(), argument);
        else if (options.model.empty() && !argument.starts_with("--")) options.model = argument;
        else throw std::invalid_argument("invalid llama option: " + std::string(argument));
    }
    return options;
}

std::string DimensionsJson(const std::vector<std::int64_t>& dimensions) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        if (index != 0) output << ',';
        output << dimensions[index];
    }
    output << ']';
    return output.str();
}

std::string JsonEscape(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    for (const char value : text) {
        if (value == '\\' || value == '"') result.push_back('\\');
        if (value == '\n') result += "\\n";
        else if (value == '\r') result += "\\r";
        else if (value == '\t') result += "\\t";
        else result.push_back(value);
    }
    return result;
}

template <typename T>
std::string ArrayJson(const std::vector<T>& values) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) output << ',';
        output << values[index];
    }
    output << ']';
    return output.str();
}

std::string StatisticsJson(const std::vector<std::uint64_t>& input) {
    if (input.empty()) return "{\"count\":0}";
    auto values = input;
    std::sort(values.begin(), values.end());
    long double sum = 0;
    for (const auto value : values) sum += value;
    const long double mean = sum / static_cast<long double>(values.size());
    long double squared_error = 0;
    for (const auto value : values) {
        const long double delta = static_cast<long double>(value) - mean;
        squared_error += delta * delta;
    }
    const long double variance = values.size() < 2 ? 0.0L :
        squared_error / static_cast<long double>(values.size() - 1U);
    auto quantile = [&](double q) {
        const double position = q * static_cast<double>(values.size() - 1U);
        const auto low = static_cast<std::size_t>(std::floor(position));
        const auto high = static_cast<std::size_t>(std::ceil(position));
        return static_cast<double>(values[low]) +
               (static_cast<double>(values[high]) - static_cast<double>(values[low])) *
               (position - static_cast<double>(low));
    };
    std::ostringstream output;
    output << "{\"count\":" << values.size() << ",\"mean_ns\":"
           << static_cast<double>(mean) << ",\"median_ns\":" << quantile(0.5)
           << ",\"stddev_ns\":" << std::sqrt(static_cast<double>(variance))
           << ",\"min_ns\":" << values.front()
           << ",\"p50_ns\":" << quantile(0.5) << ",\"p90_ns\":" << quantile(0.9)
           << ",\"p95_ns\":" << quantile(0.95) << ",\"p99_ns\":" << quantile(0.99)
           << ",\"p99_9_ns\":";
    if (values.size() >= 1000) output << quantile(0.999);
    else output << "null";
    output << ",\"max_ns\":" << values.back() << '}';
    return output.str();
}

std::string LayerSequenceJson(const std::vector<ObserverEvent32>& events,
                              std::uint32_t evaluation_index) {
    std::ostringstream output;
    output << '[';
    bool first = true;
    std::int16_t previous = -2;
    for (const auto& event : events) {
        if (event.eval_index != evaluation_index || event.layer < 0 ||
            event.layer == previous) continue;
        if (!first) output << ',';
        first = false;
        output << event.layer;
        previous = event.layer;
    }
    output << ']';
    return output.str();
}

std::string ObserverDictionaryJson(const ModelIndex& model,
                                   const std::vector<ObserverEvent32>& events) {
    std::set<std::uint16_t> operations;
    for (const auto& event : events) operations.insert(event.op_id);
    std::ostringstream output;
    output << "{\"tensor_dictionary\":{\"table\":\"gguf_tensors\",\"model_id\":\""
           << JsonEscape(model.sha256) << "\",\"count\":" << model.tensors.size()
           << "},\"operation_dictionary\":[";
    bool first = true;
    for (const auto operation : operations) {
        if (!first) output << ',';
        first = false;
        output << "{\"op_id\":" << operation << ",\"name\":\"";
#if SIDECAR_LLAMA_ENABLED
        output << JsonEscape(ggml_op_name(static_cast<ggml_op>(operation)));
#else
        output << "UNAVAILABLE";
#endif
        output << "\"}";
    }
    output << "],\"layer_lookahead_clock_domain\":\"HOST_CALLBACK_ENUMERATION\""
           << ",\"layer_lookahead_authoritative_for_device_deadlines\":false"
           << ",\"layer_lookahead_limitation\":\"PUBLIC_CALLBACK_TIMESTAMPS_DO_NOT_EXPOSE_CUDA_LAYER_EXECUTION\"}";
    return output.str();
}

std::string DemandAnalysisJson(const ModelIndex& model,
                               const InferenceResult& result) {
    if (result.samples.empty()) return "{}";
    const auto& events = result.samples.front().events;
    std::ostringstream output;
    output << "{\"reuse\":[";
    bool first = true;
    for (const auto size : {kBlock32MiB, kBlock64MiB, kBlock128MiB}) {
        if (!first) output << ',';
        first = false;
        const auto reuse = AnalyzeBlockReuse(events, model.tensors, size);
        output << "{\"block_size_bytes\":" << size
               << ",\"evaluation_count\":" << reuse.evaluation_count
               << ",\"union_block_count\":" << reuse.union_block_count
               << ",\"permanently_hot_block_count\":" << reuse.permanently_hot_block_count
               << ",\"new_after_first_block_count\":" << reuse.new_after_first_block_count
               << ",\"mean_repeated_block_fraction\":" << reuse.mean_repeated_block_fraction
               << '}';
    }
    output << "],\"layer_lookahead\":[";
    for (std::uint32_t ahead = 1; ahead <= 4; ++ahead) {
        if (ahead != 1) output << ',';
        const auto lookahead = AnalyzeLayerLookahead(events, ahead);
        output << "{\"layers_ahead\":" << ahead << ",\"lead_time_statistics\":"
               << StatisticsJson(lookahead.lead_times_ns) << '}';
    }
    output << "],\"layer_lookahead_clock_domain\":\"HOST_CALLBACK_ENUMERATION\""
           << ",\"layer_lookahead_authoritative_for_device_deadlines\":false"
           << ",\"layer_lookahead_limitation\":\"PUBLIC_CALLBACK_TIMESTAMPS_DO_NOT_EXPOSE_CUDA_LAYER_EXECUTION\"}";
    return output.str();
}

void PersistRepresentativeDenseShadow(database::Database& database,
                                      const ModelIndex& model,
                                      const InferenceResult& result,
                                      std::int64_t benchmark_id) {
    if (model.is_moe || result.samples.empty()) return;
    const auto& events = result.samples.front().events;
    std::vector<const ObserverEvent32*> first_layer_events;
    std::set<std::int16_t> seen_layers;
    for (const auto& event : events) {
        if (event.eval_index != 1 || event.layer < 0) continue;
        if (seen_layers.insert(event.layer).second) first_layer_events.push_back(&event);
    }
    if (first_layer_events.size() < 2) return;

    std::map<std::int32_t, std::vector<std::uint32_t>> layer_tensors;
    for (const auto& tensor : model.tensors) {
        if (tensor.persistent_weight && tensor.layer)
            layer_tensors[*tensor.layer].push_back(tensor.tensor_id);
    }

    const auto candidates = database.LatestPipelineBenchmarks();
    std::map<std::pair<std::uint64_t, std::string>, database::PipelineBenchmarkSummary> profiles;
    for (const auto& profile : candidates) {
        if (profile.sample_count < 100 || profile.status != "SUCCESS" || profile.p99_ns <= 0)
            continue;
        const auto size = static_cast<std::uint64_t>(profile.chunk_bytes);
        if (size != kBlock32MiB && size != kBlock64MiB && size != kBlock128MiB) continue;
        profiles.try_emplace(std::make_pair(size, profile.pipeline_type), profile);
    }
    if (profiles.empty()) return;

    if (result.samples.front().tokens.empty()) return;
    // Public callback timestamps describe host graph enumeration, not CUDA
    // execution. Causal device lead is therefore unavailable. The full first
    // token duration is retained only as an optimistic oracle upper bound.
    const std::uint64_t causal_lead = 0;
    const std::uint64_t oracle_upper_bound =
        result.samples.front().tokens.front().decode_duration_ns;
    for (std::uint32_t ahead = 1; ahead <= 4; ++ahead) {
        for (std::size_t index = 0; index + ahead < first_layer_events.size(); ++index) {
            const auto* target = first_layer_events[index + ahead];
            const auto tensor_iterator = layer_tensors.find(target->layer);
            if (tensor_iterator == layer_tensors.end()) continue;
            for (const auto block_size : {kBlock32MiB, kBlock64MiB, kBlock128MiB}) {
                const auto projection = ProjectTensorBlocks(
                    model.tensors, tensor_iterator->second, block_size);
                for (const auto& [profile_key, profile] : profiles) {
                    if (profile_key.first != block_size) continue;
                    const auto pipeline_ns = static_cast<std::uint64_t>(profile.p99_ns);
                    for (const auto& [knowledge_name, result_value] :
                         std::initializer_list<std::pair<const char*, ShadowDeadlineResult>>{
                             {"CAUSAL", EvaluateShadowDeadline(KnowledgeMode::Causal,
                                  causal_lead, oracle_upper_bound, pipeline_ns, true)},
                             {"PERFECT_ORACLE", EvaluateShadowDeadline(KnowledgeMode::PerfectOracle,
                                  causal_lead, oracle_upper_bound, pipeline_ns, true)}}) {
                        for (const auto block_id : projection.block_ids) {
                            database::ShadowPipelineRecordInput record;
                            record.inference_benchmark_id = benchmark_id;
                            record.token_index = 0;
                            record.tensor_or_block_id = static_cast<std::int64_t>(block_id);
                            record.block_size_bytes = static_cast<std::int64_t>(block_size);
                            record.pipeline_type = profile.pipeline_type;
                            record.lookahead_assumption = "NEXT_" + std::to_string(ahead) +
                                "_LAYER_DEVICE_DEADLINE_UNAVAILABLE_PUBLIC_CALLBACK_API";
                            record.knowledge_mode = std::string(knowledge_name) == "CAUSAL"
                                ? "CAUSAL_DEVICE_LEAD_UNAVAILABLE_ZERO"
                                : "PERFECT_ORACLE_FULL_TOKEN_WINDOW_UPPER_BOUND";
                            record.available_lead_time_ns = static_cast<std::int64_t>(
                                result_value.available_lead_time_ns);
                            record.pipeline_profile_reference =
                                "WU8_SESSION_" + std::to_string(profile.session_id) +
                                "_P99_NS_" + std::to_string(pipeline_ns) +
                                "_DEPTH_" + std::to_string(profile.buffer_depth) +
                                "_COMPUTE_" + profile.compute_type +
                                "_WINDOW_US_" + std::to_string(profile.compute_window_us) +
                                "_CONTENTION_" + profile.contention_test +
                                "_PHASE_" + profile.phase +
                                "_SAMPLES_" + std::to_string(profile.sample_count);
                            record.deadline_ns = static_cast<std::int64_t>(
                                result_value.available_lead_time_ns);
                            record.predicted_hit = result_value.hit;
                            record.margin_ns = result_value.margin_ns;
                            record.extrapolated = true;
                            database.InsertShadowPipelineResult(record);
                        }
                    }
                }
            }
        }
    }
}

std::string CompilerIdentity() {
#if defined(_MSC_FULL_VER)
    return "MSVC " + std::to_string(_MSC_FULL_VER);
#elif defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("GCC ") + __VERSION__;
#else
    return "unknown";
#endif
}

void PersistModel(const Options& options, const ModelIndex& model) {
    if (!options.database) return;
    if (options.database->has_parent_path())
        std::filesystem::create_directories(options.database->parent_path());
    auto database = database::Database::Open(*options.database,
                                              database::OpenMode::CreateOrOpen);
    database.Initialize();
    database::LlamaDependencyInput dependency;
    dependency.upstream_url = SIDECAR_LLAMA_CPP_UPSTREAM;
    dependency.upstream_commit = SIDECAR_LLAMA_CPP_COMMIT;
    dependency.dirty = SIDECAR_LLAMA_CPP_DIRTY != 0;
    dependency.build_configuration = SIDECAR_LLAMA_BUILD_CONFIGURATION;
    dependency.cuda_enabled = SIDECAR_CUDA_ENABLED != 0;
    dependency.compiler = CompilerIdentity();
    dependency.sidecar_git_commit = CurrentVersionInfo().git_commit;
    dependency.binary_identity_json = std::string("{\"llama_version\":\"")
#if SIDECAR_LLAMA_ENABLED
        + llama_version()
#else
        + "disabled"
#endif
        + "\",\"branch\":\"" SIDECAR_LLAMA_CPP_BRANCH
          "\",\"cuda_toolkit\":\"" SIDECAR_LLAMA_CUDA_TOOLKIT_VERSION
          "\",\"llama_dll_sha256\":\"" SIDECAR_LLAMA_DLL_SHA256
          "\",\"ggml_dll_sha256\":\"" SIDECAR_GGML_DLL_SHA256
          "\",\"ggml_base_dll_sha256\":\"" SIDECAR_GGML_BASE_DLL_SHA256
          "\",\"ggml_cpu_dll_sha256\":\"" SIDECAR_GGML_CPU_DLL_SHA256
          "\",\"ggml_cuda_dll_sha256\":\"" SIDECAR_GGML_CUDA_DLL_SHA256
          "\",\"llama_bench_sha256\":\"" SIDECAR_LLAMA_BENCH_SHA256 "\"}";
    (void)database.UpsertLlamaDependency(dependency);

    database::GgufModelInput record;
    record.model_id = model.sha256;
    record.source = model.provenance.source;
    record.source_repository = model.provenance.source_repository;
    record.source_revision = model.provenance.source_revision;
    record.ollama_model_tag = model.provenance.ollama_model_tag;
    record.ollama_digest = model.provenance.ollama_digest;
    if (model.provenance.original_blob_path) record.original_blob_path = model.provenance.original_blob_path->string();
    if (model.provenance.alias_path) record.alias_path = model.provenance.alias_path->string();
    record.local_path = model.path.string();
    record.sha256 = model.sha256;
    record.file_size_bytes = static_cast<std::int64_t>(model.file_size_bytes);
    record.gguf_version = model.gguf_version;
    record.architecture = model.architecture;
    record.quantization = model.quantization;
    if (model.parameter_count) record.parameter_count = static_cast<std::int64_t>(*model.parameter_count);
    record.tensor_count = static_cast<std::int64_t>(model.tensors.size());
    if (model.layer_count) record.layer_count = *model.layer_count;
    if (model.expert_count) record.expert_count = *model.expert_count;
    record.metadata_json = model.metadata_json;
    record.verification_status = model.verification_status;
    record.verification_detail = model.verification_detail;
    record.llama_cpp_commit = model.llama_cpp_commit;
    database.UpsertGgufModel(record);

    std::vector<database::GgufTensorInput> tensors;
    tensors.reserve(model.tensors.size());
    for (const auto& tensor : model.tensors) {
        database::GgufTensorInput input;
        input.model_id = model.sha256;
        input.tensor_id = tensor.tensor_id;
        input.tensor_name = tensor.name;
        input.tensor_type = tensor.type;
        input.dimensions_json = DimensionsJson(tensor.dimensions);
        input.tensor_bytes = static_cast<std::int64_t>(tensor.bytes);
        input.file_offset = static_cast<std::int64_t>(tensor.file_offset);
        input.file_span_bytes = static_cast<std::int64_t>(tensor.file_span_bytes);
        input.alignment_bytes = tensor.alignment_bytes;
        if (tensor.layer) input.layer_index = *tensor.layer;
        if (tensor.expert) input.expert_index = *tensor.expert;
        input.tensor_role = tensor.role;
        input.persistent_weight = tensor.persistent_weight;
        tensors.push_back(std::move(input));
    }
    database.ReplaceGgufTensors(model.sha256, tensors);

    auto persist_storage = [&](const std::filesystem::path& path, std::string role) {
        const auto storage = ResolveModelStorage(path, std::move(role), options.copy_relationship);
        database::GgufModelStorageInput input;
        input.model_id = model.sha256;
        input.requested_path = storage.requested_path.string();
        input.resolved_path = storage.resolved_path.string();
        input.volume_path = storage.volume_path;
        if (!storage.volume_unique_id.empty()) input.volume_unique_id = storage.volume_unique_id;
        input.physical_device = storage.physical_device;
        if (storage.physical_disk_number) input.physical_disk_number = *storage.physical_disk_number;
        input.device_model = storage.device_model;
        input.bus_type = storage.bus_type;
        input.is_samsung_990_pro = storage.is_samsung_990_pro;
        input.is_usb_external = storage.is_usb_external;
        input.storage_role = storage.storage_role;
        if (!storage.copy_relationship.empty()) input.copy_relationship = storage.copy_relationship;
        database.UpsertGgufModelStorage(input);
    };
    persist_storage(model.path, options.storage_role.empty()
        ? "MODEL_EXECUTION_OR_INSPECTION_PATH" : options.storage_role);
    if (model.provenance.original_blob_path &&
        std::filesystem::absolute(*model.provenance.original_blob_path) != model.path)
        persist_storage(*model.provenance.original_blob_path, "OLLAMA_SOURCE_BLOB");
    if (model.provenance.alias_path &&
        std::filesystem::absolute(*model.provenance.alias_path) != model.path)
        persist_storage(*model.provenance.alias_path, "SIDECAR_ALIAS_OR_COPY");

    if (model.is_moe) {
        std::uint64_t vram_capacity = 0;
        auto provider = hardware::CreateNativeDiscoveryProvider();
        hardware::HardwareDiscoveryService discovery(*provider);
        const auto report = discovery.Discover();
        for (const auto& gpu : report.snapshot.gpus) {
            if (gpu.vram_bytes) vram_capacity = (std::max)(vram_capacity, *gpu.vram_bytes);
        }
        const auto profile = AnalyzeMoeStaticProfile(model, vram_capacity);
        database::MoeDemandProfileRecordInput input;
        input.model_id = model.sha256;
        input.status = profile.status;
        input.expert_count = profile.expert_count;
        input.selected_experts_observable = false;
        input.expert_working_set_bytes = static_cast<std::int64_t>(profile.estimated_bytes_per_expert);
        input.reuse_json =
            "{\"evidence\":\"STATIC_POSSIBLE_EXPERT_DEMAND\",\"router_always_hot_bytes\":" +
            std::to_string(profile.router_always_hot_bytes) +
            ",\"packed_expert_tensor_bytes\":" +
            std::to_string(profile.packed_expert_tensor_bytes) +
            ",\"estimated_bytes_per_expert\":" +
            std::to_string(profile.estimated_bytes_per_expert) +
            ",\"vram_capacity_bytes\":" + std::to_string(vram_capacity) +
            ",\"candidate_experts_in_capacity\":" +
            std::to_string(profile.candidate_experts_in_capacity) + "}";
        input.causal_cue_json =
            "{\"status\":\"MOE_SELECTION_UNRESOLVED\",\"reason\":\"STRUCTURAL_INSPECTION_ONLY_NONAUTHORITATIVE_STORAGE_LOCATION\"}";
        database.InsertMoeDemandProfile(input);
    }
}

ModelIndex Inspect(const Options& options, bool compute_hash) {
    if (options.model.empty()) throw std::invalid_argument("--model <GGUF-path> is required");
    auto model = InspectGguf(options.model, options.provenance, SIDECAR_LLAMA_CPP_COMMIT,
                             compute_hash && !options.supplied_sha256);
    if (options.supplied_sha256) {
        if (options.supplied_sha256->size() != 64 ||
            !std::all_of(options.supplied_sha256->begin(), options.supplied_sha256->end(),
                [](unsigned char value) { return std::isxdigit(value) != 0; })) {
            throw std::invalid_argument("--sha256 requires exactly 64 hexadecimal characters");
        }
        model.sha256 = *options.supplied_sha256;
        std::transform(model.sha256.begin(), model.sha256.end(), model.sha256.begin(),
            [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    }
    return model;
}

InferenceConfiguration MakeConfiguration(const Options& options) {
    if (options.model.empty()) throw std::invalid_argument("--model <GGUF-path> is required");
    if (options.authoritative) {
        const auto storage = ResolveModelStorage(options.model, "PRIMARY_AUTHORITATIVE_DENSE",
                                                  "AUTHORITY_STORAGE_GATE");
        if (!storage.is_samsung_990_pro || storage.is_usb_external) {
            throw std::runtime_error(
                "authoritative inference requires a Samsung 990 PRO-backed model path");
        }
    }
    InferenceConfiguration configuration;
    configuration.model_path = options.model;
    configuration.prompt_tokens = options.prompt;
    configuration.generated_tokens = options.generate;
    configuration.context_size = options.context;
    configuration.gpu_layers = options.gpu_layers;
    configuration.warmups = options.warmups;
    configuration.repetitions = options.repetitions;
    configuration.observer_mode = options.observer;
    configuration.authoritative = options.authoritative;
    configuration.flight_recorder_path = options.trace;
    if (configuration.observer_mode == ObserverMode::LightLayerFlightRecorder) {
        auto provider = hardware::CreateNativeDiscoveryProvider();
        hardware::HardwareDiscoveryService discovery(*provider);
        configuration.machine_hash = discovery.Discover().identity.machine_hash;
    }
    return configuration;
}

std::uint64_t TensorBytes(const ModelIndex& model,
                          const std::vector<std::uint32_t>& tensor_ids) {
    std::uint64_t bytes = 0;
    for (const auto id : tensor_ids) {
        if (id < model.tensors.size()) bytes += model.tensors[id].bytes;
    }
    return bytes;
}

std::string ProjectionJson(const ModelIndex& model,
                           const std::vector<std::uint32_t>& tensor_ids) {
    std::ostringstream output;
    output << '{';
    bool first = true;
    for (const auto size : {kBlock32MiB, kBlock64MiB, kBlock128MiB}) {
        if (!first) output << ',';
        first = false;
        const auto projection = ProjectTensorBlocks(model.tensors, tensor_ids, size);
        output << '"' << size << "\":{\"count\":" << projection.block_ids.size()
               << ",\"block_ids\":" << ArrayJson(projection.block_ids)
               << ",\"useful_bytes\":" << projection.useful_tensor_bytes
               << ",\"overfetch_bytes\":" << projection.overfetch_bytes << '}';
    }
    output << '}';
    return output.str();
}

void PersistInference(const Options& options, const ModelIndex& model,
                      const InferenceConfiguration& configuration,
                      const InferenceResult& result, std::string_view phase) {
    if (!options.database) return;
    if (model.sha256.empty() || model.sha256 == "NOT_COMPUTED") {
        throw std::runtime_error(
            "persistence requires finalized model identity; pass --sha256 with the verified digest");
    }
    if (options.database->has_parent_path())
        std::filesystem::create_directories(options.database->parent_path());
    auto database = database::Database::Open(*options.database,
                                              database::OpenMode::CreateOrOpen);
    database.Initialize();
    auto provider = hardware::CreateNativeDiscoveryProvider();
    hardware::HardwareDiscoveryService discovery(*provider);
    const auto hardware_report = discovery.Discover();
    hardware::PersistDiscovery(database, hardware_report);

    database::BenchmarkSessionInput session;
    session.machine_hash = hardware_report.identity.machine_hash;
    session.sidecar_spec_version = CurrentVersionInfo().spec_version;
    session.sidecar_git_commit = CurrentVersionInfo().git_commit;
    session.llama_cpp_git_commit = SIDECAR_LLAMA_CPP_COMMIT;
    session.trace_mode = std::string(ToString(configuration.observer_mode));
    session.notes = "WU9 real llama.cpp/GGUF observation; hot data accumulated in RAM and persisted after timing";
    const auto session_id = database.StartBenchmarkSession(session);

    try {
        database.BeginWriteTransaction();
        database::InferenceFixtureInput fixture;
        fixture.fixture_id = result.fixture_id;
        fixture.fixture_name = configuration.prompt_tokens <= 128 ? "SHORT" :
                               configuration.prompt_tokens <= 512 ? "MEDIUM" : "LONG";
        fixture.fixture_version = 1;
        fixture.text_sha256 = result.fixture_text_sha256;
        fixture.text_bytes = static_cast<std::int64_t>(
            std::string("SIDECAR-WU9-DETERMINISTIC-PROMPT-V1:").size() +
            std::to_string(configuration.prompt_tokens).size());
        fixture.token_count = static_cast<std::int64_t>(result.prompt_token_ids.size());
        fixture.token_ids_json = ArrayJson(result.prompt_token_ids);
        fixture.description = "Versioned deterministic repeated Sidecar observation fixture; tokenization occurs outside prompt timing";
        database.UpsertInferenceFixture(fixture);

        const std::string canonical = model.sha256 + "\n" + std::string(phase) + "\n" +
            std::string(ToString(configuration.observer_mode)) + "\n" +
            std::to_string(configuration.gpu_layers) + "\n" +
            std::to_string(configuration.context_size) + "\n" +
            std::to_string(configuration.prompt_tokens) + "\n" +
            std::to_string(configuration.generated_tokens) + "\n" +
            std::to_string(configuration.warmups) + "\n" +
            std::to_string(configuration.repetitions) + "\n" +
            (configuration.authoritative ? "AUTHORITATIVE" : "NONAUTHORITATIVE");
        database::InferenceConfigurationRecordInput config_record;
        config_record.session_id = session_id;
        config_record.model_id = model.sha256;
        config_record.fixture_id = result.fixture_id;
        config_record.configuration_hash = core::Sha256Hex(canonical);
        config_record.phase = std::string(phase);
        config_record.observer_mode = std::string(ToString(configuration.observer_mode));
        config_record.gpu_layers = configuration.gpu_layers;
        config_record.context_size = configuration.context_size;
        config_record.prompt_tokens = configuration.prompt_tokens;
        config_record.generated_tokens = configuration.generated_tokens;
        config_record.warmups = configuration.warmups;
        config_record.repetitions = configuration.repetitions;
        config_record.seed = configuration.seed;
        config_record.backend_mode = configuration.gpu_layers < 0 ? "CUDA_FULL_OFFLOAD_REQUESTED" : "LLAMA_NATIVE_PARTIAL_OFFLOAD_CONTROL";
        config_record.load_mode = "MMAP_READ_ONLY";
        config_record.status = result.status;
        const auto configuration_id = database.InsertInferenceConfiguration(config_record);

        std::vector<std::uint64_t> prompt_samples, decode_samples, token_samples,
            initial_sampling_samples, per_token_sampling_samples;
        for (const auto& sample : result.samples) {
            prompt_samples.push_back(sample.prompt_ns);
            decode_samples.push_back(sample.decode_total_ns);
            initial_sampling_samples.push_back(sample.initial_sampling_ns);
            for (const auto& token : sample.tokens) {
                token_samples.push_back(token.decode_duration_ns);
                per_token_sampling_samples.push_back(token.sampling_ns);
            }
        }
        database::InferenceBenchmarkRecordInput benchmark_record;
        benchmark_record.configuration_id = configuration_id;
        benchmark_record.status = result.status;
        benchmark_record.model_load_ns = static_cast<std::int64_t>(result.model_load_ns);
        benchmark_record.prompt_statistics_json = StatisticsJson(prompt_samples);
        benchmark_record.decode_statistics_json = StatisticsJson(token_samples);
        benchmark_record.perf_context_json =
            "{\"prompt_tokenization_ns\":" + std::to_string(result.prompt_tokenization_ns) +
            ",\"sampling\":\"greedy_separately_timed\",\"n_gpu_layers_requested\":" +
            std::to_string(configuration.gpu_layers) +
            ",\"authoritative\":" + (configuration.authoritative ? "true" : "false") +
            ",\"initial_sampling_statistics\":" + StatisticsJson(initial_sampling_samples) +
            ",\"per_token_sampling_statistics\":" + StatisticsJson(per_token_sampling_samples) +
            ",\"runtime_configuration\":" + result.runtime_configuration_json +
            ",\"demand_analysis\":" + DemandAnalysisJson(model, result) +
            ",\"graph_split_detail\":\"" + JsonEscape(result.graph_split_detail) +
            "\",\"model_storage\":" +
            (result.model_storage ? StorageProvenanceToJson(*result.model_storage) : "null") + "}";
        benchmark_record.telemetry_json =
            "{\"capture\":\"POST_TIMED_RUN_LOW_RATE\",\"hardware\":" +
            hardware::DiscoveryReportToJson(hardware_report) + "}";
        benchmark_record.graph_split_detail = result.graph_split_detail;
        benchmark_record.message = result.message;
        const auto benchmark_id = database.InsertInferenceBenchmark(benchmark_record);

        for (const auto& sample : result.samples) {
            database::InferenceSampleRecordInput sample_record;
            sample_record.benchmark_id = benchmark_id;
            sample_record.sample_index = sample.sample_index;
            sample_record.prompt_ns = static_cast<std::int64_t>(sample.prompt_ns);
            sample_record.decode_total_ns = static_cast<std::int64_t>(sample.decode_total_ns);
            sample_record.prompt_tokens_per_second = sample.prompt_tokens_per_second;
            sample_record.generation_tokens_per_second = sample.generation_tokens_per_second;
            sample_record.graph_node_count = static_cast<std::int64_t>(sample.graph_node_count);
            sample_record.observer_event_count = static_cast<std::int64_t>(sample.events.size());
            sample_record.dropped_event_count = static_cast<std::int64_t>(sample.dropped_events);
            database.InsertInferenceSample(sample_record);

            database::ObserverProfileRecordInput profile;
            profile.benchmark_id = benchmark_id;
            profile.observer_mode = std::string(ToString(configuration.observer_mode));
            profile.ask_calls = static_cast<std::int64_t>(sample.ask_calls);
            profile.materialized_calls = static_cast<std::int64_t>(sample.materialized_calls);
            profile.compact_events = static_cast<std::int64_t>(sample.events.size());
            profile.dropped_events = static_cast<std::int64_t>(sample.dropped_events);
            profile.callback_contract = result.callback_contract;
            profile.dictionary_json = ObserverDictionaryJson(model, sample.events);
            const auto profile_id = database.InsertObserverProfile(profile);
            for (std::size_t event_index = 0; event_index < sample.events.size(); ++event_index) {
                const auto& event = sample.events[event_index];
                database::ObserverEventRecordInput event_record;
                event_record.observer_profile_id = profile_id;
                event_record.event_index = static_cast<std::int64_t>(event_index);
                event_record.timestamp_ns = static_cast<std::int64_t>(event.timestamp_ns);
                event_record.eval_index = event.eval_index;
                event_record.sequence_index = event.sequence_index;
                if (event.tensor_id_0 != 0xFFFFFFFFU) event_record.tensor_id_0 = event.tensor_id_0;
                if (event.tensor_id_1 != 0xFFFFFFFFU) event_record.tensor_id_1 = event.tensor_id_1;
                event_record.op_id = event.op_id;
                if (event.layer >= 0) event_record.layer_index = event.layer;
                event_record.event_class = event.event_class;
                event_record.flags = event.flags;
                database.InsertObserverEvent(event_record);
            }

            std::set<std::pair<std::uint32_t, std::uint32_t>> recorded_demand;
            for (std::size_t event_index = 0; event_index < sample.events.size(); ++event_index) {
                const auto& event = sample.events[event_index];
                for (const auto tensor_id : {event.tensor_id_0, event.tensor_id_1}) {
                    if (tensor_id == 0xFFFFFFFFU || tensor_id >= model.tensors.size() ||
                        !recorded_demand.emplace(event.eval_index, tensor_id).second) continue;
                    const auto& tensor = model.tensors[tensor_id];
                    database::TensorDemandRecordInput demand_record;
                    demand_record.benchmark_id = benchmark_id;
                    demand_record.token_index = event.eval_index == 0 ? -1 :
                        static_cast<std::int64_t>(event.eval_index - 1U);
                    demand_record.sequence_index = event.sequence_index;
                    demand_record.tensor_id = tensor_id;
                    demand_record.first_event_index = static_cast<std::int64_t>(event_index);
                    if (tensor.layer) demand_record.layer_index = *tensor.layer;
                    else if (event.layer >= 0) demand_record.layer_index = event.layer;
                    demand_record.tensor_bytes = static_cast<std::int64_t>(tensor.bytes);
                    demand_record.file_offset = static_cast<std::int64_t>(tensor.file_offset);
                    demand_record.causal_known_at_ns = static_cast<std::int64_t>(event.timestamp_ns);
                    demand_record.demand_ns = static_cast<std::int64_t>(event.timestamp_ns);
                    database.UpsertTensorDemandEvent(demand_record);
                }
            }

            for (const auto& token : sample.tokens) {
                std::vector<ObserverEvent32> token_events;
                for (const auto& event : sample.events) {
                    if (event.eval_index == token.token_index + 1U) token_events.push_back(event);
                }
                const auto tensor_ids = DeduplicateDemand(token_events);
                database::InferenceTokenRecordInput token_record;
                token_record.benchmark_id = benchmark_id;
                token_record.sample_index = sample.sample_index;
                token_record.token_index = token.token_index;
                token_record.input_token_id = token.input_token_id;
                token_record.output_token_id = token.output_token_id;
                token_record.context_depth = token.context_depth;
                token_record.decode_start_ns = static_cast<std::int64_t>(token.decode_start_ns);
                token_record.decode_end_ns = static_cast<std::int64_t>(token.decode_end_ns);
                token_record.decode_duration_ns = static_cast<std::int64_t>(token.decode_duration_ns);
                token_record.observer_mode = std::string(ToString(configuration.observer_mode));
                token_record.graph_node_count = static_cast<std::int64_t>(token.graph_node_count);
                token_record.layer_sequence_reference =
                    LayerSequenceJson(sample.events, token.token_index + 1U);
                token_record.unique_weight_tensor_count = static_cast<std::int64_t>(tensor_ids.size());
                token_record.unique_weight_bytes = static_cast<std::int64_t>(TensorBytes(model, tensor_ids));
                token_record.projected_blocks_json = ProjectionJson(model, tensor_ids);
                database.InsertInferenceToken(token_record);

                for (const auto block_size : {kBlock32MiB, kBlock64MiB, kBlock128MiB}) {
                    const auto projection = ProjectTensorBlocks(model.tensors, tensor_ids, block_size);
                    database::TensorBlockProjectionRecordInput projection_record;
                    projection_record.benchmark_id = benchmark_id;
                    projection_record.model_id = model.sha256;
                    projection_record.block_size_bytes = static_cast<std::int64_t>(block_size);
                    projection_record.scope = "TOKEN_UNIQUE_DEMAND_SAMPLE_" +
                        std::to_string(sample.sample_index);
                    projection_record.scope_index = token.token_index;
                    projection_record.useful_tensor_bytes = static_cast<std::int64_t>(projection.useful_tensor_bytes);
                    projection_record.projected_block_count = static_cast<std::int64_t>(projection.block_ids.size());
                    projection_record.projected_bytes = static_cast<std::int64_t>(projection.projected_bytes);
                    projection_record.overfetch_bytes = static_cast<std::int64_t>(projection.overfetch_bytes);
                    projection_record.overfetch_ratio = projection.overfetch_ratio;
                    projection_record.block_ids_json = ArrayJson(projection.block_ids);
                    database.InsertTensorBlockProjection(projection_record);
                }
            }

            const auto all_demand = DeduplicateDemand(sample.events);
            for (const auto block_size : {kBlock32MiB, kBlock64MiB, kBlock128MiB}) {
                const auto projection = ProjectTensorBlocks(model.tensors, all_demand, block_size);
                database::TensorBlockProjectionRecordInput projection_record;
                projection_record.benchmark_id = benchmark_id;
                projection_record.model_id = model.sha256;
                projection_record.block_size_bytes = static_cast<std::int64_t>(block_size);
                projection_record.scope = "RUN_UNIQUE_DEMAND";
                projection_record.scope_index = sample.sample_index;
                projection_record.useful_tensor_bytes = static_cast<std::int64_t>(projection.useful_tensor_bytes);
                projection_record.projected_block_count = static_cast<std::int64_t>(projection.block_ids.size());
                projection_record.projected_bytes = static_cast<std::int64_t>(projection.projected_bytes);
                projection_record.overfetch_bytes = static_cast<std::int64_t>(projection.overfetch_bytes);
                projection_record.overfetch_ratio = projection.overfetch_ratio;
                projection_record.block_ids_json = ArrayJson(projection.block_ids);
                database.InsertTensorBlockProjection(projection_record);
            }
        }
        if ((phase == "DEMAND" || phase == "OBSERVATION") &&
            configuration.observer_mode != ObserverMode::None) {
            PersistRepresentativeDenseShadow(database, model, result, benchmark_id);
        }
        database.CommitWriteTransaction();
        database.CompleteBenchmarkSession(session_id, result.status == "PASS" ? "COMPLETE" : "FAILED");
    } catch (...) {
        database.RollbackWriteTransaction();
        database.CompleteBenchmarkSession(session_id, "FAILED");
        throw;
    }
}

void PersistOverhead(const Options& options, const ModelIndex& model,
                     const std::vector<std::uint64_t>& baseline_prompt,
                     const std::vector<std::uint64_t>& observed_prompt,
                     const std::vector<std::uint64_t>& baseline_decode,
                     const std::vector<std::uint64_t>& observed_decode,
                     const PairedOverheadStatistics& prompt,
                     const PairedOverheadStatistics& decode) {
    if (!options.database) return;
    if (model.sha256.empty() || model.sha256 == "NOT_COMPUTED")
        throw std::runtime_error("overhead persistence requires --sha256 with verified model digest");
    auto database = database::Database::Open(*options.database, database::OpenMode::CreateOrOpen);
    database.Initialize();
    auto provider = hardware::CreateNativeDiscoveryProvider();
    hardware::HardwareDiscoveryService discovery(*provider);
    const auto report = discovery.Discover();
    hardware::PersistDiscovery(database, report);
    database::BenchmarkSessionInput session;
    session.machine_hash = report.identity.machine_hash;
    session.sidecar_spec_version = CurrentVersionInfo().spec_version;
    session.sidecar_git_commit = CurrentVersionInfo().git_commit;
    session.llama_cpp_git_commit = SIDECAR_LLAMA_CPP_COMMIT;
    session.trace_mode = std::string(ToString(options.observer));
    session.notes = "WU9 interleaved paired real-inference observer overhead qualification";
    const auto session_id = database.StartBenchmarkSession(session);
    auto insert = [&](std::string phase, const auto& baseline, const auto& observed,
                      const PairedOverheadStatistics& statistics) {
        database::ObserverOverheadRecordInput record;
        record.session_id = session_id;
        record.model_id = model.sha256;
        record.phase = std::move(phase);
        record.observer_mode = std::string(ToString(options.observer));
        record.paired_repetitions = static_cast<std::int64_t>(statistics.pairs);
        record.baseline_samples_json = ArrayJson(baseline);
        record.observed_samples_json = ArrayJson(observed);
        record.paired_delta_statistics_json =
            "{\"paired_delta_mean_ns\":" + std::to_string(statistics.paired_delta_mean_ns) +
            ",\"paired_delta_median_ns\":" + std::to_string(statistics.paired_delta_median_ns) +
            ",\"paired_mean_overhead_percent\":" + std::to_string(statistics.paired_mean_overhead_percent) +
            ",\"paired_median_overhead_percent\":" + std::to_string(statistics.overhead_percent) +
            ",\"paired_p95_overhead_percent\":" + std::to_string(statistics.paired_p95_overhead_percent) +
            ",\"paired_variance_percent_squared\":" + std::to_string(statistics.paired_variance_percent_squared) +
            ",\"paired_stddev_percent\":" + std::to_string(statistics.paired_stddev_percent) +
            ",\"throughput_change_percent\":" + std::to_string(statistics.throughput_change_percent) +
            ",\"baseline_mean_ns\":" + std::to_string(statistics.baseline_mean_ns) +
            ",\"observed_mean_ns\":" + std::to_string(statistics.observed_mean_ns) +
            ",\"baseline_median_ns\":" + std::to_string(statistics.baseline_median_ns) +
            ",\"observed_median_ns\":" + std::to_string(statistics.observed_median_ns) + "}";
        record.overhead_percent = statistics.overhead_percent;
        record.confidence_interval_json =
            "{\"method\":\"empirical_paired_percentile\",\"low_percent\":" +
            std::to_string(statistics.ci95_low_percent) + ",\"high_percent\":" +
            std::to_string(statistics.ci95_high_percent) + "}";
        record.gate_passed = statistics.gate_passed;
        database.InsertObserverOverhead(record);
    };
    insert("PROMPT", baseline_prompt, observed_prompt, prompt);
    insert("DECODE_SEQUENCE", baseline_decode, observed_decode, decode);
    database.CompleteBenchmarkSession(session_id, "COMPLETE");
}

void PrintLlamaUsage() {
    std::cout
        << "  sidecar-lab llama info [--json]\n"
        << "  sidecar-lab llama model inspect <path> [--json] [--tensors] [provenance] [--database path]\n"
        << "  sidecar-lab llama model verify <path> [--json] [provenance] [--database path]\n"
        << "  sidecar-lab llama baseline --model path [--prompt-tokens N] [--generate-tokens N] [--json]\n"
        << "  sidecar-lab llama observe --model path --observer MODE [--trace path] [--json]\n"
        << "  sidecar-lab llama overhead --model path --observer MODE [--repetitions N] [--json]\n"
        << "  sidecar-lab llama demand --model path [--json]\n"
        << "  sidecar-lab llama shadow --causal-lead-ns N --oracle-lead-ns N --pipeline-ns N [--json]\n"
        << "  sidecar-lab llama report --database path [--json]\n"
        << "  sidecar-lab llama validate --model path [--json]\n";
}

int Info(const Options& options) {
    if (options.json) {
        std::cout << "{\"llama_enabled\":" << (SIDECAR_LLAMA_ENABLED ? "true" : "false")
                  << ",\"cuda_enabled\":" << (SIDECAR_CUDA_ENABLED ? "true" : "false")
                  << ",\"upstream\":\"" << SIDECAR_LLAMA_CPP_UPSTREAM
                  << "\",\"commit\":\"" << SIDECAR_LLAMA_CPP_COMMIT
                  << "\",\"branch\":\"" << SIDECAR_LLAMA_CPP_BRANCH
                  << "\",\"build_configuration\":\"" << SIDECAR_LLAMA_BUILD_CONFIGURATION
                  << "\",\"cuda_toolkit\":\"" << SIDECAR_LLAMA_CUDA_TOOLKIT_VERSION
                  << "\",\"llama_bench_sha256\":\"" << SIDECAR_LLAMA_BENCH_SHA256
                  << "\",\"public_callback\":\"llama_context_params.cb_eval\"}\n";
    } else {
        std::cout << "SIDECAR LLAMA/GGUF OBSERVATION\n"
                  << "llama.cpp: " << (SIDECAR_LLAMA_ENABLED ? "enabled" : "disabled") << '\n'
                  << "CUDA inference: " << (SIDECAR_CUDA_ENABLED ? "enabled" : "disabled") << '\n'
                  << "Upstream: " << SIDECAR_LLAMA_CPP_UPSTREAM << '\n'
                  << "Commit: " << SIDECAR_LLAMA_CPP_COMMIT << '\n'
                  << "Branch: " << SIDECAR_LLAMA_CPP_BRANCH << '\n'
                  << "Integration: public llama_context_params.cb_eval; no source patch\n";
    }
    return 0;
}

int ModelCommand(std::string_view action, const Options& options) {
    auto model = Inspect(options, options.hash);
    if (action == "verify") {
        auto configuration = MakeConfiguration(options);
        configuration.prompt_tokens = 8;
        configuration.generated_tokens = 1;
        configuration.context_size = 512;
        configuration.warmups = 0;
        configuration.repetitions = 1;
        const auto validation = RunInference(configuration, &model);
        if (validation.status == "PASS") {
            model.verification_status = "PINNED_LLAMA_LOAD_AND_DECODE_PASS";
            model.verification_detail = validation.message;
        } else if (validation.status == "SKIPPED_UNSUPPORTED") {
            model.verification_status = "GGUF_VALID_INFERENCE_SKIPPED_UNSUPPORTED";
            model.verification_detail = validation.message;
        } else {
            model.verification_status = "FAILED";
            model.verification_detail = validation.message;
        }
    }
    PersistModel(options, model);
    std::cout << (options.json ? ModelIndexToJson(model, options.tensors)
                              : FormatModelIndex(model, options.tensors)) << (options.json ? "\n" : "");
    return model.verification_status == "FAILED" ? 4 : 0;
}

int InferenceCommand(std::string_view action, Options options) {
    if (action == "baseline") options.observer = ObserverMode::None;
    if (action == "demand") options.observer = ObserverMode::LightNode;
    if (action == "validate") {
        options.prompt = 8; options.generate = 2; options.context = 512;
        options.warmups = 0; options.repetitions = 1;
        if (options.observer == ObserverMode::None) options.observer = ObserverMode::LightLayer;
    }
    auto model = Inspect(options, false);
    const auto configuration = MakeConfiguration(options);
    const auto result = RunInference(configuration, &model);
    PersistInference(options, model, configuration, result,
                     action == "baseline" ? "BASELINE" :
                     action == "demand" ? "DEMAND" :
                     action == "validate" ? "VALIDATION" : "OBSERVATION");
    if (action == "demand" && result.status == "PASS" && !result.samples.empty()) {
        const auto demand = DeduplicateDemand(result.samples.front().events);
        const auto p32 = ProjectTensorBlocks(model.tensors, demand, kBlock32MiB);
        const auto p64 = ProjectTensorBlocks(model.tensors, demand, kBlock64MiB);
        const auto p128 = ProjectTensorBlocks(model.tensors, demand, kBlock128MiB);
        if (options.json) {
            std::cout << "{\"inference\":" << InferenceResultToJson(result)
                      << ",\"unique_tensors\":" << demand.size()
                      << ",\"projections\":["
                      << "{\"block_size\":" << p32.block_size_bytes << ",\"blocks\":" << p32.block_ids.size() << ",\"useful_bytes\":" << p32.useful_tensor_bytes << ",\"overfetch_bytes\":" << p32.overfetch_bytes << "},"
                      << "{\"block_size\":" << p64.block_size_bytes << ",\"blocks\":" << p64.block_ids.size() << ",\"useful_bytes\":" << p64.useful_tensor_bytes << ",\"overfetch_bytes\":" << p64.overfetch_bytes << "},"
                      << "{\"block_size\":" << p128.block_size_bytes << ",\"blocks\":" << p128.block_ids.size() << ",\"useful_bytes\":" << p128.useful_tensor_bytes << ",\"overfetch_bytes\":" << p128.overfetch_bytes << "}]"
                      << ",\"analysis\":" << DemandAnalysisJson(model, result) << "}\n";
        } else {
            std::cout << "demand status=" << result.status << " unique_tensors=" << demand.size()
                      << " blocks_32MiB=" << p32.block_ids.size()
                      << " blocks_64MiB=" << p64.block_ids.size()
                      << " blocks_128MiB=" << p128.block_ids.size() << '\n';
        }
    } else {
        std::cout << (options.json ? InferenceResultToJson(result)
                                  : "llama " + std::string(action) + " status=" + result.status +
                                    " message=" + result.message + "\n");
        if (options.json) std::cout << '\n';
    }
    return result.status == "PASS" || result.status == "SKIPPED_UNSUPPORTED" ? 0 : 5;
}

int Overhead(Options options) {
    if (options.observer == ObserverMode::None)
        throw std::invalid_argument("overhead requires --observer other than none");
    auto model = Inspect(options, false);
    auto baseline_configuration = MakeConfiguration(options);
    baseline_configuration.observer_mode = ObserverMode::None;
    baseline_configuration.repetitions = 1;
    auto observed_configuration = MakeConfiguration(options);
    observed_configuration.repetitions = 1;
    const auto original_trace_path = observed_configuration.flight_recorder_path;
    std::vector<std::uint64_t> baseline_prompt, observed_prompt, baseline_decode, observed_decode;
    for (std::uint32_t pair = 0; pair < options.repetitions; ++pair) {
        if (original_trace_path) {
            const auto base = *original_trace_path;
            observed_configuration.flight_recorder_path =
                base.parent_path() / (base.stem().string() + "-pair-" +
                std::to_string(pair) + base.extension().string());
        }
        InferenceResult baseline, observed;
        if ((pair & 1U) == 0U) {
            baseline = RunInference(baseline_configuration, &model);
            observed = RunInference(observed_configuration, &model);
        } else {
            observed = RunInference(observed_configuration, &model);
            baseline = RunInference(baseline_configuration, &model);
        }
        if (baseline.status != "PASS" || observed.status != "PASS") {
            const std::string status = baseline.status != "PASS" ? baseline.status : observed.status;
            const std::string message = baseline.status != "PASS" ? baseline.message : observed.message;
            std::cout << (options.json ? "{\"status\":\"" + status + "\",\"message\":\"" + message + "\"}\n"
                                      : "overhead status=" + status + " message=" + message + "\n");
            return status == "SKIPPED_UNSUPPORTED" ? 0 : 5;
        }
        baseline_prompt.push_back(baseline.samples.front().prompt_ns);
        baseline_decode.push_back(baseline.samples.front().decode_total_ns);
        observed_prompt.push_back(observed.samples.front().prompt_ns);
        observed_decode.push_back(observed.samples.front().decode_total_ns);
    }
    const auto prompt = CalculatePairedOverhead(baseline_prompt, observed_prompt);
    const auto decode = CalculatePairedOverhead(baseline_decode, observed_decode);
    PersistOverhead(options, model, baseline_prompt, observed_prompt,
                    baseline_decode, observed_decode, prompt, decode);
    if (options.json) {
        const auto gate_status = [](const PairedOverheadStatistics& value) {
            if (value.gate_passed) return "PASS";
            return value.overhead_percent <= 2.0 ? "MARGINAL" : "FAIL";
        };
        std::cout << "{\"status\":\"PASS\",\"observer_mode\":\"" << ToString(options.observer)
                  << "\",\"pairs\":" << prompt.pairs
                  << ",\"prompt_mean_overhead_percent\":" << prompt.paired_mean_overhead_percent
                  << ",\"prompt_overhead_percent\":" << prompt.overhead_percent
                  << ",\"prompt_p95_overhead_percent\":" << prompt.paired_p95_overhead_percent
                  << ",\"prompt_stddev_percent\":" << prompt.paired_stddev_percent
                  << ",\"prompt_throughput_change_percent\":" << prompt.throughput_change_percent
                  << ",\"prompt_ci95_high_percent\":" << prompt.ci95_high_percent
                  << ",\"prompt_gate_passed\":" << (prompt.gate_passed ? "true" : "false")
                  << ",\"prompt_gate_status\":\"" << gate_status(prompt) << '"'
                  << ",\"decode_mean_overhead_percent\":" << decode.paired_mean_overhead_percent
                  << ",\"decode_overhead_percent\":" << decode.overhead_percent
                  << ",\"decode_p95_overhead_percent\":" << decode.paired_p95_overhead_percent
                  << ",\"decode_stddev_percent\":" << decode.paired_stddev_percent
                  << ",\"decode_throughput_change_percent\":" << decode.throughput_change_percent
                  << ",\"decode_ci95_high_percent\":" << decode.ci95_high_percent
                  << ",\"decode_gate_passed\":" << (decode.gate_passed ? "true" : "false")
                  << ",\"decode_gate_status\":\"" << gate_status(decode) << "\"}\n";
    } else {
        std::cout << "observer=" << ToString(options.observer) << " pairs=" << prompt.pairs
                  << " prompt_overhead_pct=" << prompt.overhead_percent
                  << " prompt_gate=" << (prompt.gate_passed ? "PASS" : "FAIL")
                  << " decode_overhead_pct=" << decode.overhead_percent
                  << " decode_gate=" << (decode.gate_passed ? "PASS" : "FAIL") << '\n';
    }
    return 0;
}

int Shadow(const Options& options) {
    if (options.database) {
        auto database = database::Database::Open(*options.database,
                                                  database::OpenMode::ExistingReadWrite);
        const auto profiles = database.LatestPipelineBenchmarks();
        std::vector<database::PipelineBenchmarkSummary> selected;
        for (const auto& profile : profiles) {
            if (profile.chunk_bytes != static_cast<std::int64_t>(options.block_size_bytes) ||
                profile.sample_count < 100 || profile.status != "SUCCESS") continue;
            const bool already = std::any_of(selected.begin(), selected.end(),
                [&](const auto& value) { return value.pipeline_type == profile.pipeline_type; });
            if (!already) selected.push_back(profile);
            if (selected.size() == 2) break;
        }
        if (selected.empty()) {
            throw std::runtime_error("no exact WU8 pipeline profile supports the requested block size");
        }
        std::ostringstream output;
        output << "{\"status\":\"OFFLINE_ONLY_NO_LIVE_MOVEMENT\""
               << ",\"lead_source\":\"CALLER_SUPPLIED_NOT_PUBLIC_CALLBACK_DEVICE_TIMING\""
               << ",\"block_size_bytes\":" << options.block_size_bytes << ",\"profiles\":[";
        bool first = true;
        for (const auto& profile : selected) {
            const auto pipeline_ns = static_cast<std::uint64_t>(profile.p99_ns);
            const auto causal = EvaluateShadowDeadline(KnowledgeMode::Causal,
                options.causal_lead_ns, options.oracle_lead_ns, pipeline_ns, true);
            const auto oracle = EvaluateShadowDeadline(KnowledgeMode::PerfectOracle,
                options.causal_lead_ns, options.oracle_lead_ns, pipeline_ns, true);
            if (!first) output << ',';
            first = false;
            output << "{\"pipeline_type\":\"" << profile.pipeline_type
                   << "\",\"wu8_session_id\":" << profile.session_id
                   << ",\"aggregate_bytes\":" << profile.aggregate_bytes
                   << ",\"buffer_depth\":" << profile.buffer_depth
                   << ",\"compute_type\":\"" << profile.compute_type
                   << "\",\"compute_window_us\":" << profile.compute_window_us
                   << ",\"contention_test\":\"" << profile.contention_test
                   << "\",\"phase\":\"" << profile.phase << '"'
                   << ",\"measured_p99_ns\":" << pipeline_ns
                   << ",\"sample_count\":" << profile.sample_count
                   << ",\"causal_hit\":" << (causal.hit ? "true" : "false")
                   << ",\"causal_margin_ns\":" << causal.margin_ns
                   << ",\"oracle_hit\":" << (oracle.hit ? "true" : "false")
                   << ",\"oracle_margin_ns\":" << oracle.margin_ns << '}';
            if (options.supplied_sha256) {
                if (const auto benchmark_id = database.LatestInferenceBenchmarkId(*options.supplied_sha256)) {
                    for (const auto& [name, result] :
                         std::initializer_list<std::pair<const char*, ShadowDeadlineResult>>{
                             {"CAUSAL", causal}, {"PERFECT_ORACLE", oracle}}) {
                        database::ShadowPipelineRecordInput record;
                        record.inference_benchmark_id = *benchmark_id;
                        record.block_size_bytes = static_cast<std::int64_t>(options.block_size_bytes);
                        record.pipeline_type = profile.pipeline_type;
                        record.lookahead_assumption =
                            "EXPLICIT_CALLER_SUPPLIED_LEAD_NOT_PUBLIC_CALLBACK_DEVICE_TIMING";
                        record.knowledge_mode = name;
                        record.available_lead_time_ns = static_cast<std::int64_t>(result.available_lead_time_ns);
                        record.pipeline_profile_reference = "WU8_SESSION_" + std::to_string(profile.session_id) + "_P99";
                        record.deadline_ns = static_cast<std::int64_t>(result.available_lead_time_ns);
                        record.predicted_hit = result.hit;
                        record.margin_ns = result.margin_ns;
                        record.extrapolated = true;
                        database.InsertShadowPipelineResult(record);
                    }
                }
            }
        }
        output << "]}";
        if (options.json) std::cout << output.str() << '\n';
        else std::cout << "shadow OFFLINE_ONLY_NO_LIVE_MOVEMENT exact WU8 profiles="
                       << selected.size() << " block_bytes=" << options.block_size_bytes << '\n';
        return 0;
    }
    const auto causal = EvaluateShadowDeadline(KnowledgeMode::Causal, options.causal_lead_ns,
                                                options.oracle_lead_ns, options.pipeline_ns);
    const auto oracle = EvaluateShadowDeadline(KnowledgeMode::PerfectOracle, options.causal_lead_ns,
                                                options.oracle_lead_ns, options.pipeline_ns);
    if (options.json) {
        std::cout << "{\"status\":\"OFFLINE_ONLY_NO_LIVE_MOVEMENT\",\"causal\":{\"lead_ns\":"
                  << causal.available_lead_time_ns << ",\"pipeline_ns\":" << causal.predicted_pipeline_ns
                  << ",\"hit\":" << (causal.hit ? "true" : "false") << ",\"margin_ns\":" << causal.margin_ns
                  << "},\"perfect_oracle\":{\"lead_ns\":" << oracle.available_lead_time_ns
                  << ",\"pipeline_ns\":" << oracle.predicted_pipeline_ns
                  << ",\"hit\":" << (oracle.hit ? "true" : "false") << ",\"margin_ns\":" << oracle.margin_ns << "}}\n";
    } else {
        std::cout << "shadow OFFLINE_ONLY_NO_LIVE_MOVEMENT causal=" << (causal.hit ? "HIT" : "MISS")
                  << " oracle=" << (oracle.hit ? "HIT" : "MISS") << '\n';
    }
    return 0;
}

int Report(const Options& options) {
    if (!options.database) throw std::invalid_argument("report requires --database path");
    auto database = database::Database::Open(*options.database, database::OpenMode::ExistingReadWrite);
    const auto counts = database.LlamaCounts();
    if (options.json) {
        std::cout << "{\"dependencies\":" << counts.dependencies << ",\"models\":" << counts.models
                  << ",\"tensors\":" << counts.tensors << ",\"storage_provenance\":" << counts.storage_provenance
                  << ",\"configurations\":" << counts.configurations
                  << ",\"benchmarks\":" << counts.benchmarks << ",\"token_samples\":" << counts.token_samples
                  << ",\"observer_events\":" << counts.observer_events
                  << ",\"tensor_demand_events\":" << counts.tensor_demand_events
                  << ",\"block_projections\":" << counts.block_projections
                  << ",\"shadow_results\":" << counts.shadow_results
                  << ",\"overhead_benchmarks\":" << counts.overhead_benchmarks
                  << ",\"moe_profiles\":" << counts.moe_profiles << "}\n";
    } else {
        std::cout << "llama dependencies=" << counts.dependencies << " models=" << counts.models
                  << " tensors=" << counts.tensors << " storage_provenance=" << counts.storage_provenance
                  << " benchmarks=" << counts.benchmarks
                  << " observer_events=" << counts.observer_events
                  << " tensor_demand_events=" << counts.tensor_demand_events
                  << " block_projections=" << counts.block_projections
                  << " shadow_results=" << counts.shadow_results
                  << " overhead_benchmarks=" << counts.overhead_benchmarks
                  << " moe_profiles=" << counts.moe_profiles << '\n';
    }
    return 0;
}

}  // namespace

int RunLlamaCli(int argc, char** argv) {
    if (argc < 2 || std::string_view(argv[1]) != "llama") return -1;
    if (argc < 3) { PrintLlamaUsage(); return 1; }
    const std::string_view command(argv[2]);
    if (command == "info") return Info(ParseOptions(argc, argv, 3));
    if (command == "model") {
        if (argc < 4) { PrintLlamaUsage(); return 1; }
        const std::string_view action(argv[3]);
        if (action != "inspect" && action != "verify") throw std::invalid_argument("unknown llama model action");
        return ModelCommand(action, ParseOptions(argc, argv, 4));
    }
    if (command == "baseline" || command == "observe" || command == "demand" || command == "validate")
        return InferenceCommand(command, ParseOptions(argc, argv, 3));
    if (command == "overhead") return Overhead(ParseOptions(argc, argv, 3));
    if (command == "shadow") return Shadow(ParseOptions(argc, argv, 3));
    if (command == "report") return Report(ParseOptions(argc, argv, 3));
    PrintLlamaUsage();
    return 1;
}

}  // namespace sidecar::llama
