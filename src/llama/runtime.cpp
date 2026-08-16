#include "sidecar/llama/observation.hpp"

#include "sidecar/core/sha256.hpp"
#include "sidecar/trace/recorder.hpp"
#include "sidecar/version.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if SIDECAR_LLAMA_ENABLED
#include <ggml-backend.h>
#include <ggml.h>
#include <llama.h>
#endif

namespace sidecar::llama {

#if SIDECAR_LLAMA_ENABLED && SIDECAR_CUDA_ENABLED
namespace {

using Clock = std::chrono::steady_clock;
using Recorder = trace::FlightRecorder<trace::TraceRecord32>;

std::uint64_t NowNs() noexcept {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count());
}

std::string EscapeJson(std::string_view text) {
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

struct DeviceMemory {
    ggml_backend_dev_t device{nullptr};
    std::size_t free_bytes{0};
    std::size_t total_bytes{0};
};

struct LogCapture {
    std::mutex mutex;
    std::string text;
};

void CaptureLog(ggml_log_level, const char* message, void* user_data) noexcept {
    if (message == nullptr || user_data == nullptr) return;
    auto& capture = *static_cast<LogCapture*>(user_data);
    try {
        std::lock_guard lock(capture.mutex);
        capture.text += message;
    } catch (...) {
    }
}

void DiscardLog(ggml_log_level, const char*, void*) noexcept {}

struct LogGuard {
    ggml_log_callback previous{nullptr};
    void* previous_user_data{nullptr};
    explicit LogGuard(LogCapture& capture) {
        llama_log_get(&previous, &previous_user_data);
        llama_log_set(CaptureLog, &capture);
    }
    void Quiet() noexcept { llama_log_set(DiscardLog, nullptr); }
    ~LogGuard() { llama_log_set(previous, previous_user_data); }
};

std::optional<std::uint64_t> ParseUnsignedAfter(std::string_view text,
                                                std::string_view marker) noexcept {
    const auto marker_position = text.rfind(marker);
    if (marker_position == std::string_view::npos) return std::nullopt;
    const auto begin = text.find_first_of("0123456789", marker_position + marker.size());
    if (begin == std::string_view::npos) return std::nullopt;
    std::uint64_t value = 0;
    for (std::size_t index = begin; index < text.size() && text[index] >= '0' && text[index] <= '9'; ++index) {
        const auto digit = static_cast<std::uint64_t>(text[index] - '0');
        if (value > ((std::numeric_limits<std::uint64_t>::max)() - digit) / 10U)
            return std::nullopt;
        value = value * 10U + digit;
    }
    return value;
}

std::optional<double> ParseDoubleAfter(std::string_view text,
                                       std::string_view marker) {
    const auto marker_position = text.rfind(marker);
    if (marker_position == std::string_view::npos) return std::nullopt;
    const auto begin = text.find_first_of("0123456789.", marker_position + marker.size());
    if (begin == std::string_view::npos) return std::nullopt;
    std::size_t end = begin;
    while (end < text.size() && ((text[end] >= '0' && text[end] <= '9') || text[end] == '.')) ++end;
    try {
        return std::stod(std::string(text.substr(begin, end - begin)));
    } catch (...) {
        return std::nullopt;
    }
}

DeviceMemory ReadGpuMemory(ggml_backend_dev_t device = nullptr) noexcept {
    DeviceMemory result;
    result.device = device != nullptr ? device :
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (result.device != nullptr)
        ggml_backend_dev_memory(result.device, &result.free_bytes, &result.total_bytes);
    return result;
}

struct TensorDictionaryEntry {
    std::string name;
    std::uint32_t id{0xFFFFFFFFU};
    std::int16_t layer{-1};
};

struct ObserverState {
    ObserverMode mode{ObserverMode::None};
    const std::vector<TensorDictionaryEntry>* dictionary{nullptr};
    std::vector<ObserverEvent32>* storage{nullptr};
    std::uint64_t event_count{0};
    std::uint64_t dropped{0};
    std::uint64_t ask_calls{0};
    std::uint64_t materialized_calls{0};
    std::uint64_t graph_nodes{0};
    std::uint32_t eval_index{0};
    std::uint32_t sequence{0};
    std::int16_t last_layer{-2};
    bool forensic_requested{false};
    Recorder::ProducerHandle* flight_handle{nullptr};
};

const TensorDictionaryEntry* FindTensor(const ObserverState& state, const char* name) noexcept {
    if (name == nullptr || name[0] == '\0' || state.dictionary == nullptr) return nullptr;
    const auto& entries = *state.dictionary;
    auto iterator = std::lower_bound(entries.begin(), entries.end(), name,
        [](const TensorDictionaryEntry& entry, const char* value) {
            return entry.name.compare(value) < 0;
        });
    return iterator != entries.end() && iterator->name == name ? &*iterator : nullptr;
}

void Emit(ObserverState& state, const ggml_tensor* tensor, bool materialized) noexcept {
    ObserverEvent32 event;
    event.timestamp_ns = NowNs();
    event.eval_index = state.eval_index;
    event.sequence_index = state.sequence++;
    event.op_id = static_cast<std::uint16_t>(tensor->op);
    event.event_class = materialized ? 2U : 1U;
    event.flags = materialized ? 1U : 0U;
    std::uint32_t* destinations[2]{&event.tensor_id_0, &event.tensor_id_1};
    std::size_t found = 0;
    auto consider = [&](const ggml_tensor* candidate) noexcept {
        if (candidate == nullptr || found == 2) return;
        if (const auto* match = FindTensor(state, candidate->name)) {
            *destinations[found++] = match->id;
            if (event.layer < 0) event.layer = match->layer;
        }
    };
    consider(tensor);
    for (std::size_t index = 0; index < GGML_MAX_SRC && found < 2; ++index)
        consider(tensor->src[index]);

    if (state.storage != nullptr && state.event_count < state.storage->size()) {
        (*state.storage)[static_cast<std::size_t>(state.event_count)] = event;
    } else {
        ++state.dropped;
    }
    ++state.event_count;

    if (state.flight_handle != nullptr) {
        trace::TraceRecord32 compact;
        compact.host_timestamp_ns = event.timestamp_ns;
        compact.operation_id = event.sequence_index;
        compact.parent_operation_id = event.eval_index;
        compact.subject_id = event.tensor_id_0;
        compact.payload_bytes = 0;
        compact.event_type = trace::EventType::HostScheduleDecision;
        compact.source_tier = trace::MemoryTier::VramL1;
        compact.destination_tier = trace::MemoryTier::Unknown;
        compact.producer_id = static_cast<std::uint16_t>(state.flight_handle->producerId());
        compact.auxiliary = event.op_id;
        if (!state.flight_handle->tryPush(compact)) ++state.dropped;
    }
}

bool EvalCallback(ggml_tensor* tensor, bool ask, void* user_data) noexcept {
    auto& state = *static_cast<ObserverState*>(user_data);
    if (ask) {
        ++state.ask_calls;
        ++state.graph_nodes;
        if (state.mode == ObserverMode::CallbackNoop) return false;
        if (state.mode == ObserverMode::LightNode) {
            Emit(state, tensor, false);
            return false;
        }
        if (state.mode == ObserverMode::LightLayer ||
            state.mode == ObserverMode::LightLayerFlightRecorder) {
            std::int16_t layer = -1;
            for (std::size_t index = 0; index < GGML_MAX_SRC; ++index) {
                const auto* source = tensor->src[index];
                if (source != nullptr) {
                    if (const auto* match = FindTensor(state, source->name)) {
                        if (match->layer >= 0) { layer = match->layer; break; }
                    }
                }
            }
            if (layer != state.last_layer && layer >= 0) {
                state.last_layer = layer;
                Emit(state, tensor, false);
            }
            return false;
        }
        if (state.mode == ObserverMode::ForensicSelected &&
            !state.forensic_requested && tensor->op == GGML_OP_MUL_MAT) {
            state.forensic_requested = true;
            return true;
        }
        return false;
    }
    ++state.materialized_calls;
    if (state.mode == ObserverMode::ForensicSelected) Emit(state, tensor, true);
    // ask == false is the observation delivery callback. Returning false here
    // cancels the remaining graph; true is required to continue evaluation.
    return true;
}

std::vector<TensorDictionaryEntry> MakeDictionary(const ModelIndex* index) {
    std::vector<TensorDictionaryEntry> dictionary;
    if (index != nullptr) {
        dictionary.reserve(index->tensors.size());
        for (const auto& tensor : index->tensors) {
            dictionary.push_back({tensor.name, tensor.tensor_id,
                static_cast<std::int16_t>(tensor.layer.value_or(-1))});
        }
        std::sort(dictionary.begin(), dictionary.end(),
                  [](const auto& left, const auto& right) { return left.name < right.name; });
    }
    return dictionary;
}

std::vector<llama_token> DeterministicPrompt(const llama_vocab* vocab,
                                             std::uint32_t target_tokens) {
    const std::string unit =
        "Sidecar observes deterministic transformer execution and records compact "
        "weight-demand metadata without changing model state. ";
    std::string text;
    text.reserve(static_cast<std::size_t>(target_tokens) * 12U + 256U);
    for (std::uint32_t index = 0; index < target_tokens / 4U + 16U; ++index) text += unit;
    int count = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()),
                               nullptr, 0, true, true);
    if (count == (std::numeric_limits<int>::min)())
        throw std::runtime_error("token count overflow");
    if (count < 0) count = -count;
    std::vector<llama_token> tokens(static_cast<std::size_t>(count));
    const int actual = llama_tokenize(vocab, text.data(), static_cast<int>(text.size()),
                                      tokens.data(), count, true, true);
    if (actual < 0) throw std::runtime_error("deterministic fixture tokenization failed");
    tokens.resize(static_cast<std::size_t>(actual));
    if (tokens.size() < target_tokens) throw std::runtime_error("fixture did not yield enough tokens");
    tokens.resize(target_tokens);
    return tokens;
}

struct ModelDeleter { void operator()(llama_model* value) const noexcept { llama_model_free(value); } };
struct ContextDeleter { void operator()(llama_context* value) const noexcept { llama_free(value); } };
struct SamplerDeleter { void operator()(llama_sampler* value) const noexcept { llama_sampler_free(value); } };

void ResetObserver(ObserverState& state, std::uint32_t eval_index) noexcept {
    state.eval_index = eval_index;
    state.last_layer = -2;
    state.forensic_requested = false;
}

void ResetObserverRun(ObserverState& state) noexcept {
    state.event_count = 0;
    state.dropped = 0;
    state.ask_calls = 0;
    state.materialized_calls = 0;
    state.graph_nodes = 0;
    state.sequence = 0;
    ResetObserver(state, 0);
}

}  // namespace
#endif

InferenceResult RunInference(const InferenceConfiguration& configuration,
                             const ModelIndex* index) {
    InferenceResult result;
#if !SIDECAR_LLAMA_ENABLED
    (void)configuration; (void)index;
    result.message = "SIDECAR_ENABLE_LLAMA=OFF";
    return result;
#elif !SIDECAR_CUDA_ENABLED
    (void)configuration; (void)index;
    result.message = "SKIPPED_UNSUPPORTED: CUDA-disabled build supports GGUF inspection only";
    return result;
#else
    if (!std::filesystem::is_regular_file(configuration.model_path)) {
        result.status = "FAILED";
        result.message = "model path is not a regular file";
        return result;
    }
    if (configuration.prompt_tokens == 0 || configuration.generated_tokens == 0 ||
        configuration.repetitions == 0 ||
        configuration.context_size < configuration.prompt_tokens + configuration.generated_tokens) {
        result.status = "FAILED";
        result.message = "invalid prompt/generation/context dimensions";
        return result;
    }
    try {
        result.model_storage = ResolveModelStorage(
            configuration.model_path,
            configuration.authoritative ? "PRIMARY_AUTHORITATIVE_DENSE" :
                                          "NONAUTHORITATIVE_INFERENCE_MODEL_LOAD",
            configuration.authoritative ? "AUTHORITY_STORAGE_GATE_PASSED" :
                                          "DIRECT_MODEL_PATH");
        LogCapture log_capture;
        LogGuard log_guard(log_capture);
        static std::once_flag backend_once;
        std::call_once(backend_once, [] {
            llama_backend_init();
            ggml_backend_load_all();
        });

        const auto gpu_memory_before_model = ReadGpuMemory();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = configuration.gpu_layers;
        model_params.check_tensors = true;
        const auto load_start = Clock::now();
        std::unique_ptr<llama_model, ModelDeleter> model(
            llama_model_load_from_file(configuration.model_path.string().c_str(), model_params));
        const auto load_end = Clock::now();
        result.model_load_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(load_end - load_start).count());
        if (!model) throw std::runtime_error("pinned llama.cpp rejected model load");
        if (!llama_model_has_decoder(model.get()))
            throw std::runtime_error("model has no supported decoder");
        const auto gpu_memory_after_model = ReadGpuMemory(gpu_memory_before_model.device);

        const auto* vocab = llama_model_get_vocab(model.get());
        const auto tokenize_start = Clock::now();
        auto prompt = DeterministicPrompt(vocab, configuration.prompt_tokens);
        const auto tokenize_end = Clock::now();
        result.prompt_tokenization_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(tokenize_end - tokenize_start).count());
        result.fixture_id = "WU9-DETERMINISTIC-V1-" + std::to_string(configuration.prompt_tokens);
        result.fixture_text_sha256 = core::Sha256Hex(
            "SIDECAR-WU9-DETERMINISTIC-PROMPT-V1:" + std::to_string(configuration.prompt_tokens));
        result.prompt_token_ids.assign(prompt.begin(), prompt.end());
        auto dictionary = MakeDictionary(index);
        std::vector<ObserverEvent32> event_storage(configuration.event_capacity);

        std::unique_ptr<Recorder> recorder;
        std::optional<Recorder::ProducerHandle> flight_handle;
        if (configuration.observer_mode == ObserverMode::LightLayerFlightRecorder) {
            if (!configuration.flight_recorder_path)
                throw std::runtime_error("LIGHT_LAYER_FLIGHT_RECORDER requires a trace path");
            const auto parent = configuration.flight_recorder_path->parent_path();
            if (!parent.empty()) std::filesystem::create_directories(parent);
            recorder = std::make_unique<Recorder>();
            flight_handle.emplace(recorder->RegisterProducer("llama-callback", "host"));
            trace::TraceHeader header;
            const auto version = CurrentVersionInfo();
            header.sidecar_version = version.version;
            header.spec_version = version.spec_version;
            header.git_commit = version.git_commit;
            header.machine_hash = configuration.machine_hash.empty()
                ? "UNKNOWN_NOT_DISCOVERED" : configuration.machine_hash;
            header.session_id = "WU9";
            header.trace_mode = std::string(ToString(configuration.observer_mode));
            header.monotonic_start_ns = NowNs();
            recorder->Start(*configuration.flight_recorder_path, std::move(header));
        }

        result.samples.reserve(configuration.repetitions);
        for (std::uint32_t run = 0; run < configuration.repetitions; ++run) {
            ObserverState observer;
            observer.mode = configuration.observer_mode;
            observer.dictionary = &dictionary;

            auto context_params = llama_context_default_params();
            context_params.n_ctx = configuration.context_size;
            context_params.n_batch = (std::max)(configuration.prompt_tokens, 512U);
            context_params.n_ubatch = (std::min)(context_params.n_batch, 512U);
            context_params.no_perf = false;
            if (configuration.observer_mode != ObserverMode::None) {
                context_params.cb_eval = EvalCallback;
                context_params.cb_eval_user_data = &observer;
            }
            std::unique_ptr<llama_context, ContextDeleter> context(
                llama_init_from_model(model.get(), context_params));
            if (!context) throw std::runtime_error("llama context creation failed");
            if (result.runtime_configuration_json == "{}") {
                const auto gpu_memory_after_context = ReadGpuMemory(gpu_memory_before_model.device);
                log_guard.Quiet();
                const auto gpu_layers_offloaded = ParseUnsignedAfter(log_capture.text, "offloaded ");
                std::optional<std::uint64_t> gpu_layers_total;
                if (const auto offload_position = log_capture.text.rfind("offloaded ");
                    offload_position != std::string::npos) {
                    const auto line_end = log_capture.text.find('\n', offload_position);
                    gpu_layers_total = ParseUnsignedAfter(
                        std::string_view(log_capture.text).substr(
                            offload_position, line_end == std::string::npos ?
                                std::string::npos : line_end - offload_position), "/");
                }
                const auto cpu_model_mib = ParseDoubleAfter(log_capture.text, "CPU_Mapped model buffer size =");
                const auto cuda_model_mib = ParseDoubleAfter(log_capture.text, "CUDA0 model buffer size =");
                const auto cuda_kv_mib = ParseDoubleAfter(log_capture.text, "CUDA0 KV buffer size =");
                const auto cuda_compute_mib = ParseDoubleAfter(log_capture.text, "CUDA0 compute buffer size =");
                const auto host_compute_mib = ParseDoubleAfter(log_capture.text, "CUDA_Host compute buffer size =");
                const auto graph_splits = ParseUnsignedAfter(log_capture.text, "graph splits =");
                if (graph_splits) {
                    result.graph_split_detail = "PINNED_LLAMA_INITIALIZATION_LOG_REPORTED_" +
                        std::to_string(*graph_splits) +
                        "_SPLITS;BACKEND_TRANSITION_DETAIL_UNAVAILABLE_PUBLIC_CALLBACK_API";
                }
                std::ostringstream runtime;
                runtime << "{\"n_gpu_layers_requested\":" << configuration.gpu_layers
                        << ",\"actual_weight_placement\":\"";
                if (gpu_layers_offloaded && gpu_layers_total &&
                    *gpu_layers_offloaded == *gpu_layers_total) {
                    runtime << "FULL_GPU_OFFLOAD_REPORTED_BY_PINNED_LLAMA_LOG";
                } else if (gpu_layers_offloaded) {
                    runtime << "PARTIAL_GPU_OFFLOAD_REPORTED_BY_PINNED_LLAMA_LOG";
                } else {
                    runtime << "NOT_QUERYABLE_PUBLIC_API";
                }
                runtime << '"' << ",\"gpu_layers_offloaded_reported\":";
                if (gpu_layers_offloaded) runtime << *gpu_layers_offloaded; else runtime << "null";
                runtime << ",\"gpu_layers_total_reported\":";
                if (gpu_layers_total) runtime << *gpu_layers_total; else runtime << "null";
                runtime
                        << ",\"model_bytes_reported\":" << llama_model_size(model.get())
                        << ",\"model_parameters_reported\":" << llama_model_n_params(model.get())
                        << ",\"load_mode\":\"" << EscapeJson(llama_load_mode_name(model_params.load_mode)) << '"'
                        << ",\"mmap_default_requested\":true"
                        << ",\"context_size_actual\":" << llama_n_ctx(context.get())
                        << ",\"batch_actual\":" << llama_n_batch(context.get())
                        << ",\"ubatch_actual\":" << llama_n_ubatch(context.get())
                        << ",\"threads_actual\":" << llama_n_threads(context.get())
                        << ",\"batch_threads_actual\":" << llama_n_threads_batch(context.get())
                        << ",\"flash_attention_request\":\""
                        << EscapeJson(llama_flash_attn_type_name(context_params.flash_attn_type)) << '"'
                        << ",\"flash_attention_actual\":\"NOT_QUERYABLE_PUBLIC_API\""
                        << ",\"kv_type_k_request\":\"" << ggml_type_name(context_params.type_k) << '"'
                        << ",\"kv_type_v_request\":\"" << ggml_type_name(context_params.type_v) << '"'
                        << ",\"offload_kqv_requested\":" << (context_params.offload_kqv ? "true" : "false")
                        << ",\"op_offload_requested\":" << (context_params.op_offload ? "true" : "false")
                        << ",\"flash_attention_enabled_reported\":"
                        << (log_capture.text.find("Flash Attention enabled") != std::string::npos ? "true" : "false")
                        << ",\"graph_splits_reported\":";
                if (graph_splits) runtime << *graph_splits; else runtime << "null";
                const auto write_mib = [&](const char* name, const std::optional<double>& value) {
                    runtime << ",\"" << name << "\":";
                    if (value) runtime << *value; else runtime << "null";
                };
                write_mib("cpu_mapped_model_buffer_mib", cpu_model_mib);
                write_mib("cuda_model_buffer_mib", cuda_model_mib);
                write_mib("cuda_kv_buffer_mib", cuda_kv_mib);
                write_mib("cuda_compute_buffer_mib", cuda_compute_mib);
                write_mib("cuda_host_compute_buffer_mib", host_compute_mib);
                runtime << ",\"captured_initialization_log_sha256\":\""
                        << core::Sha256Hex(log_capture.text) << '"'
                        << ",\"gpu_device\":";
                if (gpu_memory_before_model.device != nullptr) {
                    runtime << "{\"name\":\""
                            << EscapeJson(ggml_backend_dev_name(gpu_memory_before_model.device))
                            << "\",\"description\":\""
                            << EscapeJson(ggml_backend_dev_description(gpu_memory_before_model.device))
                            << "\",\"memory_source\":\"ggml_backend_dev_memory\""
                            << ",\"total_bytes\":" << gpu_memory_before_model.total_bytes
                            << ",\"free_before_model_bytes\":" << gpu_memory_before_model.free_bytes
                            << ",\"free_after_model_bytes\":" << gpu_memory_after_model.free_bytes
                            << ",\"free_after_context_bytes\":" << gpu_memory_after_context.free_bytes
                            << ",\"model_load_free_delta_bytes\":"
                            << (gpu_memory_before_model.free_bytes >= gpu_memory_after_model.free_bytes
                                ? gpu_memory_before_model.free_bytes - gpu_memory_after_model.free_bytes : 0)
                            << ",\"model_plus_context_free_delta_bytes\":"
                            << (gpu_memory_before_model.free_bytes >= gpu_memory_after_context.free_bytes
                                ? gpu_memory_before_model.free_bytes - gpu_memory_after_context.free_bytes : 0)
                            << '}';
                } else {
                    runtime << "null";
                }
                runtime << '}';
                result.runtime_configuration_json = runtime.str();
            }
            auto sampler_params = llama_sampler_chain_default_params();
            sampler_params.no_perf = false;
            std::unique_ptr<llama_sampler, SamplerDeleter> sampler(
                llama_sampler_chain_init(sampler_params));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());

            for (std::uint32_t pass = 0; pass <= configuration.warmups; ++pass) {
                const bool measured = pass == configuration.warmups;
                if (pass != 0) {
                    llama_memory_clear(llama_get_memory(context.get()), true);
                    llama_sampler_reset(sampler.get());
                }
                ResetObserverRun(observer);
                observer.storage = measured ? &event_storage : nullptr;
                observer.flight_handle = measured && flight_handle ? &*flight_handle : nullptr;

                InferenceSample sample;
                sample.sample_index = run;
            const auto request_start = Clock::now();
            ResetObserver(observer, 0);
            const auto prompt_start = Clock::now();
            auto batch = llama_batch_get_one(prompt.data(), static_cast<int32_t>(prompt.size()));
            const int prompt_code = llama_decode(context.get(), batch);
            llama_synchronize(context.get());
            const auto prompt_end = Clock::now();
            if (prompt_code != 0) throw std::runtime_error("prompt decode failed: " + std::to_string(prompt_code));
            sample.prompt_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(prompt_end - prompt_start).count());
            sample.prompt_tokens_per_second = sample.prompt_ns == 0 ? 0.0 :
                static_cast<double>(prompt.size()) * 1.0e9 / static_cast<double>(sample.prompt_ns);

            const auto initial_sampling_start = Clock::now();
            llama_token input_token = llama_sampler_sample(sampler.get(), context.get(), -1);
            const auto initial_sampling_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now() - initial_sampling_start).count());
            sample.initial_sampling_ns = initial_sampling_ns;
            sample.tokens.reserve(configuration.generated_tokens);
            for (std::uint32_t token_index = 0; token_index < configuration.generated_tokens; ++token_index) {
                ResetObserver(observer, token_index + 1U);
                TokenTiming timing;
                timing.token_index = token_index;
                timing.input_token_id = input_token;
                timing.context_depth = configuration.prompt_tokens + token_index;
                const auto graph_nodes_before = observer.graph_nodes;
                timing.decode_start_ns = NowNs();
                auto token_batch = llama_batch_get_one(&input_token, 1);
                const int decode_code = llama_decode(context.get(), token_batch);
                llama_synchronize(context.get());
                timing.decode_end_ns = NowNs();
                timing.decode_duration_ns = timing.decode_end_ns - timing.decode_start_ns;
                if (decode_code != 0) throw std::runtime_error("token decode failed: " + std::to_string(decode_code));
                const auto sampling_start = Clock::now();
                const llama_token output_token = llama_sampler_sample(sampler.get(), context.get(), -1);
                timing.sampling_ns = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now() - sampling_start).count());
                timing.graph_node_count = observer.graph_nodes - graph_nodes_before;
                timing.output_token_id = output_token;
                sample.decode_total_ns += timing.decode_duration_ns;
                sample.tokens.push_back(timing);
                input_token = output_token;
            }
            sample.generation_tokens_per_second = sample.decode_total_ns == 0 ? 0.0 :
                static_cast<double>(configuration.generated_tokens) * 1.0e9 /
                static_cast<double>(sample.decode_total_ns);
            sample.total_request_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    Clock::now() - request_start).count());
            sample.graph_node_count = observer.graph_nodes;
            sample.ask_calls = observer.ask_calls;
            sample.materialized_calls = observer.materialized_calls;
            sample.dropped_events = observer.dropped;
            const std::size_t captured = static_cast<std::size_t>((std::min)(
                observer.event_count, static_cast<std::uint64_t>(event_storage.size())));
            sample.events.assign(event_storage.begin(), event_storage.begin() + captured);
                if (measured) result.samples.push_back(std::move(sample));
            }
        }
        if (recorder) {
            const auto metrics = recorder->Stop();
            if (!result.samples.empty()) result.samples.back().dropped_events += metrics.records_dropped;
        }
        result.status = "PASS";
        result.message = "pinned llama.cpp model load, tensor validation, tokenization, prompt decode and deterministic generation passed";
        result.callback_contract = configuration.observer_mode == ObserverMode::None
            ? "NO_CALLBACK_INSTALLED"
            : "ask=true metadata only; LIGHT returns false; FORENSIC_SELECTED requests one materialization per eval; ask=false delivery returns true to continue graph";
        return result;
    } catch (const std::exception& error) {
        result.status = "FAILED";
        result.message = error.what();
        return result;
    }
#endif
}

}  // namespace sidecar::llama
