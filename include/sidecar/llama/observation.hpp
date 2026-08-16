#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sidecar::llama {

inline constexpr std::uint32_t kObservationFormatVersion = 1;
inline constexpr std::uint64_t kBlock32MiB = 32ULL * 1024ULL * 1024ULL;
inline constexpr std::uint64_t kBlock64MiB = 64ULL * 1024ULL * 1024ULL;
inline constexpr std::uint64_t kBlock128MiB = 128ULL * 1024ULL * 1024ULL;

enum class ObserverMode : std::uint8_t {
    None,
    CallbackNoop,
    LightLayer,
    LightLayerFlightRecorder,
    LightNode,
    ForensicSelected,
};

enum class KnowledgeMode : std::uint8_t { Causal, PerfectOracle };

[[nodiscard]] std::string_view ToString(ObserverMode mode) noexcept;
[[nodiscard]] std::optional<ObserverMode> ParseObserverMode(std::string_view text) noexcept;

struct TensorIndexEntry {
    std::uint32_t tensor_id{0};
    std::string name;
    std::string type;
    std::vector<std::int64_t> dimensions;
    std::uint64_t bytes{0};
    std::uint64_t file_offset{0};
    std::uint64_t file_span_bytes{0};
    std::uint32_t alignment_bytes{0};
    std::optional<std::int32_t> layer;
    std::optional<std::int32_t> expert;
    std::string role{"weight"};
    bool persistent_weight{true};
};

struct ModelProvenance {
    std::string source{"LOCAL_FILE"};
    std::optional<std::string> source_repository;
    std::optional<std::string> source_revision;
    std::optional<std::string> ollama_model_tag;
    std::optional<std::string> ollama_digest;
    std::optional<std::filesystem::path> original_blob_path;
    std::optional<std::filesystem::path> alias_path;
};

struct StorageProvenance {
    std::filesystem::path requested_path;
    std::filesystem::path resolved_path;
    std::string volume_path;
    std::string volume_unique_id;
    std::string physical_device;
    std::optional<std::uint32_t> physical_disk_number;
    std::string device_model;
    std::string bus_type;
    bool is_samsung_990_pro{false};
    bool is_usb_external{false};
    std::string storage_role;
    std::string copy_relationship;
};

struct ModelIndex {
    std::filesystem::path path;
    ModelProvenance provenance;
    std::optional<StorageProvenance> storage;
    std::string sha256;
    std::uint64_t file_size_bytes{0};
    std::uint32_t gguf_version{0};
    std::uint32_t alignment_bytes{0};
    std::uint64_t data_offset{0};
    std::string architecture;
    std::string quantization;
    std::optional<std::uint64_t> parameter_count;
    std::optional<std::uint32_t> layer_count;
    std::optional<std::uint32_t> expert_count;
    std::string model_name;
    std::string metadata_json{"{}"};
    std::string verification_status{"NOT_VERIFIED"};
    std::string verification_detail;
    std::string llama_cpp_commit;
    bool is_moe{false};
    std::vector<TensorIndexEntry> tensors;
};

// Exactly one cache-line half. Strings and dynamic memory are deliberately absent.
struct ObserverEvent32 {
    std::uint64_t timestamp_ns{0};
    std::uint32_t eval_index{0};
    std::uint32_t sequence_index{0};
    std::uint32_t tensor_id_0{0xFFFFFFFFU};
    std::uint32_t tensor_id_1{0xFFFFFFFFU};
    std::uint16_t op_id{0};
    std::int16_t layer{-1};
    std::uint8_t event_class{0};
    std::uint8_t flags{0};
    std::uint16_t reserved{0};
};
static_assert(sizeof(ObserverEvent32) == 32);

struct BlockProjection {
    std::uint64_t block_size_bytes{0};
    std::uint64_t useful_tensor_bytes{0};
    std::vector<std::uint64_t> block_ids;
    std::uint64_t projected_bytes{0};
    std::uint64_t overfetch_bytes{0};
    double overfetch_ratio{0.0};
};

struct LayerWorkingSet {
    std::int32_t layer{0};
    std::uint64_t tensor_count{0}, weight_bytes{0};
    std::uint64_t contiguous_span_bytes{0}, gap_bytes{0};
    double contiguity_ratio{0};
    BlockProjection blocks_32_mib, blocks_64_mib, blocks_128_mib;
};

struct MoeStaticProfile {
    std::string status{"NOT_MOE"};
    std::uint32_t expert_count{0};
    std::uint64_t router_always_hot_bytes{0};
    std::uint64_t packed_expert_tensor_bytes{0};
    std::uint64_t estimated_bytes_per_expert{0};
    std::uint64_t candidate_experts_in_capacity{0};
    bool selected_experts_observable{false};
};

struct ShadowDeadlineResult {
    KnowledgeMode knowledge{KnowledgeMode::Causal};
    std::uint64_t available_lead_time_ns{0};
    std::uint64_t predicted_pipeline_ns{0};
    bool hit{false};
    std::int64_t margin_ns{0};
    bool extrapolated{false};
};

struct PairedOverheadStatistics {
    std::size_t pairs{0};
    double baseline_mean_ns{0.0};
    double observed_mean_ns{0.0};
    double baseline_median_ns{0.0};
    double observed_median_ns{0.0};
    double paired_delta_mean_ns{0.0};
    double paired_delta_median_ns{0.0};
    double paired_mean_overhead_percent{0.0};
    double overhead_percent{0.0};
    double paired_p95_overhead_percent{0.0};
    double paired_variance_percent_squared{0.0};
    double paired_stddev_percent{0.0};
    double throughput_change_percent{0.0};
    double ci95_low_percent{0.0};
    double ci95_high_percent{0.0};
    bool gate_passed{false};
};

struct BlockReuseAnalysis {
    std::uint64_t block_size_bytes{0};
    std::size_t evaluation_count{0};
    std::size_t union_block_count{0};
    std::size_t permanently_hot_block_count{0};
    std::size_t new_after_first_block_count{0};
    double mean_repeated_block_fraction{0.0};
};

struct LayerLookaheadAnalysis {
    std::uint32_t layers_ahead{0};
    std::vector<std::uint64_t> lead_times_ns;
};

struct InferenceConfiguration {
    std::filesystem::path model_path;
    std::uint32_t prompt_tokens{128};
    std::uint32_t generated_tokens{32};
    std::uint32_t context_size{4096};
    std::int32_t gpu_layers{-1};
    std::uint32_t seed{0x53494445U};
    std::uint32_t warmups{1};
    std::uint32_t repetitions{1};
    ObserverMode observer_mode{ObserverMode::None};
    bool authoritative{false};
    std::size_t event_capacity{1U << 20U};
    std::optional<std::filesystem::path> flight_recorder_path;
    std::string machine_hash;
};

struct TokenTiming {
    std::uint32_t token_index{0};
    std::int32_t input_token_id{0};
    std::int32_t output_token_id{0};
    std::uint32_t context_depth{0};
    std::uint64_t decode_start_ns{0};
    std::uint64_t decode_end_ns{0};
    std::uint64_t decode_duration_ns{0};
    std::uint64_t sampling_ns{0};
    std::uint64_t graph_node_count{0};
};

struct InferenceSample {
    std::uint32_t sample_index{0};
    std::uint64_t prompt_ns{0};
    std::uint64_t initial_sampling_ns{0};
    std::uint64_t decode_total_ns{0};
    std::uint64_t total_request_ns{0};
    double prompt_tokens_per_second{0.0};
    double generation_tokens_per_second{0.0};
    std::uint64_t graph_node_count{0};
    std::uint64_t ask_calls{0};
    std::uint64_t materialized_calls{0};
    std::uint64_t dropped_events{0};
    std::vector<TokenTiming> tokens;
    std::vector<ObserverEvent32> events;
};

struct InferenceResult {
    std::string status{"SKIPPED_UNSUPPORTED"};
    std::string message;
    std::uint64_t model_load_ns{0};
    std::uint64_t prompt_tokenization_ns{0};
    std::string fixture_id;
    std::string fixture_text_sha256;
    std::vector<std::int32_t> prompt_token_ids;
    std::string callback_contract;
    std::string graph_split_detail{"BACKEND_DETAIL_UNAVAILABLE_PUBLIC_API"};
    std::string runtime_configuration_json{"{}"};
    std::optional<StorageProvenance> model_storage;
    std::vector<InferenceSample> samples;
};

[[nodiscard]] std::optional<std::int32_t> ParseLayerIndex(std::string_view name) noexcept;
[[nodiscard]] std::optional<std::int32_t> ParseExpertIndex(std::string_view name) noexcept;
[[nodiscard]] std::string ClassifyTensorRole(std::string_view name);
[[nodiscard]] bool IsMoeTensor(std::string_view name) noexcept;
[[nodiscard]] BlockProjection ProjectTensorBlocks(
    const std::vector<TensorIndexEntry>& tensors,
    const std::vector<std::uint32_t>& demanded_tensor_ids,
    std::uint64_t block_size_bytes);
[[nodiscard]] std::vector<LayerWorkingSet> AnalyzeLayerWorkingSets(
    const std::vector<TensorIndexEntry>& tensors);
[[nodiscard]] MoeStaticProfile AnalyzeMoeStaticProfile(
    const ModelIndex& model, std::uint64_t capacity_bytes);
[[nodiscard]] std::vector<std::uint32_t> DeduplicateDemand(
    const std::vector<ObserverEvent32>& events);
[[nodiscard]] ShadowDeadlineResult EvaluateShadowDeadline(
    KnowledgeMode knowledge, std::uint64_t causal_lead_time_ns,
    std::uint64_t oracle_lead_time_ns, std::uint64_t predicted_pipeline_ns,
    bool extrapolated = false) noexcept;
[[nodiscard]] PairedOverheadStatistics CalculatePairedOverhead(
    const std::vector<std::uint64_t>& baseline_ns,
    const std::vector<std::uint64_t>& observed_ns,
    double gate_percent = 2.0);
[[nodiscard]] BlockReuseAnalysis AnalyzeBlockReuse(
    const std::vector<ObserverEvent32>& events,
    const std::vector<TensorIndexEntry>& tensors,
    std::uint64_t block_size_bytes);
[[nodiscard]] LayerLookaheadAnalysis AnalyzeLayerLookahead(
    const std::vector<ObserverEvent32>& events,
    std::uint32_t layers_ahead);

[[nodiscard]] ModelIndex InspectGguf(const std::filesystem::path& path,
                                     const ModelProvenance& provenance,
                                     std::string llama_cpp_commit,
                                     bool compute_sha256 = true);
[[nodiscard]] StorageProvenance ResolveModelStorage(
    const std::filesystem::path& path, std::string storage_role = {},
    std::string copy_relationship = {});
[[nodiscard]] std::string StorageProvenanceToJson(const StorageProvenance& provenance);
[[nodiscard]] InferenceResult RunInference(const InferenceConfiguration& configuration,
                                           const ModelIndex* index = nullptr);
[[nodiscard]] std::string ModelIndexToJson(const ModelIndex& model,
                                           bool include_tensors = true);
[[nodiscard]] std::string FormatModelIndex(const ModelIndex& model,
                                           bool include_tensors = false);
[[nodiscard]] std::string InferenceResultToJson(const InferenceResult& result);

}  // namespace sidecar::llama
