#include "sidecar/llama/observation.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace sidecar::llama {
namespace {

std::string Lower(std::string_view text) {
    std::string result(text);
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return result;
}

std::optional<std::int32_t> ParseIndexAfter(std::string_view text,
                                            std::string_view marker) noexcept {
    const auto position = text.find(marker);
    if (position == std::string_view::npos) return std::nullopt;
    std::size_t begin = position + marker.size();
    std::size_t end = begin;
    while (end < text.size() && text[end] >= '0' && text[end] <= '9') ++end;
    if (end == begin) return std::nullopt;
    std::int64_t value = 0;
    for (std::size_t index = begin; index < end; ++index) {
        value = value * 10 + (text[index] - '0');
        if (value > (std::numeric_limits<std::int32_t>::max)()) return std::nullopt;
    }
    return static_cast<std::int32_t>(value);
}

std::string Escape(std::string_view text) {
    std::ostringstream output;
    for (const unsigned char value : text) {
        switch (value) {
            case '\"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (value < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned>(value) << std::dec;
                } else {
                    output << static_cast<char>(value);
                }
        }
    }
    return output.str();
}

template <typename T>
double Median(std::vector<T> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2U;
    if ((values.size() & 1U) != 0U) return static_cast<double>(values[middle]);
    return (static_cast<double>(values[middle - 1U]) +
            static_cast<double>(values[middle])) / 2.0;
}

double Quantile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = q * static_cast<double>(values.size() - 1U);
    const auto low = static_cast<std::size_t>(std::floor(position));
    const auto high = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(low);
    return values[low] + (values[high] - values[low]) * fraction;
}

}  // namespace

std::string_view ToString(ObserverMode mode) noexcept {
    switch (mode) {
        case ObserverMode::None: return "OBSERVER_NONE";
        case ObserverMode::CallbackNoop: return "CALLBACK_NOOP";
        case ObserverMode::LightLayer: return "LIGHT_LAYER";
        case ObserverMode::LightLayerFlightRecorder: return "LIGHT_LAYER_FLIGHT_RECORDER";
        case ObserverMode::LightNode: return "LIGHT_NODE";
        case ObserverMode::ForensicSelected: return "FORENSIC_SELECTED";
    }
    return "UNKNOWN";
}

std::optional<ObserverMode> ParseObserverMode(std::string_view text) noexcept {
    if (text == "none" || text == "OBSERVER_NONE") return ObserverMode::None;
    if (text == "noop" || text == "CALLBACK_NOOP") return ObserverMode::CallbackNoop;
    if (text == "light-layer" || text == "LIGHT_LAYER") return ObserverMode::LightLayer;
    if (text == "light-layer-fr" || text == "LIGHT_LAYER_FLIGHT_RECORDER")
        return ObserverMode::LightLayerFlightRecorder;
    if (text == "light-node" || text == "LIGHT_NODE") return ObserverMode::LightNode;
    if (text == "forensic" || text == "FORENSIC_SELECTED")
        return ObserverMode::ForensicSelected;
    return std::nullopt;
}

std::optional<std::int32_t> ParseLayerIndex(std::string_view name) noexcept {
    if (const auto layer = ParseIndexAfter(name, "blk.")) return layer;
    if (const auto layer = ParseIndexAfter(name, "layers.")) return layer;
    if (const auto layer = ParseIndexAfter(name, "layer.")) return layer;
    if (const auto layer = ParseIndexAfter(name, "block.")) return layer;
    return std::nullopt;
}

std::optional<std::int32_t> ParseExpertIndex(std::string_view name) noexcept {
    if (const auto expert = ParseIndexAfter(name, "experts.")) return expert;
    if (const auto expert = ParseIndexAfter(name, "expert.")) return expert;
    if (const auto expert = ParseIndexAfter(name, "exp.")) return expert;
    return std::nullopt;
}

bool IsMoeTensor(std::string_view name) noexcept {
    return name.find("expert") != std::string_view::npos ||
           name.find("exps") != std::string_view::npos ||
           name.find("moe") != std::string_view::npos ||
           name.find("router") != std::string_view::npos ||
           name.find("gate_inp") != std::string_view::npos;
}

std::string ClassifyTensorRole(std::string_view name) {
    const auto lower = Lower(name);
    if (lower.find("token_embd") != std::string::npos ||
        lower.find("embed_tokens") != std::string::npos) return "token_embedding";
    if (lower.find("output") != std::string::npos ||
        lower.find("lm_head") != std::string::npos) return "output";
    if (lower.find("attn") != std::string::npos ||
        lower.find("attention") != std::string::npos) return "attention_weight";
    if (lower.find("norm") != std::string::npos) return "normalization_weight";
    if (lower.find("router") != std::string::npos ||
        lower.find("gate_inp") != std::string::npos) return "moe_router_weight";
    if (IsMoeTensor(lower)) return "moe_expert_weight";
    if (lower.find("ffn") != std::string::npos ||
        lower.find("mlp") != std::string::npos) return "feed_forward_weight";
    return "weight";
}

BlockProjection ProjectTensorBlocks(const std::vector<TensorIndexEntry>& tensors,
                                    const std::vector<std::uint32_t>& demanded_tensor_ids,
                                    std::uint64_t block_size_bytes) {
    if (block_size_bytes == 0) throw std::invalid_argument("block size must be non-zero");
    std::set<std::uint64_t> blocks;
    std::set<std::uint32_t> unique_tensors;
    std::uint64_t useful = 0;
    for (const auto tensor_id : demanded_tensor_ids) {
        if (!unique_tensors.insert(tensor_id).second) continue;
        const auto match = std::find_if(tensors.begin(), tensors.end(),
            [tensor_id](const TensorIndexEntry& entry) { return entry.tensor_id == tensor_id; });
        if (match == tensors.end() || match->file_span_bytes == 0) continue;
        useful += match->bytes;
        const auto first = match->file_offset / block_size_bytes;
        const auto last = (match->file_offset + match->file_span_bytes - 1U) / block_size_bytes;
        for (auto block = first; block <= last; ++block) blocks.insert(block);
    }
    BlockProjection result;
    result.block_size_bytes = block_size_bytes;
    result.useful_tensor_bytes = useful;
    result.block_ids.assign(blocks.begin(), blocks.end());
    result.projected_bytes = static_cast<std::uint64_t>(blocks.size()) * block_size_bytes;
    result.overfetch_bytes = result.projected_bytes > useful ? result.projected_bytes - useful : 0;
    result.overfetch_ratio = useful == 0 ? 0.0 :
        static_cast<double>(result.overfetch_bytes) / static_cast<double>(useful);
    return result;
}

std::vector<LayerWorkingSet> AnalyzeLayerWorkingSets(
    const std::vector<TensorIndexEntry>& tensors) {
    std::map<std::int32_t, std::vector<std::uint32_t>> by_layer;
    for (const auto& tensor : tensors) {
        if (tensor.persistent_weight && tensor.layer) by_layer[*tensor.layer].push_back(tensor.tensor_id);
    }
    std::vector<LayerWorkingSet> result;
    result.reserve(by_layer.size());
    for (const auto& [layer, ids] : by_layer) {
        LayerWorkingSet working_set;
        working_set.layer = layer;
        working_set.tensor_count = ids.size();
        std::uint64_t first = (std::numeric_limits<std::uint64_t>::max)();
        std::uint64_t last = 0;
        for (const auto id : ids) {
            const auto iterator = std::find_if(tensors.begin(), tensors.end(),
                [id](const auto& tensor) { return tensor.tensor_id == id; });
            if (iterator == tensors.end()) continue;
            working_set.weight_bytes += iterator->bytes;
            first = (std::min)(first, iterator->file_offset);
            last = (std::max)(last, iterator->file_offset + iterator->file_span_bytes);
        }
        if (first != (std::numeric_limits<std::uint64_t>::max)() && last >= first)
            working_set.contiguous_span_bytes = last - first;
        working_set.gap_bytes = working_set.contiguous_span_bytes > working_set.weight_bytes
            ? working_set.contiguous_span_bytes - working_set.weight_bytes : 0;
        working_set.contiguity_ratio = working_set.contiguous_span_bytes == 0 ? 0.0 :
            static_cast<double>(working_set.weight_bytes) /
            static_cast<double>(working_set.contiguous_span_bytes);
        working_set.blocks_32_mib = ProjectTensorBlocks(tensors, ids, kBlock32MiB);
        working_set.blocks_64_mib = ProjectTensorBlocks(tensors, ids, kBlock64MiB);
        working_set.blocks_128_mib = ProjectTensorBlocks(tensors, ids, kBlock128MiB);
        result.push_back(std::move(working_set));
    }
    return result;
}

MoeStaticProfile AnalyzeMoeStaticProfile(const ModelIndex& model,
                                         std::uint64_t capacity_bytes) {
    MoeStaticProfile profile;
    if (!model.is_moe) return profile;
    profile.status = "STATIC_LAYOUT_ONLY_SELECTED_EXPERTS_NOT_OBSERVED";
    profile.expert_count = model.expert_count.value_or(0);
    for (const auto& tensor : model.tensors) {
        if (tensor.role == "moe_router_weight") profile.router_always_hot_bytes += tensor.bytes;
        else if (tensor.role == "moe_expert_weight") profile.packed_expert_tensor_bytes += tensor.bytes;
    }
    if (profile.expert_count != 0) {
        profile.estimated_bytes_per_expert =
            profile.packed_expert_tensor_bytes / profile.expert_count;
        if (profile.estimated_bytes_per_expert != 0)
            profile.candidate_experts_in_capacity = capacity_bytes /
                profile.estimated_bytes_per_expert;
    }
    return profile;
}

std::vector<std::uint32_t> DeduplicateDemand(const std::vector<ObserverEvent32>& events) {
    std::set<std::uint32_t> unique;
    for (const auto& event : events) {
        if (event.tensor_id_0 != 0xFFFFFFFFU) unique.insert(event.tensor_id_0);
        if (event.tensor_id_1 != 0xFFFFFFFFU) unique.insert(event.tensor_id_1);
    }
    return {unique.begin(), unique.end()};
}

ShadowDeadlineResult EvaluateShadowDeadline(KnowledgeMode knowledge,
                                            std::uint64_t causal_lead_time_ns,
                                            std::uint64_t oracle_lead_time_ns,
                                            std::uint64_t predicted_pipeline_ns,
                                            bool extrapolated) noexcept {
    ShadowDeadlineResult result;
    result.knowledge = knowledge;
    result.available_lead_time_ns = knowledge == KnowledgeMode::Causal
        ? causal_lead_time_ns : oracle_lead_time_ns;
    result.predicted_pipeline_ns = predicted_pipeline_ns;
    result.hit = result.available_lead_time_ns >= predicted_pipeline_ns;
    if (result.hit) {
        const auto margin = result.available_lead_time_ns - predicted_pipeline_ns;
        result.margin_ns = margin > static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)())
            ? (std::numeric_limits<std::int64_t>::max)() : static_cast<std::int64_t>(margin);
    } else {
        const auto deficit = predicted_pipeline_ns - result.available_lead_time_ns;
        result.margin_ns = deficit > static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)())
            ? (std::numeric_limits<std::int64_t>::min)() : -static_cast<std::int64_t>(deficit);
    }
    result.extrapolated = extrapolated;
    return result;
}

PairedOverheadStatistics CalculatePairedOverhead(
    const std::vector<std::uint64_t>& baseline_ns,
    const std::vector<std::uint64_t>& observed_ns,
    double gate_percent) {
    if (baseline_ns.empty() || baseline_ns.size() != observed_ns.size()) {
        throw std::invalid_argument("paired overhead requires equal non-empty samples");
    }
    std::vector<double> deltas;
    std::vector<double> percentages;
    deltas.reserve(baseline_ns.size());
    percentages.reserve(baseline_ns.size());
    for (std::size_t index = 0; index < baseline_ns.size(); ++index) {
        const double baseline = static_cast<double>(baseline_ns[index]);
        const double delta = static_cast<double>(observed_ns[index]) - baseline;
        deltas.push_back(delta);
        percentages.push_back(baseline == 0.0 ? 0.0 : delta * 100.0 / baseline);
    }
    PairedOverheadStatistics result;
    result.pairs = baseline_ns.size();
    const auto mean = [](const auto& values) {
        long double total = 0.0;
        for (const auto value : values) total += static_cast<long double>(value);
        return static_cast<double>(total / static_cast<long double>(values.size()));
    };
    result.baseline_mean_ns = mean(baseline_ns);
    result.observed_mean_ns = mean(observed_ns);
    result.baseline_median_ns = Median(baseline_ns);
    result.observed_median_ns = Median(observed_ns);
    result.paired_delta_mean_ns = mean(deltas);
    result.paired_delta_median_ns = Median(deltas);
    result.paired_mean_overhead_percent = mean(percentages);
    result.overhead_percent = Median(percentages);
    result.paired_p95_overhead_percent = Quantile(percentages, 0.95);
    long double squared_error = 0.0;
    for (const auto value : percentages) {
        const long double delta = static_cast<long double>(value) -
                                  result.paired_mean_overhead_percent;
        squared_error += delta * delta;
    }
    result.paired_variance_percent_squared = percentages.size() < 2 ? 0.0 :
        static_cast<double>(squared_error / static_cast<long double>(percentages.size() - 1U));
    result.paired_stddev_percent = std::sqrt(result.paired_variance_percent_squared);
    result.throughput_change_percent = result.observed_median_ns == 0.0 ? 0.0 :
        (result.baseline_median_ns / result.observed_median_ns - 1.0) * 100.0;
    result.ci95_low_percent = Quantile(percentages, 0.025);
    result.ci95_high_percent = Quantile(percentages, 0.975);
    result.gate_passed = result.overhead_percent <= gate_percent &&
                         result.ci95_high_percent <= gate_percent;
    return result;
}

BlockReuseAnalysis AnalyzeBlockReuse(
    const std::vector<ObserverEvent32>& events,
    const std::vector<TensorIndexEntry>& tensors,
    std::uint64_t block_size_bytes) {
    if (block_size_bytes == 0) throw std::invalid_argument("block size must be non-zero");
    std::map<std::uint32_t, std::vector<ObserverEvent32>> by_evaluation;
    for (const auto& event : events) {
        if (event.eval_index > 0) by_evaluation[event.eval_index].push_back(event);
    }
    std::vector<std::set<std::uint64_t>> block_sets;
    block_sets.reserve(by_evaluation.size());
    for (const auto& [evaluation, evaluation_events] : by_evaluation) {
        (void)evaluation;
        const auto projection = ProjectTensorBlocks(
            tensors, DeduplicateDemand(evaluation_events), block_size_bytes);
        block_sets.emplace_back(projection.block_ids.begin(), projection.block_ids.end());
    }

    BlockReuseAnalysis result;
    result.block_size_bytes = block_size_bytes;
    result.evaluation_count = block_sets.size();
    if (block_sets.empty()) return result;
    std::set<std::uint64_t> union_blocks = block_sets.front();
    std::set<std::uint64_t> hot_blocks = block_sets.front();
    long double repeated_fraction_total = 0.0;
    for (std::size_t index = 1; index < block_sets.size(); ++index) {
        const auto& current = block_sets[index];
        std::set<std::uint64_t> repeated;
        std::set_intersection(block_sets[index - 1].begin(), block_sets[index - 1].end(),
                              current.begin(), current.end(),
                              std::inserter(repeated, repeated.end()));
        if (!current.empty()) {
            repeated_fraction_total += static_cast<long double>(repeated.size()) /
                                       static_cast<long double>(current.size());
        }
        std::set<std::uint64_t> next_hot;
        std::set_intersection(hot_blocks.begin(), hot_blocks.end(),
                              current.begin(), current.end(),
                              std::inserter(next_hot, next_hot.end()));
        hot_blocks = std::move(next_hot);
        union_blocks.insert(current.begin(), current.end());
    }
    result.union_block_count = union_blocks.size();
    result.permanently_hot_block_count = hot_blocks.size();
    std::set<std::uint64_t> new_after_first;
    std::set_difference(union_blocks.begin(), union_blocks.end(),
                        block_sets.front().begin(), block_sets.front().end(),
                        std::inserter(new_after_first, new_after_first.end()));
    result.new_after_first_block_count = new_after_first.size();
    result.mean_repeated_block_fraction = block_sets.size() < 2 ? 0.0 :
        static_cast<double>(repeated_fraction_total /
                            static_cast<long double>(block_sets.size() - 1U));
    return result;
}

LayerLookaheadAnalysis AnalyzeLayerLookahead(
    const std::vector<ObserverEvent32>& events,
    std::uint32_t layers_ahead) {
    if (layers_ahead == 0) throw std::invalid_argument("layer lookahead must be positive");
    std::map<std::uint32_t, std::vector<const ObserverEvent32*>> by_evaluation;
    for (const auto& event : events) {
        if (event.layer >= 0) by_evaluation[event.eval_index].push_back(&event);
    }
    LayerLookaheadAnalysis result;
    result.layers_ahead = layers_ahead;
    for (auto& [evaluation, evaluation_events] : by_evaluation) {
        (void)evaluation;
        std::sort(evaluation_events.begin(), evaluation_events.end(), [](const auto* left, const auto* right) {
            return left->sequence_index < right->sequence_index;
        });
        std::vector<const ObserverEvent32*> first_layer_events;
        std::set<std::int16_t> seen_layers;
        for (const auto* event : evaluation_events) {
            if (seen_layers.insert(event->layer).second) first_layer_events.push_back(event);
        }
        for (std::size_t index = 0; index + layers_ahead < first_layer_events.size(); ++index) {
            const auto start = first_layer_events[index]->timestamp_ns;
            const auto deadline = first_layer_events[index + layers_ahead]->timestamp_ns;
            if (deadline >= start) result.lead_times_ns.push_back(deadline - start);
        }
    }
    return result;
}

std::string ModelIndexToJson(const ModelIndex& model, bool include_tensors) {
    std::ostringstream output;
    output << "{\"format_version\":" << kObservationFormatVersion
           << ",\"path\":\"" << Escape(model.path.string())
           << "\",\"storage\":";
    if (model.storage) output << StorageProvenanceToJson(*model.storage);
    else output << "null";
    output
           << ",\"source\":\"" << Escape(model.provenance.source)
           << "\",\"source_repository\":";
    if (model.provenance.source_repository) output << '"' << Escape(*model.provenance.source_repository) << '"';
    else output << "null";
    output << ",\"source_revision\":";
    if (model.provenance.source_revision) output << '"' << Escape(*model.provenance.source_revision) << '"';
    else output << "null";
    output << ",\"ollama_model_tag\":";
    if (model.provenance.ollama_model_tag) output << '"' << Escape(*model.provenance.ollama_model_tag) << '"';
    else output << "null";
    output << ",\"ollama_digest\":";
    if (model.provenance.ollama_digest) output << '"' << Escape(*model.provenance.ollama_digest) << '"';
    else output << "null";
    output << ",\"original_blob_path\":";
    if (model.provenance.original_blob_path) output << '"' << Escape(model.provenance.original_blob_path->string()) << '"';
    else output << "null";
    output << ",\"alias_path\":";
    if (model.provenance.alias_path) output << '"' << Escape(model.provenance.alias_path->string()) << '"';
    else output << "null";
    output << ",\"sha256\":\"" << model.sha256
           << "\",\"file_size_bytes\":" << model.file_size_bytes
           << ",\"gguf_version\":" << model.gguf_version
           << ",\"alignment_bytes\":" << model.alignment_bytes
           << ",\"data_offset\":" << model.data_offset
           << ",\"architecture\":\"" << Escape(model.architecture)
           << "\",\"quantization\":\"" << Escape(model.quantization)
           << "\",\"parameter_count\":";
    if (model.parameter_count) output << *model.parameter_count; else output << "null";
    output << ",\"layer_count\":";
    if (model.layer_count) output << *model.layer_count; else output << "null";
    output << ",\"expert_count\":";
    if (model.expert_count) output << *model.expert_count; else output << "null";
    output << ",\"model_name\":\"" << Escape(model.model_name)
           << "\",\"is_moe\":" << (model.is_moe ? "true" : "false")
           << ",\"tensor_count\":" << model.tensors.size()
           << ",\"verification_status\":\"" << Escape(model.verification_status)
           << "\",\"verification_detail\":\"" << Escape(model.verification_detail)
           << "\",\"llama_cpp_commit\":\"" << Escape(model.llama_cpp_commit)
           << "\",\"metadata\":" << model.metadata_json;
    std::vector<std::uint32_t> persistent_ids;
    std::uint64_t persistent_bytes = 0;
    for (const auto& tensor : model.tensors) {
        if (!tensor.persistent_weight) continue;
        persistent_ids.push_back(tensor.tensor_id);
        persistent_bytes += tensor.bytes;
    }
    const auto all32 = ProjectTensorBlocks(model.tensors, persistent_ids, kBlock32MiB);
    const auto all64 = ProjectTensorBlocks(model.tensors, persistent_ids, kBlock64MiB);
    const auto all128 = ProjectTensorBlocks(model.tensors, persistent_ids, kBlock128MiB);
    output << ",\"persistent_weight_summary\":{\"tensor_count\":" << persistent_ids.size()
           << ",\"bytes\":" << persistent_bytes
           << ",\"blocks_32_mib\":" << all32.block_ids.size()
           << ",\"overfetch_32_mib\":" << all32.overfetch_ratio
           << ",\"blocks_64_mib\":" << all64.block_ids.size()
           << ",\"overfetch_64_mib\":" << all64.overfetch_ratio
           << ",\"blocks_128_mib\":" << all128.block_ids.size()
           << ",\"overfetch_128_mib\":" << all128.overfetch_ratio << '}';
    const auto layers = AnalyzeLayerWorkingSets(model.tensors);
    output << ",\"layer_working_sets\":[";
    for (std::size_t index = 0; index < layers.size(); ++index) {
        if (index != 0) output << ',';
        const auto& layer = layers[index];
        output << "{\"layer\":" << layer.layer
               << ",\"tensor_count\":" << layer.tensor_count
               << ",\"weight_bytes\":" << layer.weight_bytes
               << ",\"contiguous_span_bytes\":" << layer.contiguous_span_bytes
               << ",\"gap_bytes\":" << layer.gap_bytes
               << ",\"contiguity_ratio\":" << layer.contiguity_ratio
               << ",\"blocks_32_mib\":" << layer.blocks_32_mib.block_ids.size()
               << ",\"overfetch_32_mib\":" << layer.blocks_32_mib.overfetch_ratio
               << ",\"blocks_64_mib\":" << layer.blocks_64_mib.block_ids.size()
               << ",\"overfetch_64_mib\":" << layer.blocks_64_mib.overfetch_ratio
               << ",\"blocks_128_mib\":" << layer.blocks_128_mib.block_ids.size()
               << ",\"overfetch_128_mib\":" << layer.blocks_128_mib.overfetch_ratio
               << ",\"block_ids_32_mib\":[";
        for (std::size_t block = 0; block < layer.blocks_32_mib.block_ids.size(); ++block) {
            if (block != 0) output << ',';
            output << layer.blocks_32_mib.block_ids[block];
        }
        output << "],\"block_ids_64_mib\":[";
        for (std::size_t block = 0; block < layer.blocks_64_mib.block_ids.size(); ++block) {
            if (block != 0) output << ',';
            output << layer.blocks_64_mib.block_ids[block];
        }
        output << "],\"block_ids_128_mib\":[";
        for (std::size_t block = 0; block < layer.blocks_128_mib.block_ids.size(); ++block) {
            if (block != 0) output << ',';
            output << layer.blocks_128_mib.block_ids[block];
        }
        output << "]}";
    }
    output << ']';
    if (model.is_moe) {
        const auto moe = AnalyzeMoeStaticProfile(model, 0);
        output << ",\"moe_static_profile\":{\"status\":\"" << Escape(moe.status)
               << "\",\"expert_count\":" << moe.expert_count
               << ",\"router_always_hot_bytes\":" << moe.router_always_hot_bytes
               << ",\"packed_expert_tensor_bytes\":" << moe.packed_expert_tensor_bytes
               << ",\"estimated_bytes_per_expert\":" << moe.estimated_bytes_per_expert
               << ",\"selected_experts_observable\":false}";
    }
    if (include_tensors) {
        output << ",\"tensors\":[";
        for (std::size_t index = 0; index < model.tensors.size(); ++index) {
            if (index != 0) output << ',';
            const auto& tensor = model.tensors[index];
            output << "{\"tensor_id\":" << tensor.tensor_id
                   << ",\"name\":\"" << Escape(tensor.name)
                   << "\",\"type\":\"" << Escape(tensor.type)
                   << "\",\"dimensions\":[";
            for (std::size_t dim = 0; dim < tensor.dimensions.size(); ++dim) {
                if (dim != 0) output << ',';
                output << tensor.dimensions[dim];
            }
            output << "],\"bytes\":" << tensor.bytes
                   << ",\"file_offset\":" << tensor.file_offset
                   << ",\"file_span_bytes\":" << tensor.file_span_bytes
                   << ",\"alignment_bytes\":" << tensor.alignment_bytes
                   << ",\"layer\":";
            if (tensor.layer) output << *tensor.layer; else output << "null";
            output << ",\"expert\":";
            if (tensor.expert) output << *tensor.expert; else output << "null";
            output << ",\"role\":\"" << Escape(tensor.role)
                   << "\",\"persistent_weight\":" << (tensor.persistent_weight ? "true" : "false")
                   << '}';
        }
        output << ']';
    }
    output << '}';
    return output.str();
}

std::string FormatModelIndex(const ModelIndex& model, bool include_tensors) {
    std::ostringstream output;
    output << "GGUF MODEL\n"
           << "Path: " << model.path.string() << '\n'
           << "Source: " << model.provenance.source << '\n';
    if (model.storage) {
        output << "Resolved path: " << model.storage->resolved_path.string() << '\n'
               << "Backing volume: " << model.storage->volume_path << '\n'
               << "Physical device: " << model.storage->physical_device << '\n'
               << "Device model: " << model.storage->device_model << '\n'
               << "Bus: " << model.storage->bus_type << '\n'
               << "Samsung 990 PRO: " << (model.storage->is_samsung_990_pro ? "yes" : "no") << '\n'
               << "USB/external: " << (model.storage->is_usb_external ? "yes" : "no") << '\n';
    }
    if (model.provenance.ollama_model_tag) output << "Ollama model: " << *model.provenance.ollama_model_tag << '\n';
    if (model.provenance.ollama_digest) output << "Ollama digest: " << *model.provenance.ollama_digest << '\n';
    output << "SHA-256: " << model.sha256 << '\n'
           << "Bytes: " << model.file_size_bytes << '\n'
           << "GGUF version: " << model.gguf_version << '\n'
           << "Architecture: " << model.architecture << '\n'
           << "Quantization: " << model.quantization << '\n'
           << "Parameters: " << (model.parameter_count ? std::to_string(*model.parameter_count) : "unknown") << '\n'
           << "Layers: " << (model.layer_count ? std::to_string(*model.layer_count) : "unknown") << '\n'
           << "Experts: " << (model.expert_count ? std::to_string(*model.expert_count) : "none/unknown") << '\n'
           << "Tensors: " << model.tensors.size() << '\n'
           << "Verification: " << model.verification_status << " (" << model.verification_detail << ")\n"
           << "llama.cpp: " << model.llama_cpp_commit << '\n';
    if (include_tensors) {
        for (const auto& tensor : model.tensors) {
            output << "  [" << tensor.tensor_id << "] " << tensor.name
                   << " type=" << tensor.type << " bytes=" << tensor.bytes
                   << " offset=" << tensor.file_offset;
            if (tensor.layer) output << " layer=" << *tensor.layer;
            if (tensor.expert) output << " expert=" << *tensor.expert;
            output << '\n';
        }
    }
    return output.str();
}

std::string StorageProvenanceToJson(const StorageProvenance& provenance) {
    std::ostringstream output;
    output << "{\"requested_path\":\"" << Escape(provenance.requested_path.string())
           << "\",\"resolved_path\":\"" << Escape(provenance.resolved_path.string())
           << "\",\"volume_path\":\"" << Escape(provenance.volume_path)
           << "\",\"volume_unique_id\":\"" << Escape(provenance.volume_unique_id)
           << "\",\"physical_device\":\"" << Escape(provenance.physical_device)
           << "\",\"physical_disk_number\":";
    if (provenance.physical_disk_number) output << *provenance.physical_disk_number;
    else output << "null";
    output << ",\"device_model\":\"" << Escape(provenance.device_model)
           << "\",\"bus_type\":\"" << Escape(provenance.bus_type)
           << "\",\"is_samsung_990_pro\":" << (provenance.is_samsung_990_pro ? "true" : "false")
           << ",\"is_usb_external\":" << (provenance.is_usb_external ? "true" : "false")
           << ",\"storage_role\":\"" << Escape(provenance.storage_role)
           << "\",\"copy_relationship\":\"" << Escape(provenance.copy_relationship) << "\"}";
    return output.str();
}

std::string InferenceResultToJson(const InferenceResult& result) {
    std::ostringstream output;
    output << "{\"format_version\":" << kObservationFormatVersion
           << ",\"status\":\"" << Escape(result.status)
           << "\",\"message\":\"" << Escape(result.message)
           << "\",\"model_load_ns\":" << result.model_load_ns
           << ",\"prompt_tokenization_ns\":" << result.prompt_tokenization_ns
           << ",\"fixture_id\":\"" << Escape(result.fixture_id)
           << "\",\"fixture_text_sha256\":\"" << Escape(result.fixture_text_sha256) << '"'
           << ",\"prompt_token_ids\":[";
    for (std::size_t token_index = 0; token_index < result.prompt_token_ids.size(); ++token_index) {
        if (token_index != 0) output << ',';
        output << result.prompt_token_ids[token_index];
    }
    output << ']'
           << ",\"model_storage\":";
    if (result.model_storage) output << StorageProvenanceToJson(*result.model_storage);
    else output << "null";
    output
           << ",\"callback_contract\":\"" << Escape(result.callback_contract)
           << "\",\"graph_split_detail\":\"" << Escape(result.graph_split_detail)
           << "\",\"runtime_configuration\":" << result.runtime_configuration_json
           << ",\"samples\":[";
    for (std::size_t sample_index = 0; sample_index < result.samples.size(); ++sample_index) {
        if (sample_index != 0) output << ',';
        const auto& sample = result.samples[sample_index];
        output << "{\"sample_index\":" << sample.sample_index
               << ",\"prompt_ns\":" << sample.prompt_ns
               << ",\"initial_sampling_ns\":" << sample.initial_sampling_ns
               << ",\"decode_total_ns\":" << sample.decode_total_ns
               << ",\"total_request_ns\":" << sample.total_request_ns
               << ",\"prompt_tokens_per_second\":" << sample.prompt_tokens_per_second
               << ",\"generation_tokens_per_second\":" << sample.generation_tokens_per_second
               << ",\"graph_node_count\":" << sample.graph_node_count
               << ",\"ask_calls\":" << sample.ask_calls
               << ",\"materialized_calls\":" << sample.materialized_calls
               << ",\"observer_event_count\":" << sample.events.size()
               << ",\"dropped_events\":" << sample.dropped_events
               << ",\"tokens\":[";
        for (std::size_t token_index = 0; token_index < sample.tokens.size(); ++token_index) {
            if (token_index != 0) output << ',';
            const auto& token = sample.tokens[token_index];
            output << "{\"token_index\":" << token.token_index
                   << ",\"input_token_id\":" << token.input_token_id
                   << ",\"output_token_id\":" << token.output_token_id
                   << ",\"context_depth\":" << token.context_depth
                   << ",\"decode_start_ns\":" << token.decode_start_ns
                   << ",\"decode_end_ns\":" << token.decode_end_ns
                   << ",\"decode_duration_ns\":" << token.decode_duration_ns
                   << ",\"sampling_ns\":" << token.sampling_ns
                   << ",\"graph_node_count\":" << token.graph_node_count << '}';
        }
        output << "],\"observer_events\":[";
        for (std::size_t event_index = 0; event_index < sample.events.size(); ++event_index) {
            if (event_index != 0) output << ',';
            const auto& event = sample.events[event_index];
            output << "{\"timestamp_ns\":" << event.timestamp_ns
                   << ",\"eval_index\":" << event.eval_index
                   << ",\"sequence_index\":" << event.sequence_index
                   << ",\"tensor_id_0\":" << event.tensor_id_0
                   << ",\"tensor_id_1\":" << event.tensor_id_1
                   << ",\"op_id\":" << event.op_id
                   << ",\"layer\":" << event.layer
                   << ",\"event_class\":" << static_cast<unsigned>(event.event_class)
                   << ",\"flags\":" << static_cast<unsigned>(event.flags) << '}';
        }
        output << "]}";
    }
    output << "]}";
    return output.str();
}

}  // namespace sidecar::llama
