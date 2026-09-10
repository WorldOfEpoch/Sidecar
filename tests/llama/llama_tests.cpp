#include "sidecar/database/database.hpp"
#include "sidecar/llama/observation.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Expect(bool condition, const std::string& message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

std::filesystem::path Temporary(std::string_view name) {
    return std::filesystem::temp_directory_path() /
        ("sidecar-wu9-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
         "-" + std::string(name));
}

void TestLayerAndRole() {
    using namespace sidecar::llama;
    Expect(ParseLayerIndex("blk.27.attn_q.weight") == 27, "llama block layer parse");
    Expect(ParseLayerIndex("model.layers.4.mlp.experts.7.weight") == 4, "HF-style layer parse");
    Expect(ParseExpertIndex("model.layers.4.mlp.experts.7.weight") == 7, "expert parse");
    Expect(!ParseLayerIndex("token_embd.weight"), "global tensor has no layer");
    Expect(ClassifyTensorRole("blk.1.attn_q.weight") == "attention_weight", "attention role");
    Expect(ClassifyTensorRole("blk.1.ffn_gate_exps.weight") == "moe_expert_weight", "MoE role");
    Expect(IsMoeTensor("blk.1.ffn_gate_exps.weight"), "MoE classification");
}

void TestProjectionAndDemand() {
    using namespace sidecar::llama;
    std::vector<TensorIndexEntry> tensors{
        {0, "blk.0.a", "q5_K", {1}, 8, 0, 8, 32, 0, std::nullopt, "weight", true},
        {1, "blk.0.b", "q5_K", {1}, 16, 31, 16, 32, 0, std::nullopt, "weight", true},
        {2, "blk.1.c", "q5_K", {1}, 8, 128, 8, 32, 1, std::nullopt, "weight", true},
    };
    const auto projection = ProjectTensorBlocks(tensors, {0, 1, 1, 2}, 32);
    Expect(projection.block_ids == std::vector<std::uint64_t>({0, 1, 4}), "block projection exact IDs");
    Expect(projection.useful_tensor_bytes == 32, "deduplicated useful bytes");
    Expect(projection.projected_bytes == 96 && projection.overfetch_bytes == 64, "overfetch math");

    ObserverEvent32 a; a.tensor_id_0 = 3; a.tensor_id_1 = 7;
    ObserverEvent32 b; b.tensor_id_0 = 7; b.tensor_id_1 = 9;
    Expect(DeduplicateDemand({a, b}) == std::vector<std::uint32_t>({3, 7, 9}),
           "demand deduplication");
    const auto layers = AnalyzeLayerWorkingSets(tensors);
    Expect(layers.size() == 2 && layers[0].tensor_count == 2 &&
           layers[1].weight_bytes == 8, "dense per-layer working sets");
    Expect(sizeof(ObserverEvent32) == 32, "compact observer ABI");

    ObserverEvent32 e0; e0.eval_index = 1; e0.tensor_id_0 = 0; e0.tensor_id_1 = 1;
    ObserverEvent32 e1; e1.eval_index = 2; e1.tensor_id_0 = 0; e1.tensor_id_1 = 2;
    const auto reuse = AnalyzeBlockReuse({e0, e1}, tensors, 32);
    Expect(reuse.evaluation_count == 2 && reuse.union_block_count == 3 &&
           reuse.permanently_hot_block_count == 1 &&
           reuse.new_after_first_block_count == 1 &&
           reuse.mean_repeated_block_fraction == 0.5,
           "dense block reuse separates hot, repeated, and new demand");

    ObserverEvent32 l0; l0.eval_index = 1; l0.sequence_index = 0; l0.layer = 0; l0.timestamp_ns = 10;
    ObserverEvent32 l1; l1.eval_index = 1; l1.sequence_index = 1; l1.layer = 1; l1.timestamp_ns = 20;
    ObserverEvent32 l2; l2.eval_index = 1; l2.sequence_index = 2; l2.layer = 2; l2.timestamp_ns = 35;
    const auto one_ahead = AnalyzeLayerLookahead({l0, l1, l2}, 1);
    const auto two_ahead = AnalyzeLayerLookahead({l0, l1, l2}, 2);
    Expect(one_ahead.lead_times_ns == std::vector<std::uint64_t>({10, 15}) &&
           two_ahead.lead_times_ns == std::vector<std::uint64_t>({25}),
           "layer cadence math preserves within-evaluation host timestamps");

    ModelIndex moe;
    moe.is_moe = true;
    moe.expert_count = 4;
    TensorIndexEntry router; router.role = "moe_router_weight"; router.bytes = 16;
    TensorIndexEntry experts; experts.role = "moe_expert_weight"; experts.bytes = 400;
    moe.tensors = {router, experts};
    const auto moe_profile = AnalyzeMoeStaticProfile(moe, 250);
    Expect(moe_profile.router_always_hot_bytes == 16 &&
           moe_profile.estimated_bytes_per_expert == 100 &&
           moe_profile.candidate_experts_in_capacity == 2 &&
           !moe_profile.selected_experts_observable, "MoE static-only classification");

    ModelIndex dense;
    TensorIndexEntry dense_weights; dense_weights.bytes = 300; dense_weights.persistent_weight = true;
    dense.tensors = {dense_weights};
    OversizedFeasibilityInputs limited;
    limited.usable_vram_bytes = 100;
    limited.h2d_bytes_per_second = 100;
    limited.compute_window_ns = 1'000'000'000ULL;
    limited.staging_lead_time_ns = 3'000'000'000ULL;
    const auto dense_limited = AnalyzeOversizedFeasibility(dense, limited);
    Expect(dense_limited.minimum_nonresident_bytes == 200 &&
           dense_limited.candidate_h2d_bytes_per_token == 200 &&
           dense_limited.theoretical_h2d_floor_ns == 2'000'000'000ULL &&
           dense_limited.status == "DENSE_STATIC_BANDWIDTH_LIMITED",
           "dense feasibility identifies a recurring nonresident bandwidth floor");

    limited.h2d_bytes_per_second = 1'000;
    limited.compute_window_ns = 300'000'000ULL;
    limited.staging_lead_time_ns = 300'000'000ULL;
    const auto dense_plausible = AnalyzeOversizedFeasibility(dense, limited);
    Expect(dense_plausible.status == "DENSE_STATICALLY_PLAUSIBLE_NEEDS_RUNTIME_VALIDATION" &&
           dense_plausible.compute_window_margin_ns == 100'000'000LL,
           "dense feasibility keeps a positive result nonauthoritative");

    ModelIndex large_dense;
    TensorIndexEntry large_weights;
    large_weights.bytes = 50'000'000'000ULL;
    large_weights.persistent_weight = true;
    large_dense.tensors = {large_weights};
    OversizedFeasibilityInputs large_inputs;
    large_inputs.usable_vram_bytes = 25'000'000'000ULL;
    large_inputs.h2d_bytes_per_second = 25'000'000'000ULL;
    large_inputs.compute_window_ns = 64'000'000ULL;
    const auto large_result = AnalyzeOversizedFeasibility(large_dense, large_inputs);
    Expect(large_result.candidate_h2d_bytes_per_token == 25'000'000'000ULL &&
           large_result.theoretical_h2d_floor_ns == 1'000'000'000ULL &&
           large_result.status == "DENSE_STATIC_BANDWIDTH_LIMITED",
           "large dense transfer floor avoids 64-bit numerator overflow");

    moe.layer_count = 2;
    OversizedFeasibilityInputs moe_inputs;
    moe_inputs.usable_vram_bytes = 50;
    moe_inputs.h2d_bytes_per_second = 1'000;
    moe_inputs.assumed_active_experts_per_layer = 2;
    const auto moe_feasibility = AnalyzeOversizedFeasibility(moe, moe_inputs);
    Expect(moe_feasibility.requires_runtime_demand &&
           moe_feasibility.moe_expert_bytes_per_layer_estimate == 50 &&
           moe_feasibility.candidate_h2d_bytes_per_token == 200 &&
           moe_feasibility.status == "MOE_STATIC_LAYOUT_NEEDS_RUNTIME_DEMAND",
           "MoE feasibility never promotes static assumptions to demand evidence");
    ModelIndex plan_model = moe;
    plan_model.layer_count = 4;
    const auto plan = BuildConventionalBaselinePlan(plan_model);
    Expect(plan.size() == 6 && plan.front().gpu_layers == 0 &&
           plan[1].gpu_layers == 1 && plan[2].gpu_layers == 2 &&
           plan[3].gpu_layers == 3 && plan[4].gpu_layers == -1 &&
           plan[5].gpu_layers == -2,
           "conventional baseline plan includes CPU, partial, full-request, and max-stable controls");
}

void TestShadowAndOverhead() {
    using namespace sidecar::llama;
    const auto causal = EvaluateShadowDeadline(KnowledgeMode::Causal, 10, 100, 50);
    const auto oracle = EvaluateShadowDeadline(KnowledgeMode::PerfectOracle, 10, 100, 50);
    const auto qualified = EvaluateShadowDeadline(KnowledgeMode::Causal, 10, 100, 50, true);
    Expect(!causal.hit && causal.margin_ns == -40, "causal miss is not promoted to oracle");
    Expect(oracle.hit && oracle.margin_ns == 50, "perfect oracle hit");
    Expect(qualified.extrapolated, "shadow result preserves timing qualification");
    const auto overhead = CalculatePairedOverhead({100, 100, 100}, {101, 101, 101});
    Expect(overhead.pairs == 3 && overhead.overhead_percent == 1.0 &&
           overhead.paired_mean_overhead_percent == 1.0 &&
           overhead.paired_p95_overhead_percent == 1.0 &&
           overhead.throughput_change_percent < 0.0 && overhead.gate_passed,
           "paired overhead <2 percent gate");
    bool rejected = false;
    try { (void)CalculatePairedOverhead({1}, {1, 2}); } catch (const std::invalid_argument&) { rejected = true; }
    Expect(rejected, "unpaired overhead is rejected");
}

void TestMigrationAndPersistence() {
    const auto path = Temporary("migration.db");
    {
        auto database = sidecar::database::Database::Open(path, sidecar::database::OpenMode::CreateOrOpen);
        database.Initialize();
        const auto status = database.Status();
        Expect(status.schema_version == 9 && status.latest_migration == 9, "migration 009 status");
        Expect(status.foreign_key_violations == 0, "migration 009 foreign keys");
        sidecar::database::LlamaDependencyInput dependency;
        dependency.upstream_url = "https://github.com/ggml-org/llama.cpp.git";
        dependency.upstream_commit = "test-commit";
        dependency.build_configuration = "test";
        dependency.compiler = "test";
        dependency.sidecar_git_commit = "test";
        (void)database.UpsertLlamaDependency(dependency);
        const auto first = database.LlamaCounts();
        (void)database.UpsertLlamaDependency(dependency);
        const auto second = database.LlamaCounts();
        Expect(first.dependencies == 1 && second.dependencies == 1, "dependency upsert is idempotent");
    }
    std::filesystem::remove(path);
}

void TestJsonAndCorruptInput() {
    sidecar::llama::ModelIndex model;
    model.path = "fixture.gguf";
    model.sha256 = "abc";
    model.architecture = "llama";
    model.quantization = "Q5_K_M";
    model.metadata_json = "{}";
    const auto first = sidecar::llama::ModelIndexToJson(model);
    const auto second = sidecar::llama::ModelIndexToJson(model);
    Expect(first == second && first.find("\"architecture\":\"llama\"") != std::string::npos,
           "model JSON stability");

    sidecar::llama::InferenceResult inference;
    inference.status = "PASS";
    inference.prompt_token_ids = {1, 2};
    sidecar::llama::InferenceSample sample;
    sidecar::llama::ObserverEvent32 event;
    event.timestamp_ns = 11;
    event.eval_index = 3;
    event.layer = 7;
    sample.events.push_back(event);
    inference.samples.push_back(sample);
    const auto inference_first = sidecar::llama::InferenceResultToJson(inference);
    const auto inference_second = sidecar::llama::InferenceResultToJson(inference);
    Expect(inference_first == inference_second &&
           inference_first.find("\"prompt_token_ids\":[1,2]") != std::string::npos &&
           inference_first.find("\"observer_events\":[{\"timestamp_ns\":11") != std::string::npos,
           "inference JSON includes deterministic prompt and compact event evidence");
#if SIDECAR_LLAMA_ENABLED
    const auto corrupt = Temporary("corrupt.gguf");
    { std::ofstream output(corrupt, std::ios::binary); output << "not-a-gguf"; }
    bool rejected = false;
    try { (void)sidecar::llama::InspectGguf(corrupt, {}, "test", false); }
    catch (const std::exception&) { rejected = true; }
    Expect(rejected, "corrupt GGUF rejected");
    std::filesystem::remove(corrupt);
#endif
}

void TestStorageProvenance() {
    const auto path = Temporary("storage.bin");
    { std::ofstream output(path, std::ios::binary); output << "sidecar"; }
    const auto storage = sidecar::llama::ResolveModelStorage(
        path, "UNIT_TEST", "NO_COPY");
    Expect(!storage.resolved_path.empty(), "storage final path resolved");
    Expect(!storage.volume_path.empty(), "storage volume resolved");
#if defined(_WIN32)
    Expect(storage.physical_disk_number.has_value(), "storage physical disk mapped");
    Expect(!storage.physical_device.empty() && !storage.device_model.empty() &&
           storage.bus_type != "UNKNOWN", "storage device identity resolved");
#endif
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    TestLayerAndRole();
    TestProjectionAndDemand();
    TestShadowAndOverhead();
    TestMigrationAndPersistence();
    TestJsonAndCorruptInput();
    TestStorageProvenance();
    if (failures == 0) std::cout << "Sidecar WU9 llama/GGUF tests passed\n";
    return failures == 0 ? 0 : 1;
}
