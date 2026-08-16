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
