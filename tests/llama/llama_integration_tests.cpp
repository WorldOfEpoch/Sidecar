#include "sidecar/llama/observation.hpp"
#include "sidecar/trace/format.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::string configured;
#if defined(_WIN32)
    char* value = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&value, &size, "SIDECAR_WU9_MODEL") == 0 && value != nullptr) {
        configured = value;
        std::free(value);
    }
#else
    if (const char* value = std::getenv("SIDECAR_WU9_MODEL")) configured = value;
#endif
    if (configured.empty() || !std::filesystem::is_regular_file(configured)) {
        std::cout << "SKIPPED: SIDECAR_WU9_MODEL is not configured\n";
        return 0;
    }
    sidecar::llama::ModelProvenance provenance;
    provenance.source = "TEST_CONFIGURED_LOCAL_FILE";
    auto model = sidecar::llama::InspectGguf(configured, provenance,
                                              SIDECAR_LLAMA_CPP_COMMIT, false);
    sidecar::llama::InferenceConfiguration configuration;
    configuration.model_path = configured;
    configuration.prompt_tokens = 8;
    configuration.generated_tokens = 2;
    configuration.context_size = 512;
    configuration.warmups = 0;
    configuration.repetitions = 1;
    std::vector<std::int32_t> reference_output_tokens;
    std::uint64_t noop_ask_calls = 0;
    for (const auto mode : {sidecar::llama::ObserverMode::None,
                            sidecar::llama::ObserverMode::CallbackNoop,
                            sidecar::llama::ObserverMode::LightLayer,
                            sidecar::llama::ObserverMode::LightNode,
                            sidecar::llama::ObserverMode::ForensicSelected}) {
        configuration.observer_mode = mode;
        const auto result = sidecar::llama::RunInference(configuration, &model);
        const bool callbacks_expected = mode != sidecar::llama::ObserverMode::None;
        const bool events_expected = mode == sidecar::llama::ObserverMode::LightLayer ||
                                     mode == sidecar::llama::ObserverMode::LightNode ||
                                     mode == sidecar::llama::ObserverMode::ForensicSelected;
        const bool materialized_expected = mode == sidecar::llama::ObserverMode::ForensicSelected;
        if (result.status != "PASS" || result.samples.size() != 1 ||
            result.samples.front().tokens.size() != 2 ||
            (callbacks_expected && result.samples.front().ask_calls == 0) ||
            (events_expected && result.samples.front().events.empty()) ||
            (materialized_expected && result.samples.front().materialized_calls == 0)) {
            std::cerr << "WU9 integration failed for mode "
                      << sidecar::llama::ToString(mode) << ": "
                      << result.status << ' ' << result.message << '\n';
            return 1;
        }
        std::vector<std::int32_t> output_tokens;
        for (const auto& token : result.samples.front().tokens) {
            output_tokens.push_back(token.output_token_id);
        }
        if (mode == sidecar::llama::ObserverMode::None) {
            reference_output_tokens = output_tokens;
        } else if (output_tokens != reference_output_tokens) {
            std::cerr << "WU9 observer changed deterministic output tokens for mode "
                      << sidecar::llama::ToString(mode) << '\n';
            return 1;
        }
        if (mode == sidecar::llama::ObserverMode::CallbackNoop) {
            noop_ask_calls = result.samples.front().ask_calls;
        } else if (mode == sidecar::llama::ObserverMode::ForensicSelected &&
                   result.samples.front().ask_calls != noop_ask_calls) {
            std::cerr << "WU9 forensic callback did not traverse the complete graph\n";
            return 1;
        }
    }
    const auto trace_path = std::filesystem::temp_directory_path() /
        ("sidecar-wu9-real-flight-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".sctrace");
    configuration.observer_mode = sidecar::llama::ObserverMode::LightLayerFlightRecorder;
    configuration.flight_recorder_path = trace_path;
    configuration.machine_hash = "integration-test-machine";
    const auto flight = sidecar::llama::RunInference(configuration, &model);
    if (flight.status != "PASS" || flight.samples.empty() ||
        flight.samples.front().events.empty() || !std::filesystem::is_regular_file(trace_path)) {
        std::cerr << "WU9 real-workload Flight Recorder integration failed\n";
        std::error_code ignored;
        std::filesystem::remove(trace_path, ignored);
        return 1;
    }
    const auto parsed_trace = sidecar::trace::TraceReader::Read(trace_path, true);
    std::error_code ignored;
    std::filesystem::remove(trace_path, ignored);
    if (parsed_trace.records32.empty() ||
        parsed_trace.header.machine_hash != "integration-test-machine") {
        std::cerr << "WU9 compact trace integrity validation failed\n";
        return 1;
    }
    std::cout << "WU9 pinned llama.cpp CUDA callback integration passed\n";
    return 0;
}
