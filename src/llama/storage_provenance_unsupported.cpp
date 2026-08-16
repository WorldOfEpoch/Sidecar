#include "sidecar/llama/observation.hpp"

namespace sidecar::llama {

StorageProvenance ResolveModelStorage(const std::filesystem::path& path,
                                      std::string storage_role,
                                      std::string copy_relationship) {
    StorageProvenance result;
    result.requested_path = std::filesystem::absolute(path);
    result.resolved_path = std::filesystem::weakly_canonical(path);
    result.storage_role = std::move(storage_role);
    result.copy_relationship = std::move(copy_relationship);
    result.device_model = "UNKNOWN_UNSUPPORTED_PLATFORM";
    result.bus_type = "UNKNOWN";
    return result;
}

}  // namespace sidecar::llama
