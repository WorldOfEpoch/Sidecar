#pragma once

namespace sidecar::storage {

// Returns -1 when argv is not a storage command.
int RunStorageCli(int argc, char** argv);

}  // namespace sidecar::storage
