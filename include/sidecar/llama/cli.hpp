#pragma once

namespace sidecar::llama {

// Returns -1 when argv does not name the llama command family.
int RunLlamaCli(int argc, char** argv);

}  // namespace sidecar::llama
