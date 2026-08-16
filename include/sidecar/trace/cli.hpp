#pragma once

namespace sidecar::trace {

// Returns -1 when the command is not handled by the trace subsystem.
int RunTraceCommand(int argc, char** argv);

}  // namespace sidecar::trace
