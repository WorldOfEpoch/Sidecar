#pragma once

#include "sidecar/database/database.hpp"
#include "sidecar/hardware/hardware.hpp"

namespace sidecar::hardware {

void PersistDiscovery(database::Database& database, const DiscoveryReport& report);

}  // namespace sidecar::hardware

