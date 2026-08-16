#include "sidecar/database/database.hpp"

#include "sidecar/database/schema.hpp"
#include "sidecar/database/migrations.hpp"

#include <sqlite3.h>

#include <utility>

namespace sidecar::database {
namespace {

class Statement final {
public:
    Statement(sqlite3* database, const char* sql) : database_(database) {
        const int result = sqlite3_prepare_v2(database_, sql, -1, &statement_, nullptr);
        if (result != SQLITE_OK) {
            throw DatabaseError(result, sqlite3_errmsg(database_));
        }
    }

    ~Statement() {
        if (statement_ != nullptr) {
            sqlite3_finalize(statement_);
        }
    }

    Statement(const Statement&) = delete;
    Statement& operator=(const Statement&) = delete;

    [[nodiscard]] sqlite3_stmt* get() const noexcept { return statement_; }

private:
    sqlite3* database_;
    sqlite3_stmt* statement_{nullptr};
};

void CheckResult(sqlite3* database, int result) {
    if (result != SQLITE_OK) {
        throw DatabaseError(result, sqlite3_errmsg(database));
    }
}

void CheckDone(sqlite3* database, int result) {
    if (result != SQLITE_DONE) {
        throw DatabaseError(result, sqlite3_errmsg(database));
    }
}

void BindText(sqlite3* database, sqlite3_stmt* statement, int index, const std::string& value) {
    CheckResult(database,
                sqlite3_bind_text(statement, index, value.c_str(),
                                  static_cast<int>(value.size()), SQLITE_TRANSIENT));
}

void BindOptionalText(sqlite3* database,
                      sqlite3_stmt* statement,
                      int index,
                      const std::optional<std::string>& value) {
    if (value.has_value()) {
        BindText(database, statement, index, *value);
    } else {
        CheckResult(database, sqlite3_bind_null(statement, index));
    }
}

void BindOptionalInt64(sqlite3* database,
                       sqlite3_stmt* statement,
                       int index,
                       const std::optional<std::int64_t>& value) {
    if (value.has_value()) {
        CheckResult(database, sqlite3_bind_int64(statement, index, *value));
    } else {
        CheckResult(database, sqlite3_bind_null(statement, index));
    }
}

void BindOptionalDouble(sqlite3* database,
                        sqlite3_stmt* statement,
                        int index,
                        const std::optional<double>& value) {
    if (value.has_value()) {
        CheckResult(database, sqlite3_bind_double(statement, index, *value));
    } else {
        CheckResult(database, sqlite3_bind_null(statement, index));
    }
}

}  // namespace

DatabaseError::DatabaseError(int sqlite_code, const std::string& message)
    : std::runtime_error(message), sqlite_code_(sqlite_code) {}

int DatabaseError::sqliteCode() const noexcept {
    return sqlite_code_;
}

Database Database::Open(const std::filesystem::path& path, OpenMode mode) {
    sqlite3* handle = nullptr;
    int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX;
    if (mode == OpenMode::CreateOrOpen) {
        flags |= SQLITE_OPEN_CREATE;
    }

    const std::string path_utf8 = path.generic_string();
    const int result = sqlite3_open_v2(path_utf8.c_str(), &handle, flags, nullptr);
    if (result != SQLITE_OK) {
        const std::string message = handle != nullptr ? sqlite3_errmsg(handle) : "SQLite open failed";
        if (handle != nullptr) {
            sqlite3_close(handle);
        }
        throw DatabaseError(result, message);
    }

    sqlite3_extended_result_codes(handle, 1);
    sqlite3_busy_timeout(handle, 5000);

    Database database(handle, path);
    database.Execute("PRAGMA foreign_keys = ON;");
    return database;
}

Database::Database(sqlite3* handle, std::filesystem::path path) noexcept
    : handle_(handle), path_(std::move(path)) {}

Database::~Database() {
    if (handle_ != nullptr) {
        sqlite3_close(handle_);
    }
}

Database::Database(Database&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)), path_(std::move(other.path_)) {}

Database& Database::operator=(Database&& other) noexcept {
    if (this != &other) {
        if (handle_ != nullptr) {
            sqlite3_close(handle_);
        }
        handle_ = std::exchange(other.handle_, nullptr);
        path_ = std::move(other.path_);
    }
    return *this;
}

void Database::Initialize() {
    const auto application_id = QueryInt64("PRAGMA application_id;");
    if (application_id != 0 && application_id != kApplicationId) {
        throw DatabaseError(SQLITE_MISMATCH,
                            "database belongs to a different application");
    }

    const auto existing_version = QueryInt64("PRAGMA user_version;");
    if (existing_version > kCurrentSchemaVersion) {
        throw DatabaseError(SQLITE_MISMATCH,
                            "database schema is newer than this Sidecar build");
    }
    if (existing_version == kCurrentSchemaVersion) {
        if (!TableExists("schema_migrations") ||
            QueryInt64("SELECT COALESCE(MAX(version), 0) FROM schema_migrations;") !=
                kCurrentSchemaVersion) {
            throw DatabaseError(SQLITE_CORRUPT,
                                "database migration metadata is inconsistent");
        }
        return;
    }

    Execute("PRAGMA journal_mode = WAL;");
    Execute("PRAGMA synchronous = NORMAL;");
    Execute("BEGIN IMMEDIATE;");
    try {
        auto migration_version = existing_version;
        if (migration_version < 1) {
            Execute(std::string(kSchemaSql));
            migration_version = 1;
        }
        if (migration_version < 2) {
            Execute(std::string(kMigration2Sql));
            migration_version = 2;
        }
        if (migration_version < 3) {
            Execute(std::string(kMigration3Sql));
            migration_version = 3;
        }
        if (migration_version < 4) {
            Execute(std::string(kMigration4Sql));
            migration_version = 4;
        }
        if (migration_version < 5) {
            Execute(std::string(kMigration5Sql));
            migration_version = 5;
        }
        if (migration_version < 6) {
            Execute(std::string(kMigration6Sql));
            migration_version = 6;
        }
        if (migration_version < 7) {
            Execute(std::string(kMigration7Sql));
            migration_version = 7;
        }
        if (migration_version < 8) {
            Execute(std::string(kMigration8Sql));
            migration_version = 8;
        }
        if (migration_version < 9) {
            Execute(std::string(kMigration9Sql));
            migration_version = 9;
        }
        Execute("COMMIT;");
    } catch (...) {
        sqlite3_exec(handle_, "ROLLBACK;", nullptr, nullptr, nullptr);
        throw;
    }

    const auto migrated_version = QueryInt64("PRAGMA user_version;");
    if (migrated_version != kCurrentSchemaVersion) {
        throw DatabaseError(SQLITE_CORRUPT,
                            "database migration did not reach the expected version");
    }
}

DatabaseStatus Database::Status() const {
    DatabaseStatus status;
    status.path = path_;
    status.foreign_keys_enabled = QueryInt64("PRAGMA foreign_keys;") == 1;
    status.application_id = QueryInt64("PRAGMA application_id;");
    status.schema_version = QueryInt64("PRAGMA user_version;");
    status.user_table_count = QueryInt64(
        "SELECT COUNT(*) FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%';");
    status.foreign_key_violations =
        QueryInt64("SELECT COUNT(*) FROM pragma_foreign_key_check;");
    status.initialized = TableExists("schema_migrations");
    if (status.initialized) {
        status.latest_migration = QueryInt64(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations;");
    }
    return status;
}

void Database::UpsertHardwareProfile(const HardwareProfileInput& profile) {
    static constexpr const char* kSql = R"sql(
INSERT INTO hardware_profiles(
    machine_hash, host_name, os_name, os_version, os_build, cpu_model,
    physical_core_count, logical_core_count, numa_node_count,
    installed_ram_bytes, available_ram_bytes, discovery_json,
    identity_version, identity_quality, identity_basis,
    system_manufacturer, system_product, smbios_version,
    cpu_vendor, cpu_architecture, processor_group_count, cpu_package_count,
    ram_speed_mt_s, motherboard_manufacturer, motherboard_model,
    firmware_revision, topology_json, snapshot_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(machine_hash) DO UPDATE SET
    discovered_at = CURRENT_TIMESTAMP,
    host_name = excluded.host_name,
    os_name = excluded.os_name,
    os_version = excluded.os_version,
    os_build = excluded.os_build,
    cpu_model = excluded.cpu_model,
    physical_core_count = excluded.physical_core_count,
    logical_core_count = excluded.logical_core_count,
    numa_node_count = excluded.numa_node_count,
    installed_ram_bytes = excluded.installed_ram_bytes,
    available_ram_bytes = excluded.available_ram_bytes,
    discovery_json = excluded.discovery_json,
    identity_version = excluded.identity_version,
    identity_quality = excluded.identity_quality,
    identity_basis = excluded.identity_basis,
    system_manufacturer = excluded.system_manufacturer,
    system_product = excluded.system_product,
    smbios_version = excluded.smbios_version,
    cpu_vendor = excluded.cpu_vendor,
    cpu_architecture = excluded.cpu_architecture,
    processor_group_count = excluded.processor_group_count,
    cpu_package_count = excluded.cpu_package_count,
    ram_speed_mt_s = excluded.ram_speed_mt_s,
    motherboard_manufacturer = excluded.motherboard_manufacturer,
    motherboard_model = excluded.motherboard_model,
    firmware_revision = excluded.firmware_revision,
    topology_json = excluded.topology_json,
    snapshot_json = excluded.snapshot_json;
)sql";

    Statement statement(handle_, kSql);
    sqlite3_stmt* raw = statement.get();
    BindText(handle_, raw, 1, profile.machine_hash);
    BindText(handle_, raw, 2, profile.host_name);
    BindText(handle_, raw, 3, profile.os_name);
    BindText(handle_, raw, 4, profile.os_version);
    BindText(handle_, raw, 5, profile.os_build);
    BindText(handle_, raw, 6, profile.cpu_model);
    BindOptionalInt64(handle_, raw, 7, profile.physical_core_count);
    BindOptionalInt64(handle_, raw, 8, profile.logical_core_count);
    BindOptionalInt64(handle_, raw, 9, profile.numa_node_count);
    BindOptionalInt64(handle_, raw, 10, profile.installed_ram_bytes);
    BindOptionalInt64(handle_, raw, 11, profile.available_ram_bytes);
    BindText(handle_, raw, 12, profile.discovery_json);
    CheckResult(handle_, sqlite3_bind_int64(raw, 13, profile.identity_version));
    BindText(handle_, raw, 14, profile.identity_quality);
    BindText(handle_, raw, 15, profile.identity_basis_json);
    BindText(handle_, raw, 16, profile.system_manufacturer);
    BindText(handle_, raw, 17, profile.system_product);
    BindText(handle_, raw, 18, profile.smbios_version);
    BindText(handle_, raw, 19, profile.cpu_vendor);
    BindText(handle_, raw, 20, profile.cpu_architecture);
    BindOptionalInt64(handle_, raw, 21, profile.processor_group_count);
    BindOptionalInt64(handle_, raw, 22, profile.cpu_package_count);
    BindOptionalInt64(handle_, raw, 23, profile.ram_speed_mt_s);
    BindText(handle_, raw, 24, profile.motherboard_manufacturer);
    BindText(handle_, raw, 25, profile.motherboard_model);
    BindText(handle_, raw, 26, profile.firmware_revision);
    BindText(handle_, raw, 27, profile.topology_json);
    BindText(handle_, raw, 28, profile.snapshot_json);
    CheckDone(handle_, sqlite3_step(raw));
}

void Database::RefreshHardwareInventory(
    const HardwareProfileInput& profile,
    const std::vector<GpuDeviceInput>& gpus,
    const std::vector<StorageDeviceInput>& storage_devices,
    bool refresh_gpu_presence,
    bool refresh_storage_presence) {
    Execute("BEGIN IMMEDIATE;");
    try {
        UpsertHardwareProfile(profile);

        if (refresh_gpu_presence) {
            Statement statement(handle_,
                                "UPDATE gpu_devices SET present = 0 "
                                "WHERE machine_hash = ?;");
            BindText(handle_, statement.get(), 1, profile.machine_hash);
            CheckDone(handle_, sqlite3_step(statement.get()));
        }
        if (refresh_storage_presence) {
            Statement statement(handle_,
                                "UPDATE storage_devices SET present = 0 "
                                "WHERE machine_hash = ?;");
            BindText(handle_, statement.get(), 1, profile.machine_hash);
            CheckDone(handle_, sqlite3_step(statement.get()));
        }

        static constexpr const char* kGpuSql = R"sql(
INSERT INTO gpu_devices(
    machine_hash, gpu_uuid, model, cuda_device_index, vram_bytes,
    pci_domain, pci_bus, pci_device, negotiated_pcie_generation,
    maximum_pcie_generation, negotiated_lane_width, maximum_lane_width,
    cuda_compute_capability_major, cuda_compute_capability_minor,
    async_engine_count, concurrent_kernels, unified_addressing,
    can_map_host_memory, managed_memory, pageable_memory_access,
    attributes_json, location_paths_json, present, last_seen_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
ON CONFLICT(machine_hash, gpu_uuid) DO UPDATE SET
    model = excluded.model,
    cuda_device_index = excluded.cuda_device_index,
    vram_bytes = excluded.vram_bytes,
    pci_domain = excluded.pci_domain,
    pci_bus = excluded.pci_bus,
    pci_device = excluded.pci_device,
    negotiated_pcie_generation = excluded.negotiated_pcie_generation,
    maximum_pcie_generation = excluded.maximum_pcie_generation,
    negotiated_lane_width = excluded.negotiated_lane_width,
    maximum_lane_width = excluded.maximum_lane_width,
    cuda_compute_capability_major = excluded.cuda_compute_capability_major,
    cuda_compute_capability_minor = excluded.cuda_compute_capability_minor,
    async_engine_count = excluded.async_engine_count,
    concurrent_kernels = excluded.concurrent_kernels,
    unified_addressing = excluded.unified_addressing,
    can_map_host_memory = excluded.can_map_host_memory,
    managed_memory = excluded.managed_memory,
    pageable_memory_access = excluded.pageable_memory_access,
    attributes_json = excluded.attributes_json,
    location_paths_json = excluded.location_paths_json,
    present = 1,
    last_seen_at = CURRENT_TIMESTAMP;
)sql";
        if (refresh_gpu_presence) for (const auto& gpu : gpus) {
            Statement statement(handle_, kGpuSql);
            sqlite3_stmt* raw = statement.get();
            BindText(handle_, raw, 1, profile.machine_hash);
            BindText(handle_, raw, 2, gpu.persistent_id);
            BindText(handle_, raw, 3, gpu.model);
            BindOptionalInt64(handle_, raw, 4, gpu.cuda_device_index);
            BindOptionalInt64(handle_, raw, 5, gpu.vram_bytes);
            BindOptionalInt64(handle_, raw, 6, gpu.pci_domain);
            BindOptionalInt64(handle_, raw, 7, gpu.pci_bus);
            BindOptionalInt64(handle_, raw, 8, gpu.pci_device);
            BindOptionalInt64(handle_, raw, 9, gpu.negotiated_pcie_generation);
            BindOptionalInt64(handle_, raw, 10, gpu.maximum_pcie_generation);
            BindOptionalInt64(handle_, raw, 11, gpu.negotiated_lane_width);
            BindOptionalInt64(handle_, raw, 12, gpu.maximum_lane_width);
            BindOptionalInt64(handle_, raw, 13, gpu.compute_capability_major);
            BindOptionalInt64(handle_, raw, 14, gpu.compute_capability_minor);
            BindOptionalInt64(handle_, raw, 15, gpu.async_engine_count);
            BindOptionalInt64(handle_, raw, 16, gpu.concurrent_kernels);
            BindOptionalInt64(handle_, raw, 17, gpu.unified_addressing);
            BindOptionalInt64(handle_, raw, 18, gpu.can_map_host_memory);
            BindOptionalInt64(handle_, raw, 19, gpu.managed_memory);
            BindOptionalInt64(handle_, raw, 20, gpu.pageable_memory_access);
            BindText(handle_, raw, 21, gpu.attributes_json);
            BindText(handle_, raw, 22, gpu.location_paths_json);
            CheckDone(handle_, sqlite3_step(raw));
        }

        static constexpr const char* kStorageSql = R"sql(
INSERT INTO storage_devices(
    machine_hash, persistent_id, model, firmware, capacity_bytes,
    filesystem, bus_type, pci_path, numa_node, topology_json,
    volumes_json, topology_confidence, topology_source, present, last_seen_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP)
ON CONFLICT(machine_hash, persistent_id) DO UPDATE SET
    model = excluded.model,
    firmware = excluded.firmware,
    capacity_bytes = excluded.capacity_bytes,
    filesystem = excluded.filesystem,
    bus_type = excluded.bus_type,
    pci_path = excluded.pci_path,
    numa_node = excluded.numa_node,
    topology_json = excluded.topology_json,
    volumes_json = excluded.volumes_json,
    topology_confidence = excluded.topology_confidence,
    topology_source = excluded.topology_source,
    present = 1,
    last_seen_at = CURRENT_TIMESTAMP;
)sql";
        if (refresh_storage_presence) for (const auto& storage : storage_devices) {
            Statement statement(handle_, kStorageSql);
            sqlite3_stmt* raw = statement.get();
            BindText(handle_, raw, 1, profile.machine_hash);
            BindText(handle_, raw, 2, storage.persistent_id);
            BindText(handle_, raw, 3, storage.model);
            BindText(handle_, raw, 4, storage.firmware);
            BindOptionalInt64(handle_, raw, 5, storage.capacity_bytes);
            BindText(handle_, raw, 6, storage.filesystem);
            BindText(handle_, raw, 7, storage.bus_type);
            BindText(handle_, raw, 8, storage.pci_path);
            BindOptionalInt64(handle_, raw, 9, storage.numa_node);
            BindText(handle_, raw, 10, storage.topology_json);
            BindText(handle_, raw, 11, storage.volumes_json);
            BindText(handle_, raw, 12, storage.topology_confidence);
            BindText(handle_, raw, 13, storage.topology_source);
            CheckDone(handle_, sqlite3_step(raw));
        }

        Execute("COMMIT;");
    } catch (...) {
        sqlite3_exec(handle_, "ROLLBACK;", nullptr, nullptr, nullptr);
        throw;
    }
}

HardwareInventoryCounts Database::InventoryCounts(const std::string& machine_hash) const {
    const auto count = [&](const char* sql) {
        Statement statement(handle_, sql);
        BindText(handle_, statement.get(), 1, machine_hash);
        const int result = sqlite3_step(statement.get());
        if (result != SQLITE_ROW) {
            throw DatabaseError(result, sqlite3_errmsg(handle_));
        }
        return sqlite3_column_int64(statement.get(), 0);
    };

    HardwareInventoryCounts counts;
    counts.active_gpus = count(
        "SELECT COUNT(*) FROM gpu_devices WHERE machine_hash = ? AND present = 1;");
    counts.total_gpu_rows = count(
        "SELECT COUNT(*) FROM gpu_devices WHERE machine_hash = ?;");
    counts.active_storage_devices = count(
        "SELECT COUNT(*) FROM storage_devices WHERE machine_hash = ? AND present = 1;");
    counts.total_storage_rows = count(
        "SELECT COUNT(*) FROM storage_devices WHERE machine_hash = ?;");
    return counts;
}

std::int64_t Database::StartBenchmarkSession(const BenchmarkSessionInput& session) {
    static constexpr const char* kSql = R"sql(
INSERT INTO benchmark_sessions(
    machine_hash, sidecar_spec_version, sidecar_git_commit,
    llama_cpp_git_commit, trace_mode, notes, cuda_runtime_version,
    cuda_driver_version, cuda_toolkit_version, nvidia_driver_version, os_version)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";

    Statement statement(handle_, kSql);
    sqlite3_stmt* raw = statement.get();
    BindText(handle_, raw, 1, session.machine_hash);
    BindText(handle_, raw, 2, session.sidecar_spec_version);
    BindText(handle_, raw, 3, session.sidecar_git_commit);
    BindOptionalText(handle_, raw, 4, session.llama_cpp_git_commit);
    BindText(handle_, raw, 5, session.trace_mode);
    BindOptionalText(handle_, raw, 6, session.notes);
    BindOptionalInt64(handle_, raw, 7, session.cuda_runtime_version);
    BindOptionalInt64(handle_, raw, 8, session.cuda_driver_version);
    BindOptionalText(handle_, raw, 9, session.cuda_toolkit_version);
    BindOptionalText(handle_, raw, 10, session.nvidia_driver_version);
    BindOptionalText(handle_, raw, 11, session.os_version);
    CheckDone(handle_, sqlite3_step(raw));
    return sqlite3_last_insert_rowid(handle_);
}

void Database::CompleteBenchmarkSession(std::int64_t session_id, const std::string& status) {
    static constexpr const char* kSql = R"sql(
UPDATE benchmark_sessions
SET completed_at = CURRENT_TIMESTAMP, status = ?
WHERE session_id = ?;
)sql";

    Statement statement(handle_, kSql);
    sqlite3_stmt* raw = statement.get();
    BindText(handle_, raw, 1, status);
    CheckResult(handle_, sqlite3_bind_int64(raw, 2, session_id));
    CheckDone(handle_, sqlite3_step(raw));
    if (sqlite3_changes(handle_) != 1) {
        throw DatabaseError(SQLITE_NOTFOUND, "benchmark session was not found");
    }
}

std::int64_t Database::InsertTraceConfiguration(
    const TraceConfigurationInput& configuration) {
    static constexpr const char* kSql = R"sql(
INSERT INTO trace_configuration(
    session_id, record_size_bytes, ring_capacity, producer_count,
    collector_batch_size, collector_strategy, timestamp_method)
VALUES (?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, kSql);
    auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, configuration.session_id));
    CheckResult(handle_, sqlite3_bind_int64(raw, 2, configuration.record_size_bytes));
    CheckResult(handle_, sqlite3_bind_int64(raw, 3, configuration.ring_capacity));
    CheckResult(handle_, sqlite3_bind_int64(raw, 4, configuration.producer_count));
    CheckResult(handle_, sqlite3_bind_int64(raw, 5, configuration.collector_batch_size));
    BindText(handle_, raw, 6, configuration.collector_strategy);
    BindText(handle_, raw, 7, configuration.timestamp_method);
    CheckDone(handle_, sqlite3_step(raw));
    return sqlite3_last_insert_rowid(handle_);
}

std::int64_t Database::InsertTraceBenchmark(const TraceBenchmarkInput& benchmark) {
    static constexpr const char* kSql = R"sql(
INSERT INTO trace_benchmarks(
    session_id, configuration_id, workload, repetition,
    baseline_duration_ns, traced_duration_ns, events_generated,
    events_written, events_dropped, events_per_second,
    producer_cpu_ns, collector_cpu_ns, observer_overhead_raw,
    ring_high_water_mark, trace_bytes, disk_write_bytes_per_second,
    try_push_p50_ns, try_push_p95_ns, try_push_p99_ns, raw_samples_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, kSql);
    auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, benchmark.session_id));
    BindOptionalInt64(handle_, raw, 2, benchmark.configuration_id);
    BindText(handle_, raw, 3, benchmark.workload);
    CheckResult(handle_, sqlite3_bind_int64(raw, 4, benchmark.repetition));
    CheckResult(handle_, sqlite3_bind_int64(raw, 5, benchmark.baseline_duration_ns));
    CheckResult(handle_, sqlite3_bind_int64(raw, 6, benchmark.traced_duration_ns));
    CheckResult(handle_, sqlite3_bind_int64(raw, 7, benchmark.events_generated));
    CheckResult(handle_, sqlite3_bind_int64(raw, 8, benchmark.events_written));
    CheckResult(handle_, sqlite3_bind_int64(raw, 9, benchmark.events_dropped));
    CheckResult(handle_, sqlite3_bind_double(raw, 10, benchmark.events_per_second));
    BindOptionalInt64(handle_, raw, 11, benchmark.producer_cpu_ns);
    BindOptionalInt64(handle_, raw, 12, benchmark.collector_cpu_ns);
    CheckResult(handle_, sqlite3_bind_double(raw, 13, benchmark.observer_overhead_raw));
    BindOptionalInt64(handle_, raw, 14, benchmark.ring_high_water_mark);
    BindOptionalInt64(handle_, raw, 15, benchmark.trace_bytes);
    if (benchmark.disk_write_bytes_per_second) {
        CheckResult(handle_, sqlite3_bind_double(raw, 16, *benchmark.disk_write_bytes_per_second));
    } else CheckResult(handle_, sqlite3_bind_null(raw, 16));
    if (benchmark.try_push_p50_ns) {
        CheckResult(handle_, sqlite3_bind_double(raw, 17, *benchmark.try_push_p50_ns));
    } else CheckResult(handle_, sqlite3_bind_null(raw, 17));
    if (benchmark.try_push_p95_ns) {
        CheckResult(handle_, sqlite3_bind_double(raw, 18, *benchmark.try_push_p95_ns));
    } else CheckResult(handle_, sqlite3_bind_null(raw, 18));
    if (benchmark.try_push_p99_ns) {
        CheckResult(handle_, sqlite3_bind_double(raw, 19, *benchmark.try_push_p99_ns));
    } else CheckResult(handle_, sqlite3_bind_null(raw, 19));
    BindText(handle_, raw, 20, benchmark.raw_samples_json);
    CheckDone(handle_, sqlite3_step(raw));
    return sqlite3_last_insert_rowid(handle_);
}

std::int64_t Database::InsertTraceFile(const TraceFileInput& trace_file) {
    static constexpr const char* kSql = R"sql(
INSERT INTO trace_files(
    session_id, path, format_version, record_size_bytes, trace_mode,
    trace_bytes, event_count, dropped_events, ring_high_water_mark,
    is_complete, sha256)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, kSql);
    auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, trace_file.session_id));
    BindText(handle_, raw, 2, trace_file.path);
    CheckResult(handle_, sqlite3_bind_int64(raw, 3, trace_file.format_version));
    CheckResult(handle_, sqlite3_bind_int64(raw, 4, trace_file.record_size_bytes));
    BindText(handle_, raw, 5, trace_file.trace_mode);
    CheckResult(handle_, sqlite3_bind_int64(raw, 6, trace_file.trace_bytes));
    CheckResult(handle_, sqlite3_bind_int64(raw, 7, trace_file.event_count));
    CheckResult(handle_, sqlite3_bind_int64(raw, 8, trace_file.dropped_events));
    BindOptionalInt64(handle_, raw, 9, trace_file.ring_high_water_mark);
    CheckResult(handle_, sqlite3_bind_int(raw, 10, trace_file.is_complete ? 1 : 0));
    BindOptionalText(handle_, raw, 11, trace_file.sha256);
    CheckDone(handle_, sqlite3_step(raw));
    return sqlite3_last_insert_rowid(handle_);
}

std::int64_t Database::InsertHostMemoryBenchmark(
    const HostMemoryBenchmarkInput& benchmark) {
    static constexpr const char* kSql = R"sql(
INSERT INTO host_memory_benchmarks(
    session_id, method, mode, requested_bytes, measured_repetitions,
    status, safety_status, timer_method, timer_bracket_overhead_ns,
    timer_resolution_ns, cuda_allocation_flags, cuda_registration_flags,
    statistics_json, amortization_json, message)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, kSql);
    auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, benchmark.session_id));
    BindText(handle_, raw, 2, benchmark.method);
    BindText(handle_, raw, 3, benchmark.mode);
    CheckResult(handle_, sqlite3_bind_int64(raw, 4, benchmark.requested_bytes));
    CheckResult(handle_, sqlite3_bind_int64(raw, 5, benchmark.measured_repetitions));
    BindText(handle_, raw, 6, benchmark.status);
    BindText(handle_, raw, 7, benchmark.safety_status);
    BindText(handle_, raw, 8, benchmark.timer_method);
    CheckResult(handle_, sqlite3_bind_int64(raw, 9, benchmark.timer_bracket_overhead_ns));
    CheckResult(handle_, sqlite3_bind_int64(raw, 10, benchmark.timer_resolution_ns));
    BindOptionalText(handle_, raw, 11, benchmark.cuda_allocation_flags);
    BindOptionalText(handle_, raw, 12, benchmark.cuda_registration_flags);
    BindText(handle_, raw, 13, benchmark.statistics_json);
    BindText(handle_, raw, 14, benchmark.amortization_json);
    BindOptionalText(handle_, raw, 15, benchmark.message);
    CheckDone(handle_, sqlite3_step(raw));
    return sqlite3_last_insert_rowid(handle_);
}

void Database::InsertHostMemorySample(const HostMemorySampleInput& sample) {
    static constexpr const char* kSql = R"sql(
INSERT INTO host_memory_samples(
    host_memory_benchmark_id, repetition, cold_setup, status,
    allocation_raw_ns, allocation_corrected_ns,
    first_touch_raw_ns, first_touch_corrected_ns,
    warm_touch_raw_ns, warm_touch_corrected_ns,
    registration_raw_ns, registration_corrected_ns,
    unregistration_raw_ns, unregistration_corrected_ns,
    cleanup_raw_ns, cleanup_corrected_ns,
    before_snapshot_json, after_setup_snapshot_json, after_cleanup_snapshot_json,
    available_recovery_delta_bytes, available_recovery_percent, noisy,
    cuda_error, windows_error, message)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, kSql);
    auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, sample.benchmark_id));
    CheckResult(handle_, sqlite3_bind_int64(raw, 2, sample.repetition));
    CheckResult(handle_, sqlite3_bind_int(raw, 3, sample.cold_setup ? 1 : 0));
    BindText(handle_, raw, 4, sample.status);
    BindOptionalInt64(handle_, raw, 5, sample.allocation_raw_ns);
    BindOptionalInt64(handle_, raw, 6, sample.allocation_corrected_ns);
    BindOptionalInt64(handle_, raw, 7, sample.first_touch_raw_ns);
    BindOptionalInt64(handle_, raw, 8, sample.first_touch_corrected_ns);
    BindOptionalInt64(handle_, raw, 9, sample.warm_touch_raw_ns);
    BindOptionalInt64(handle_, raw, 10, sample.warm_touch_corrected_ns);
    BindOptionalInt64(handle_, raw, 11, sample.registration_raw_ns);
    BindOptionalInt64(handle_, raw, 12, sample.registration_corrected_ns);
    BindOptionalInt64(handle_, raw, 13, sample.unregistration_raw_ns);
    BindOptionalInt64(handle_, raw, 14, sample.unregistration_corrected_ns);
    BindOptionalInt64(handle_, raw, 15, sample.cleanup_raw_ns);
    BindOptionalInt64(handle_, raw, 16, sample.cleanup_corrected_ns);
    BindText(handle_, raw, 17, sample.before_snapshot_json);
    BindText(handle_, raw, 18, sample.after_setup_snapshot_json);
    BindText(handle_, raw, 19, sample.after_cleanup_snapshot_json);
    BindOptionalInt64(handle_, raw, 20, sample.available_recovery_delta_bytes);
    BindOptionalDouble(handle_, raw, 21, sample.available_recovery_percent);
    CheckResult(handle_, sqlite3_bind_int(raw, 22, sample.noisy ? 1 : 0));
    BindOptionalInt64(handle_, raw, 23, sample.cuda_error);
    BindOptionalInt64(handle_, raw, 24, sample.windows_error);
    BindOptionalText(handle_, raw, 25, sample.message);
    CheckDone(handle_, sqlite3_step(raw));
}

std::int64_t Database::InsertPersistentArenaTest(
    const PersistentArenaTestInput& test) {
    static constexpr const char* kSql = R"sql(
INSERT INTO persistent_arena_tests(
    session_id, backend, capacity_bytes, reuse_count, setup_raw_ns,
    cleanup_raw_ns, reuse_statistics_json, raw_reuse_samples_json,
    contents_verified, status, message)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, kSql);
    auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, test.session_id));
    BindText(handle_, raw, 2, test.backend);
    CheckResult(handle_, sqlite3_bind_int64(raw, 3, test.capacity_bytes));
    CheckResult(handle_, sqlite3_bind_int64(raw, 4, test.reuse_count));
    BindOptionalInt64(handle_, raw, 5, test.setup_raw_ns);
    BindOptionalInt64(handle_, raw, 6, test.cleanup_raw_ns);
    BindText(handle_, raw, 7, test.reuse_statistics_json);
    BindText(handle_, raw, 8, test.raw_reuse_samples_json);
    CheckResult(handle_, sqlite3_bind_int(raw, 9, test.contents_verified ? 1 : 0));
    BindText(handle_, raw, 10, test.status);
    BindOptionalText(handle_, raw, 11, test.message);
    CheckDone(handle_, sqlite3_step(raw));
    return sqlite3_last_insert_rowid(handle_);
}

MemoryLaboratoryCounts Database::MemoryCounts() const {
    MemoryLaboratoryCounts counts;
    counts.benchmarks = QueryInt64("SELECT COUNT(*) FROM host_memory_benchmarks;");
    counts.samples = QueryInt64("SELECT COUNT(*) FROM host_memory_samples;");
    counts.pressure_events = QueryInt64("SELECT COUNT(*) FROM memory_pressure_events;");
    counts.arena_tests = QueryInt64("SELECT COUNT(*) FROM persistent_arena_tests;");
    return counts;
}

std::int64_t Database::InsertCudaTransferConfiguration(
    const CudaTransferConfigurationInput& value) {
    static constexpr const char* sql = R"sql(
INSERT INTO cuda_transfer_configurations(
 session_id, configuration_hash, experiment, direction, host_memory_class, api_mode,
 transfer_bytes, stream_mode, batch_count, chunk_count, warmup_count,
 measured_repetitions, device_index, host_buffer_bytes, device_buffer_bytes,
 validation_mode, timer_mode, host_timer_bracket_overhead_ns,
 host_timer_resolution_ns, cuda_event_mode, ordering_seed, status)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, sql); auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, value.session_id));
    BindText(handle_, raw, 2, value.configuration_hash);
    BindText(handle_, raw, 3, value.experiment); BindText(handle_, raw, 4, value.direction);
    BindText(handle_, raw, 5, value.host_memory_class); BindText(handle_, raw, 6, value.api_mode);
    CheckResult(handle_, sqlite3_bind_int64(raw, 7, value.transfer_bytes));
    BindText(handle_, raw, 8, value.stream_mode);
    CheckResult(handle_, sqlite3_bind_int64(raw, 9, value.batch_count));
    CheckResult(handle_, sqlite3_bind_int64(raw, 10, value.chunk_count));
    CheckResult(handle_, sqlite3_bind_int64(raw, 11, value.warmup_count));
    CheckResult(handle_, sqlite3_bind_int64(raw, 12, value.measured_repetitions));
    CheckResult(handle_, sqlite3_bind_int64(raw, 13, value.device_index));
    CheckResult(handle_, sqlite3_bind_int64(raw, 14, value.host_buffer_bytes));
    CheckResult(handle_, sqlite3_bind_int64(raw, 15, value.device_buffer_bytes));
    BindText(handle_, raw, 16, value.validation_mode); BindText(handle_, raw, 17, value.timer_mode);
    CheckResult(handle_, sqlite3_bind_int64(raw, 18, value.host_timer_bracket_overhead_ns));
    CheckResult(handle_, sqlite3_bind_int64(raw, 19, value.host_timer_resolution_ns));
    BindText(handle_, raw, 20, value.cuda_event_mode);
    CheckResult(handle_, sqlite3_bind_int64(raw, 21, value.ordering_seed));
    BindText(handle_, raw, 22, value.status); CheckDone(handle_, sqlite3_step(raw));
    return sqlite3_last_insert_rowid(handle_);
}

std::int64_t Database::InsertCudaTransferBenchmark(
    const CudaTransferBenchmarkInput& value) {
    static constexpr const char* sql = R"sql(
INSERT INTO cuda_transfer_benchmarks(
 cuda_transfer_configuration_id, status, verified, noisy, async_behavior,
 statistics_json, message) VALUES (?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, sql); auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, value.configuration_id));
    BindText(handle_, raw, 2, value.status);
    CheckResult(handle_, sqlite3_bind_int(raw, 3, value.verified ? 1 : 0));
    CheckResult(handle_, sqlite3_bind_int(raw, 4, value.noisy ? 1 : 0));
    BindText(handle_, raw, 5, value.async_behavior); BindText(handle_, raw, 6, value.statistics_json);
    BindOptionalText(handle_, raw, 7, value.message); CheckDone(handle_, sqlite3_step(raw));
    return sqlite3_last_insert_rowid(handle_);
}

void Database::InsertCudaTransferSample(const CudaTransferSampleInput& value) {
    static constexpr const char* sql = R"sql(
INSERT INTO cuda_transfer_samples(
 cuda_transfer_benchmark_id, repetition, payload_bytes, host_api_raw_ns,
 device_duration_ns, end_to_end_raw_ns, status, cuda_error, message)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, sql); auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, value.benchmark_id));
    CheckResult(handle_, sqlite3_bind_int64(raw, 2, value.repetition));
    CheckResult(handle_, sqlite3_bind_int64(raw, 3, value.payload_bytes));
    CheckResult(handle_, sqlite3_bind_int64(raw, 4, value.host_api_raw_ns));
    CheckResult(handle_, sqlite3_bind_int64(raw, 5, value.device_duration_ns));
    CheckResult(handle_, sqlite3_bind_int64(raw, 6, value.end_to_end_raw_ns));
    BindText(handle_, raw, 7, value.status); BindOptionalInt64(handle_, raw, 8, value.cuda_error);
    BindOptionalText(handle_, raw, 9, value.message); CheckDone(handle_, sqlite3_step(raw));
}

std::int64_t Database::InsertCudaBidirectionalBenchmark(
    const CudaBidirectionalBenchmarkInput& value) {
    static constexpr const char* sql = R"sql(
INSERT INTO cuda_bidirectional_benchmarks(
 session_id, host_memory_class, transfer_bytes, measured_repetitions,
 isolated_h2d_median_ns, isolated_d2h_median_ns, concurrent_h2d_median_ns,
 concurrent_d2h_median_ns, makespan_median_ns, aggregate_bytes_per_second,
 concurrency_benefit, verified, status, raw_samples_json, message)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, sql); auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, value.session_id));
    BindText(handle_, raw, 2, value.host_memory_class);
    CheckResult(handle_, sqlite3_bind_int64(raw, 3, value.transfer_bytes));
    CheckResult(handle_, sqlite3_bind_int64(raw, 4, value.measured_repetitions));
    CheckResult(handle_, sqlite3_bind_double(raw, 5, value.isolated_h2d_median_ns));
    CheckResult(handle_, sqlite3_bind_double(raw, 6, value.isolated_d2h_median_ns));
    CheckResult(handle_, sqlite3_bind_double(raw, 7, value.concurrent_h2d_median_ns));
    CheckResult(handle_, sqlite3_bind_double(raw, 8, value.concurrent_d2h_median_ns));
    CheckResult(handle_, sqlite3_bind_double(raw, 9, value.makespan_median_ns));
    CheckResult(handle_, sqlite3_bind_double(raw, 10, value.aggregate_bytes_per_second));
    CheckResult(handle_, sqlite3_bind_double(raw, 11, value.concurrency_benefit));
    CheckResult(handle_, sqlite3_bind_int(raw, 12, value.verified ? 1 : 0));
    BindText(handle_, raw, 13, value.status); BindText(handle_, raw, 14, value.raw_samples_json);
    BindOptionalText(handle_, raw, 15, value.message); CheckDone(handle_, sqlite3_step(raw));
    return sqlite3_last_insert_rowid(handle_);
}

void Database::InsertCudaLinkState(const CudaLinkStateInput& value) {
    static constexpr const char* sql = R"sql(
INSERT INTO cuda_link_state_observations(
 session_id, phase, temperature_c, graphics_clock_mhz, memory_clock_mhz,
 power_watts, power_limit_watts, pcie_generation, pcie_width)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, sql); auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, value.session_id)); BindText(handle_, raw, 2, value.phase);
    BindOptionalInt64(handle_, raw, 3, value.temperature_c); BindOptionalInt64(handle_, raw, 4, value.graphics_clock_mhz);
    BindOptionalInt64(handle_, raw, 5, value.memory_clock_mhz); BindOptionalDouble(handle_, raw, 6, value.power_watts);
    BindOptionalDouble(handle_, raw, 7, value.power_limit_watts); BindOptionalInt64(handle_, raw, 8, value.pcie_generation);
    BindOptionalInt64(handle_, raw, 9, value.pcie_width); CheckDone(handle_, sqlite3_step(raw));
}

void Database::InsertCudaTransferProfile(const CudaTransferProfileInput& value) {
    static constexpr const char* sql = R"sql(
INSERT INTO cuda_transfer_profiles(
 session_id, direction, host_memory_class, api_mode, peak_bytes_per_second,
 peak_size_bytes, knee_80_bytes, knee_90_bytes, knee_95_bytes, candidate_sizes_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, sql); auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, value.session_id)); BindText(handle_, raw, 2, value.direction);
    BindText(handle_, raw, 3, value.host_memory_class); BindText(handle_, raw, 4, value.api_mode);
    CheckResult(handle_, sqlite3_bind_double(raw, 5, value.peak_bytes_per_second));
    CheckResult(handle_, sqlite3_bind_int64(raw, 6, value.peak_size_bytes));
    CheckResult(handle_, sqlite3_bind_int64(raw, 7, value.knee_80_bytes));
    CheckResult(handle_, sqlite3_bind_int64(raw, 8, value.knee_90_bytes));
    CheckResult(handle_, sqlite3_bind_int64(raw, 9, value.knee_95_bytes));
    BindText(handle_, raw, 10, value.candidate_sizes_json); CheckDone(handle_, sqlite3_step(raw));
}

CudaTransferCounts Database::TransferCounts() const {
    CudaTransferCounts counts;
    counts.configurations = QueryInt64("SELECT COUNT(*) FROM cuda_transfer_configurations;");
    counts.benchmarks = QueryInt64("SELECT COUNT(*) FROM cuda_transfer_benchmarks;");
    counts.samples = QueryInt64("SELECT COUNT(*) FROM cuda_transfer_samples;");
    counts.bidirectional = QueryInt64("SELECT COUNT(*) FROM cuda_bidirectional_benchmarks;");
    counts.link_observations = QueryInt64("SELECT COUNT(*) FROM cuda_link_state_observations;");
    counts.profiles = QueryInt64("SELECT COUNT(*) FROM cuda_transfer_profiles;");
    return counts;
}

CudaTransferReportData Database::TransferReport() const {
    CudaTransferReportData report;
    report.counts = TransferCounts();
    static constexpr const char* profiles_sql = R"sql(
SELECT p.session_id, p.direction, p.host_memory_class, p.api_mode,
       p.peak_bytes_per_second, p.peak_size_bytes, p.knee_80_bytes,
       p.knee_90_bytes, p.knee_95_bytes,
       json_extract(b.statistics_json, '$.host_api_ns.median'),
       json_extract(b.statistics_json, '$.device_ns.median'),
       json_extract(b.statistics_json, '$.end_to_end_ns.median'),
       json_extract(b.statistics_json, '$.device_ns.p95'),
       json_extract(b.statistics_json, '$.device_ns.p99'),
       json_extract(b.statistics_json, '$.p99_meaningful')
FROM cuda_transfer_profiles p
JOIN cuda_transfer_configurations c
  ON c.session_id = p.session_id
 AND c.direction = p.direction
 AND c.host_memory_class = p.host_memory_class
 AND c.api_mode = p.api_mode
 AND c.transfer_bytes = p.peak_size_bytes
JOIN cuda_transfer_benchmarks b USING(cuda_transfer_configuration_id)
WHERE p.session_id = (SELECT MAX(session_id) FROM cuda_transfer_profiles)
ORDER BY p.direction, p.host_memory_class, p.api_mode;
)sql";
    Statement profiles(handle_, profiles_sql);
    while (sqlite3_step(profiles.get()) == SQLITE_ROW) {
        CudaTransferProfileSummary value;
        value.session_id = sqlite3_column_int64(profiles.get(), 0);
        const auto text = [&](int column) {
            const auto* raw = sqlite3_column_text(profiles.get(), column);
            return raw ? std::string(reinterpret_cast<const char*>(raw)) : std::string{};
        };
        value.direction = text(1); value.host_memory_class = text(2); value.api_mode = text(3);
        value.peak_bytes_per_second = sqlite3_column_double(profiles.get(), 4);
        value.peak_size_bytes = sqlite3_column_int64(profiles.get(), 5);
        value.knee_80_bytes = sqlite3_column_int64(profiles.get(), 6);
        value.knee_90_bytes = sqlite3_column_int64(profiles.get(), 7);
        value.knee_95_bytes = sqlite3_column_int64(profiles.get(), 8);
        value.host_api_median_ns = sqlite3_column_double(profiles.get(), 9);
        value.device_median_ns = sqlite3_column_double(profiles.get(), 10);
        value.end_to_end_median_ns = sqlite3_column_double(profiles.get(), 11);
        value.device_p95_ns = sqlite3_column_double(profiles.get(), 12);
        value.device_p99_ns = sqlite3_column_double(profiles.get(), 13);
        value.p99_meaningful = sqlite3_column_int(profiles.get(), 14) != 0;
        report.latest_profiles.push_back(std::move(value));
    }
    Statement bidirectional(handle_,
        "SELECT COALESCE(MAX(aggregate_bytes_per_second), 0) "
        "FROM cuda_bidirectional_benchmarks;");
    if (sqlite3_step(bidirectional.get()) == SQLITE_ROW)
        report.best_bidirectional_bytes_per_second = sqlite3_column_double(
            bidirectional.get(), 0);

    static constexpr const char* supplemental_sql = R"sql(
SELECT c.session_id, c.experiment, c.direction, c.host_memory_class,
       c.transfer_bytes, c.batch_count, c.chunk_count, COUNT(*),
       AVG(s.host_api_raw_ns), AVG(s.device_duration_ns),
       AVG(s.end_to_end_raw_ns),
       AVG(CASE WHEN s.device_duration_ns > 0
                THEN s.payload_bytes * 1000000000.0 / s.device_duration_ns
                ELSE 0 END)
FROM cuda_transfer_configurations c
JOIN cuda_transfer_benchmarks b USING(cuda_transfer_configuration_id)
JOIN cuda_transfer_samples s USING(cuda_transfer_benchmark_id)
WHERE c.experiment IN ('BATCH_SYNC_EACH', 'BATCH_ONE_SYNC',
                       'CONTIGUOUS_CHUNKING')
  AND c.session_id = (
      SELECT MAX(c2.session_id) FROM cuda_transfer_configurations c2
      WHERE c2.experiment = c.experiment)
GROUP BY c.cuda_transfer_configuration_id
ORDER BY c.experiment, c.direction, c.host_memory_class,
         c.transfer_bytes, c.batch_count, c.chunk_count;
)sql";
    Statement supplemental(handle_, supplemental_sql);
    while (sqlite3_step(supplemental.get()) == SQLITE_ROW) {
        CudaTransferExperimentSummary value;
        const auto text = [&](int column) {
            const auto* raw = sqlite3_column_text(supplemental.get(), column);
            return raw ? std::string(reinterpret_cast<const char*>(raw)) : std::string{};
        };
        value.session_id = sqlite3_column_int64(supplemental.get(), 0);
        value.experiment = text(1); value.direction = text(2);
        value.host_memory_class = text(3);
        value.transfer_bytes = sqlite3_column_int64(supplemental.get(), 4);
        value.batch_count = sqlite3_column_int64(supplemental.get(), 5);
        value.chunk_count = sqlite3_column_int64(supplemental.get(), 6);
        value.sample_count = sqlite3_column_int64(supplemental.get(), 7);
        value.mean_host_api_ns = sqlite3_column_double(supplemental.get(), 8);
        value.mean_device_ns = sqlite3_column_double(supplemental.get(), 9);
        value.mean_end_to_end_ns = sqlite3_column_double(supplemental.get(), 10);
        value.mean_bytes_per_second = sqlite3_column_double(supplemental.get(), 11);
        report.latest_supplemental.push_back(std::move(value));
    }

    static constexpr const char* sustained_sql = R"sql(
SELECT c.session_id, c.direction, c.host_memory_class, c.transfer_bytes,
       s.payload_bytes,
       s.payload_bytes * 1000000000.0 / s.device_duration_ns AS bytes_per_second
FROM cuda_transfer_configurations c
JOIN cuda_transfer_benchmarks b USING(cuda_transfer_configuration_id)
JOIN cuda_transfer_samples s USING(cuda_transfer_benchmark_id)
WHERE c.experiment = 'SUSTAINED' AND b.status = 'SUCCESS'
  AND s.status = 'SUCCESS' AND s.device_duration_ns > 0
ORDER BY bytes_per_second DESC LIMIT 1;
)sql";
    Statement sustained(handle_, sustained_sql);
    if (sqlite3_step(sustained.get()) == SQLITE_ROW) {
        CudaSustainedSummary value;
        const auto text = [&](int column) {
            const auto* raw = sqlite3_column_text(sustained.get(), column);
            return raw ? std::string(reinterpret_cast<const char*>(raw)) : std::string{};
        };
        value.session_id = sqlite3_column_int64(sustained.get(), 0);
        value.direction = text(1); value.host_memory_class = text(2);
        value.transfer_bytes = sqlite3_column_int64(sustained.get(), 3);
        value.payload_bytes = sqlite3_column_int64(sustained.get(), 4);
        value.bytes_per_second = sqlite3_column_double(sustained.get(), 5);
        report.best_sustained = std::move(value);
    }

    static constexpr const char* link_sql = R"sql(
SELECT session_id, phase, temperature_c, graphics_clock_mhz, memory_clock_mhz,
       power_watts, power_limit_watts, pcie_generation, pcie_width
FROM cuda_link_state_observations
WHERE session_id = (SELECT MAX(session_id) FROM cuda_link_state_observations)
ORDER BY cuda_link_state_observation_id;
)sql";
    Statement link(handle_, link_sql);
    const auto optional_int = [&](sqlite3_stmt* statement, int column)
        -> std::optional<std::int64_t> {
        if (sqlite3_column_type(statement, column) == SQLITE_NULL) return std::nullopt;
        return sqlite3_column_int64(statement, column);
    };
    const auto optional_double = [&](sqlite3_stmt* statement, int column)
        -> std::optional<double> {
        if (sqlite3_column_type(statement, column) == SQLITE_NULL) return std::nullopt;
        return sqlite3_column_double(statement, column);
    };
    while (sqlite3_step(link.get()) == SQLITE_ROW) {
        CudaLinkStateSummary value;
        value.session_id = sqlite3_column_int64(link.get(), 0);
        const auto* phase = sqlite3_column_text(link.get(), 1);
        value.phase = phase ? reinterpret_cast<const char*>(phase) : "";
        value.temperature_c = optional_int(link.get(), 2);
        value.graphics_clock_mhz = optional_int(link.get(), 3);
        value.memory_clock_mhz = optional_int(link.get(), 4);
        value.power_watts = optional_double(link.get(), 5);
        value.power_limit_watts = optional_double(link.get(), 6);
        value.pcie_generation = optional_int(link.get(), 7);
        value.pcie_width = optional_int(link.get(), 8);
        report.latest_link_states.push_back(std::move(value));
    }
    return report;
}

std::int64_t Database::InsertComputeWorkloadProfile(
    const ComputeWorkloadProfileInput& value) {
    static constexpr const char* sql = R"sql(
INSERT INTO compute_workload_profiles(
 session_id, workload, target_compute_us, calibration_statistics_json,
 calibration_samples, alu_iterations, memory_working_set_bytes, memory_passes,
 memory_block_size, memory_elements_per_thread, gemm_m, gemm_n, gemm_k,
 gemm_repetitions, gemm_a_type, gemm_b_type, gemm_c_type, gemm_compute_type,
 gemm_math_mode, gemm_algorithm, cublas_version, validated, status, message)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, sql); auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, value.session_id));
    BindText(handle_, raw, 2, value.workload);
    CheckResult(handle_, sqlite3_bind_double(raw, 3, value.target_compute_us));
    BindText(handle_, raw, 4, value.calibration_statistics_json);
    CheckResult(handle_, sqlite3_bind_int64(raw, 5, value.calibration_samples));
    CheckResult(handle_, sqlite3_bind_int64(raw, 6, value.alu_iterations));
    CheckResult(handle_, sqlite3_bind_int64(raw, 7, value.memory_working_set_bytes));
    CheckResult(handle_, sqlite3_bind_int64(raw, 8, value.memory_passes));
    CheckResult(handle_, sqlite3_bind_int64(raw, 9, value.memory_block_size));
    CheckResult(handle_, sqlite3_bind_int64(raw, 10, value.memory_elements_per_thread));
    CheckResult(handle_, sqlite3_bind_int64(raw, 11, value.gemm_m));
    CheckResult(handle_, sqlite3_bind_int64(raw, 12, value.gemm_n));
    CheckResult(handle_, sqlite3_bind_int64(raw, 13, value.gemm_k));
    CheckResult(handle_, sqlite3_bind_int64(raw, 14, value.gemm_repetitions));
    BindText(handle_, raw, 15, value.gemm_a_type); BindText(handle_, raw, 16, value.gemm_b_type);
    BindText(handle_, raw, 17, value.gemm_c_type); BindText(handle_, raw, 18, value.gemm_compute_type);
    BindText(handle_, raw, 19, value.gemm_math_mode); BindText(handle_, raw, 20, value.gemm_algorithm);
    BindOptionalInt64(handle_, raw, 21, value.cublas_version);
    CheckResult(handle_, sqlite3_bind_int(raw, 22, value.validated ? 1 : 0));
    BindText(handle_, raw, 23, value.status); BindOptionalText(handle_, raw, 24, value.message);
    CheckDone(handle_, sqlite3_step(raw));
    return sqlite3_last_insert_rowid(handle_);
}

std::int64_t Database::InsertCudaOverlapConfiguration(
    const CudaOverlapConfigurationInput& value) {
    static constexpr const char* sql = R"sql(
INSERT INTO cuda_overlap_configurations(
 session_id, compute_workload_profile_id, configuration_hash, phase,
 direction, host_memory_class, transfer_bytes, target_compute_us,
 measured_repetitions, ordering_seed, gate_delay_ns, gate_margin_ns,
 device_index, instrumentation_mode, refinement_reason, status)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, sql); auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, value.session_id));
    CheckResult(handle_, sqlite3_bind_int64(raw, 2, value.compute_workload_profile_id));
    BindText(handle_, raw, 3, value.configuration_hash); BindText(handle_, raw, 4, value.phase);
    BindText(handle_, raw, 5, value.direction); BindText(handle_, raw, 6, value.host_memory_class);
    CheckResult(handle_, sqlite3_bind_int64(raw, 7, value.transfer_bytes));
    CheckResult(handle_, sqlite3_bind_double(raw, 8, value.target_compute_us));
    CheckResult(handle_, sqlite3_bind_int64(raw, 9, value.measured_repetitions));
    CheckResult(handle_, sqlite3_bind_int64(raw, 10, value.ordering_seed));
    CheckResult(handle_, sqlite3_bind_int64(raw, 11, value.gate_delay_ns));
    CheckResult(handle_, sqlite3_bind_int64(raw, 12, value.gate_margin_ns));
    CheckResult(handle_, sqlite3_bind_int64(raw, 13, value.device_index));
    BindText(handle_, raw, 14, value.instrumentation_mode);
    BindText(handle_, raw, 15, value.refinement_reason); BindText(handle_, raw, 16, value.status);
    CheckDone(handle_, sqlite3_step(raw));
    return sqlite3_last_insert_rowid(handle_);
}

std::int64_t Database::InsertCudaOverlapBenchmark(
    const CudaOverlapBenchmarkInput& value) {
    static constexpr const char* sql = R"sql(
INSERT INTO cuda_overlap_benchmarks(
 cuda_overlap_configuration_id, status, refined, statistics_json,
 telemetry_before_json, telemetry_after_json, message)
VALUES (?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, sql); auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, value.configuration_id));
    BindText(handle_, raw, 2, value.status);
    CheckResult(handle_, sqlite3_bind_int(raw, 3, value.refined ? 1 : 0));
    BindText(handle_, raw, 4, value.statistics_json);
    BindText(handle_, raw, 5, value.telemetry_before_json);
    BindText(handle_, raw, 6, value.telemetry_after_json);
    BindOptionalText(handle_, raw, 7, value.message);
    CheckDone(handle_, sqlite3_step(raw));
    return sqlite3_last_insert_rowid(handle_);
}

void Database::InsertCudaOverlapSample(const CudaOverlapSampleInput& value) {
    static constexpr const char* sql = R"sql(
INSERT INTO cuda_overlap_samples(
 cuda_overlap_benchmark_id, repetition, baseline_block_id, c0_reference_ns,
 t0_reference_ns, cc_ns, tc_ns, makespan_device_primary_ns,
 makespan_device_crosscheck_ns, makespan_host_ns, host_submission_ns,
 gate_delay_ns, gate_actual_ns, gate_margin_ns, critical_path_delta_ns, compute_path_added_ns,
 compute_path_added_percent, compute_slowdown, transfer_slowdown,
 overlap_efficiency_raw, overlap_efficiency_normalized, hidden_fraction_raw,
 hidden_fraction_normalized, compute_retention, gate_valid, fit_1_percent,
 fit_2_percent, fit_5_percent, instrumentation_status, status, native_error, message)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, sql); auto* raw = statement.get(); int i = 1;
    CheckResult(handle_, sqlite3_bind_int64(raw, i++, value.benchmark_id));
    CheckResult(handle_, sqlite3_bind_int64(raw, i++, value.repetition));
    CheckResult(handle_, sqlite3_bind_int64(raw, i++, value.baseline_block_id));
    for (auto v : {value.c0_reference_ns, value.t0_reference_ns, value.cc_ns,
                   value.tc_ns, value.makespan_device_primary_ns,
                   value.makespan_device_crosscheck_ns, value.makespan_host_ns,
                   value.host_submission_ns, value.gate_delay_ns,
                   value.gate_actual_ns,
                   value.gate_margin_ns})
        CheckResult(handle_, sqlite3_bind_int64(raw, i++, v));
    for (auto v : {value.critical_path_delta_ns, value.compute_path_added_ns,
                   value.compute_path_added_percent, value.compute_slowdown,
                   value.transfer_slowdown, value.overlap_efficiency_raw,
                   value.overlap_efficiency_normalized, value.hidden_fraction_raw,
                   value.hidden_fraction_normalized, value.compute_retention})
        CheckResult(handle_, sqlite3_bind_double(raw, i++, v));
    for (bool v : {value.gate_valid, value.fit_1_percent, value.fit_2_percent,
                   value.fit_5_percent})
        CheckResult(handle_, sqlite3_bind_int(raw, i++, v ? 1 : 0));
    BindText(handle_, raw, i++, value.instrumentation_status);
    BindText(handle_, raw, i++, value.status);
    BindOptionalInt64(handle_, raw, i++, value.native_error);
    BindOptionalText(handle_, raw, i++, value.message);
    CheckDone(handle_, sqlite3_step(raw));
}

void Database::InsertCudaOverlapFitProfile(
    const CudaOverlapFitProfileInput& value) {
    static constexpr const char* sql = R"sql(
INSERT INTO cuda_overlap_fit_profiles(
 session_id, workload, direction, host_memory_class, target_compute_us,
 tolerance_percent, largest_measured_bytes, p99_added_percent, fit_rate, confidence)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, sql); auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, value.session_id));
    BindText(handle_, raw, 2, value.workload); BindText(handle_, raw, 3, value.direction);
    BindText(handle_, raw, 4, value.host_memory_class);
    CheckResult(handle_, sqlite3_bind_double(raw, 5, value.target_compute_us));
    CheckResult(handle_, sqlite3_bind_double(raw, 6, value.tolerance_percent));
    CheckResult(handle_, sqlite3_bind_int64(raw, 7, value.largest_measured_bytes));
    CheckResult(handle_, sqlite3_bind_double(raw, 8, value.p99_added_percent));
    CheckResult(handle_, sqlite3_bind_double(raw, 9, value.fit_rate));
    BindText(handle_, raw, 10, value.confidence); CheckDone(handle_, sqlite3_step(raw));
}

void Database::InsertInstrumentationControl(
    const InstrumentationControlInput& value) {
    static constexpr const char* sql = R"sql(
INSERT INTO instrumentation_control_benchmarks(
 session_id, workload, direction, transfer_bytes, target_compute_us,
 instrumentation_mode, compute_statistics_json, transfer_statistics_json,
 host_statistics_json, compute_bias_percent, transfer_bias_percent, rejected, status)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
)sql";
    Statement statement(handle_, sql); auto* raw = statement.get();
    CheckResult(handle_, sqlite3_bind_int64(raw, 1, value.session_id));
    BindText(handle_, raw, 2, value.workload); BindText(handle_, raw, 3, value.direction);
    CheckResult(handle_, sqlite3_bind_int64(raw, 4, value.transfer_bytes));
    CheckResult(handle_, sqlite3_bind_double(raw, 5, value.target_compute_us));
    BindText(handle_, raw, 6, value.instrumentation_mode);
    BindText(handle_, raw, 7, value.compute_statistics_json);
    BindText(handle_, raw, 8, value.transfer_statistics_json);
    BindText(handle_, raw, 9, value.host_statistics_json);
    CheckResult(handle_, sqlite3_bind_double(raw, 10, value.compute_bias_percent));
    CheckResult(handle_, sqlite3_bind_double(raw, 11, value.transfer_bias_percent));
    CheckResult(handle_, sqlite3_bind_int(raw, 12, value.rejected ? 1 : 0));
    BindText(handle_, raw, 13, value.status); CheckDone(handle_, sqlite3_step(raw));
}

CudaOverlapCounts Database::OverlapCounts() const {
    CudaOverlapCounts value;
    value.profiles = QueryInt64("SELECT COUNT(*) FROM compute_workload_profiles;");
    value.configurations = QueryInt64("SELECT COUNT(*) FROM cuda_overlap_configurations;");
    value.benchmarks = QueryInt64("SELECT COUNT(*) FROM cuda_overlap_benchmarks;");
    value.samples = QueryInt64("SELECT COUNT(*) FROM cuda_overlap_samples;");
    value.fit_profiles = QueryInt64("SELECT COUNT(*) FROM cuda_overlap_fit_profiles;");
    value.instrumentation_controls = QueryInt64("SELECT COUNT(*) FROM instrumentation_control_benchmarks;");
    return value;
}

std::vector<CudaOverlapEnvelopeSummary> Database::LatestOverlapEnvelopes() const {
    std::vector<CudaOverlapEnvelopeSummary> result;
    static constexpr const char* sql = R"sql(
SELECT session_id, workload, direction, host_memory_class, target_compute_us,
       tolerance_percent, largest_measured_bytes, p99_added_percent, fit_rate,
       confidence
FROM cuda_overlap_fit_profiles
WHERE session_id = (SELECT MAX(session_id) FROM cuda_overlap_fit_profiles)
ORDER BY workload, direction, host_memory_class, target_compute_us,
         tolerance_percent;
)sql";
    Statement statement(handle_, sql);
    while (sqlite3_step(statement.get()) == SQLITE_ROW) {
        CudaOverlapEnvelopeSummary value;
        const auto text = [&](int column) {
            const auto* raw = sqlite3_column_text(statement.get(), column);
            return raw ? std::string(reinterpret_cast<const char*>(raw)) : std::string{};
        };
        value.session_id = sqlite3_column_int64(statement.get(), 0);
        value.workload = text(1); value.direction = text(2);
        value.host_memory_class = text(3);
        value.target_compute_us = sqlite3_column_double(statement.get(), 4);
        value.tolerance_percent = sqlite3_column_double(statement.get(), 5);
        value.largest_measured_bytes = sqlite3_column_int64(statement.get(), 6);
        value.p99_added_percent = sqlite3_column_double(statement.get(), 7);
        value.fit_rate = sqlite3_column_double(statement.get(), 8);
        value.confidence = text(9);
        result.push_back(std::move(value));
    }
    return result;
}

void Database::BeginWriteTransaction() { Execute("BEGIN IMMEDIATE;"); }

void Database::CommitWriteTransaction() { Execute("COMMIT;"); }

void Database::RollbackWriteTransaction() noexcept {
    sqlite3_exec(handle_, "ROLLBACK;", nullptr, nullptr, nullptr);
}

void Database::Execute(const std::string& sql) const {
    char* error_message = nullptr;
    const int result = sqlite3_exec(handle_, sql.c_str(), nullptr, nullptr, &error_message);
    if (result != SQLITE_OK) {
        const std::string message = error_message != nullptr ? error_message : sqlite3_errmsg(handle_);
        sqlite3_free(error_message);
        throw DatabaseError(result, message);
    }
}

std::int64_t Database::QueryInt64(const std::string& sql) const {
    Statement statement(handle_, sql.c_str());
    const int result = sqlite3_step(statement.get());
    if (result != SQLITE_ROW) {
        throw DatabaseError(result, sqlite3_errmsg(handle_));
    }
    return sqlite3_column_int64(statement.get(), 0);
}

bool Database::TableExists(const std::string& table_name) const {
    Statement statement(handle_,
                        "SELECT EXISTS(SELECT 1 FROM sqlite_master "
                        "WHERE type = 'table' AND name = ?);");
    BindText(handle_, statement.get(), 1, table_name);
    const int result = sqlite3_step(statement.get());
    if (result != SQLITE_ROW) {
        throw DatabaseError(result, sqlite3_errmsg(handle_));
    }
    return sqlite3_column_int(statement.get(), 0) == 1;
}

}  // namespace sidecar::database
