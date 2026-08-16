ALTER TABLE hardware_profiles ADD COLUMN identity_version INTEGER NOT NULL DEFAULT 1;
ALTER TABLE hardware_profiles ADD COLUMN identity_quality TEXT NOT NULL DEFAULT 'FALLBACK';
ALTER TABLE hardware_profiles ADD COLUMN identity_basis TEXT NOT NULL DEFAULT '[]';
ALTER TABLE hardware_profiles ADD COLUMN system_manufacturer TEXT;
ALTER TABLE hardware_profiles ADD COLUMN system_product TEXT;
ALTER TABLE hardware_profiles ADD COLUMN smbios_version TEXT;
ALTER TABLE hardware_profiles ADD COLUMN cpu_vendor TEXT;
ALTER TABLE hardware_profiles ADD COLUMN cpu_architecture TEXT;
ALTER TABLE hardware_profiles ADD COLUMN processor_group_count INTEGER;
ALTER TABLE hardware_profiles ADD COLUMN cpu_package_count INTEGER;
ALTER TABLE hardware_profiles ADD COLUMN topology_json TEXT;
ALTER TABLE hardware_profiles ADD COLUMN snapshot_json TEXT;

ALTER TABLE gpu_devices ADD COLUMN cuda_device_index INTEGER;
ALTER TABLE gpu_devices ADD COLUMN concurrent_kernels INTEGER;
ALTER TABLE gpu_devices ADD COLUMN unified_addressing INTEGER;
ALTER TABLE gpu_devices ADD COLUMN can_map_host_memory INTEGER;
ALTER TABLE gpu_devices ADD COLUMN managed_memory INTEGER;
ALTER TABLE gpu_devices ADD COLUMN pageable_memory_access INTEGER;
ALTER TABLE gpu_devices ADD COLUMN attributes_json TEXT;
ALTER TABLE gpu_devices ADD COLUMN location_paths_json TEXT;
ALTER TABLE gpu_devices ADD COLUMN present INTEGER NOT NULL DEFAULT 1;
ALTER TABLE gpu_devices ADD COLUMN last_seen_at TEXT;

ALTER TABLE storage_devices ADD COLUMN volumes_json TEXT;
ALTER TABLE storage_devices ADD COLUMN topology_confidence TEXT NOT NULL DEFAULT 'UNKNOWN';
ALTER TABLE storage_devices ADD COLUMN topology_source TEXT;
ALTER TABLE storage_devices ADD COLUMN present INTEGER NOT NULL DEFAULT 1;
ALTER TABLE storage_devices ADD COLUMN last_seen_at TEXT;

INSERT INTO schema_migrations(version, name)
VALUES (2, 'Machine Identity V1 and refreshable hardware inventory');

PRAGMA user_version = 2;
