# M0 hardware discovery

Hardware discovery is observation only. It performs no transfer, allocation,
storage-throughput, CUDA timing, or contention benchmark.

## Provider architecture

`IHardwareDiscoveryProvider` returns raw identity inputs plus a current
`MachineSnapshot`. `HardwareDiscoveryService` computes identity independently of
snapshot state. Tests use synthetic providers, so CUDA, NVML, and particular
hardware are not test prerequisites.

The native Windows provider uses APIs rather than console-output parsing:

- `GetSystemFirmwareTable(RSMB)` for SMBIOS system, board, BIOS, UUID, and memory
  device structures;
- registry processor descriptors plus
  `GetLogicalProcessorInformationEx`, processor-group APIs, and NUMA APIs;
- `GetPhysicallyInstalledSystemMemory` and `GlobalMemoryStatusEx`;
- SetupAPI, Config Manager ancestor traversal, storage property queries, disk
  IOCTLs, and volume extents for physical-disk/volume discovery;
- Windows PnP location paths for PCI/ACPI routing evidence.

Memory channel count remains unknown because SMBIOS DIMM count is not proof of
active channel configuration.

## CUDA and NVML

`SIDECAR_ENABLE_CUDA` is a CMake switch. When enabled, CMake requires a detectable
CUDA Toolkit and links only the CUDA discovery component to `CUDA::cudart`.
Unrelated core/database/trace code remains buildable with the switch off.

CUDA discovery records device identity, VRAM, compute capability, PCI address,
copy-engine count, concurrent-kernel support, unified addressing, host-memory
mapping, managed/pageable memory capabilities, memory-pool capability, and a
small structured attribute set. It launches no kernels.

NVML is loaded dynamically at runtime. When available, it cross-checks UUID/PCI
identity and adds driver version, current/max PCIe generation and width, power
limit, and temperature snapshot fields. Missing NVML leaves nullable fields and
a structured warning; Sidecar never treats `nvidia-smi` text as its API.

## Storage and topology confidence

Physical storage identifiers are SHA-256 digests of Windows device-instance
identity, keeping raw serial-bearing IDs out of normal output. Capacity, model,
firmware, bus classification, disk number, filesystems, mount points, and device
location paths are current snapshot data and never enter `machine_hash`.

Endpoint location paths are `DIRECTLY_REPORTED`. If a disk endpoint exposes no
path, Sidecar walks Windows device parents to the nearest reported path and labels
the association `DERIVED`. Missing paths remain `UNKNOWN`.

The topology document is explicitly `DISCOVERED`. GPU and storage endpoints are
not labeled CPU-direct or PCH-routed unless Windows evidence establishes that.
The empirical contention topology remains unavailable until the later contention
laboratory measures it.

## Persistence

Schema migration 2 adds identity version/quality/basis, snapshot/topology JSON,
CUDA capability fields, and refresh metadata. Discovery refreshes a machine row
transactionally. GPU/storage rows use stable unique keys and `present` flags:
current devices are upserted, missing devices become inactive, and historical
rows that future benchmark foreign keys may reference are not destructively
deleted.

## CLI and JSON

```text
sidecar-lab info [--json] [--database path] [--no-persist]
sidecar-lab topology [--json] [--database path] [--no-persist]
```

Default discovery persists to `data/sidecar.db`; `--no-persist` performs a
read-only scan. JSON uses fixed schemas:

- `sidecar.hardware.discovery.v1`
- `sidecar.hardware.topology.v1`

Object fields have fixed order. Device/topology collections are sorted by stable
IDs before serialization. Snapshot JSON can legitimately change as available
RAM, temperature, PCIe power state, or installed devices change; identical input
objects serialize byte-identically.

