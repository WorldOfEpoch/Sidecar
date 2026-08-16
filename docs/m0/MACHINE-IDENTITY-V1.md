# Machine Identity V1

`SIDECAR_MACHINE_ID_VERSION = 1`

Machine Identity identifies the stable physical platform. Machine Snapshot is a
separate observation of current hardware and software state. Only Machine
Identity contributes to `machine_hash`.

## Canonical material

Sidecar serializes fixed UTF-8 fields in exactly this order, including the final
newline:

```text
SIDECAR-MACHINE-ID-V1
system_uuid=<normalized-or-<missing>>
system_vendor=<normalized-or-<missing>>
system_product=<normalized-or-<missing>>
board_vendor=<normalized-or-<missing>>
board_product=<normalized-or-<missing>>
platform_serial_hash=<component-hash-or-<missing>>
fallback_installation_hash=<component-hash-or-<missing>>
```

`machine_hash` is the lowercase hexadecimal SHA-256 digest of these bytes. It is
not a hash of JSON, field iteration order, or a C++ object representation.

Platform serials are never placed directly in canonical material. If a usable
system or board serial exists, Sidecar hashes a second fixed representation:

```text
SIDECAR-PLATFORM-SERIAL-V1
system_serial=<normalized-or-<missing>>
board_serial=<normalized-or-<missing>>
```

Only that digest enters the machine material. Raw UUIDs and serials are consumed
transiently during discovery and are not printed or persisted in the ordinary
hardware snapshot.

## Normalization

Text values are trimmed, internal whitespace runs become one ASCII space, and
ASCII casing is lowered. UUID input accepts braces and hyphens, validates exactly
32 hexadecimal digits, rejects all-zero/all-FF values, and is serialized in
canonical `8-4-4-4-12` form.

The following are treated as missing, case-insensitively after normalization:
empty text, `To Be Filled By O.E.M.`, `Default string`, `Unknown`, `None`, `N/A`,
`Not specified`, common generic system/board names, all-X values, and common
decimal placeholder serials.

## Identity quality

- `STRONG`: a valid SMBIOS System UUID plus at least two usable system/board
  descriptors.
- `MODERATE`: a valid UUID with incomplete descriptors; or usable hashed
  platform serial identity plus descriptors; or at least three usable platform
  descriptors.
- `FALLBACK`: SMBIOS cannot confidently establish physical identity. When
  available, Sidecar uses a hashed Windows MachineGuid and clearly labels that
  identity as dependent on the Windows installation. Reinstallation or cloning
  may therefore change or duplicate it. If even that source is unavailable, the
  collision/stability warning is explicit.

Sidecar never invents a serial or upgrades fallback evidence to strong identity.

## Explicit exclusions

The following never participate in Machine Identity V1:

- installed, available, or utilized RAM;
- GPU model, UUID, telemetry, clocks, drivers, or CUDA version;
- storage identity, capacity, inventory, volumes, or topology;
- OS version/build, host/network name, or benchmark data;
- negotiated PCIe generation or width.

Therefore RAM upgrades, GPU replacement, NVMe changes, and ordinary driver/OS
updates do not alter the hash. A deliberate future identity-format change must
increment the identity version rather than silently changing V1.

