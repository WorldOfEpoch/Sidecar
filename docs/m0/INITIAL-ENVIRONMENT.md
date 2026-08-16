# Initial development environment

Observed on 2026-08-15 before the M0 bootstrap build. Unknown or unavailable
tools are recorded rather than inferred.

| Component | Observed value |
| --- | --- |
| OS | Windows 11 Home, version `10.0.26200`, build `26200`, x64 |
| CPU | Intel Core Ultra 9 285K, 24 physical / 24 logical cores reported |
| RAM | 99,801,816 KiB visible to Windows (approximately 95.2 GiB) |
| GPU | NVIDIA GeForce RTX 3090, 24,576 MiB VRAM |
| GPU UUID | Redacted from the public snapshot (unique hardware identifier) |
| GPU PCI ID | `00000000:02:00.0` |
| NVIDIA driver | `591.86` |
| CUDA toolkit | `12.8`, nvcc `12.8.93` |
| CMake | `4.2.3` |
| Git | `2.53.0.windows.1` |
| Configured compiler | MSVC `19.38.33144.0`, x64 |
| Windows SDK | `10.0.22621.0` |
| SQLite CLI/development package | Not found on the shell path |
| SQLite used by Sidecar | Official amalgamation `3.53.4`, hash-pinned by CMake |

The GPU was at 37 C and idle P8 state during discovery. These are environment
observations, not benchmark results. CUDA and CUPTI components remain disabled
in the bootstrap build until their M0 implementations exist.

