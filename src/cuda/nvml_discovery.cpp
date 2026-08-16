#include "sidecar/hardware/native_components.hpp"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <nvml.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <string>

namespace sidecar::hardware {
namespace {

template <typename Function>
Function LoadFunction(HMODULE module, const char* name) {
    return reinterpret_cast<Function>(GetProcAddress(module, name));
}

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

}  // namespace

void EnrichNvidiaDevicesWithNvml(MachineSnapshot& snapshot,
                                 std::vector<std::string>& warnings) {
    HMODULE module = LoadLibraryW(L"nvml.dll");
    if (module == nullptr) {
        module = LoadLibraryW(L"C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    }
    if (module == nullptr) {
        snapshot.nvml_available = false;
        warnings.emplace_back("NVML runtime was not available; NVIDIA telemetry fields are unknown");
        return;
    }

    using InitFn = nvmlReturn_t (*)();
    using ShutdownFn = nvmlReturn_t (*)();
    using DriverVersionFn = nvmlReturn_t (*)(char*, unsigned int);
    using CountFn = nvmlReturn_t (*)(unsigned int*);
    using HandleFn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
    using StringFn = nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int);
    using MemoryFn = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*);
    using PciFn = nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t*);
    using UIntFn = nvmlReturn_t (*)(nvmlDevice_t, unsigned int*);
    using TemperatureFn = nvmlReturn_t (*)(nvmlDevice_t, nvmlTemperatureSensors_t,
                                           unsigned int*);
    using ClockFn = nvmlReturn_t (*)(nvmlDevice_t, nvmlClockType_t, unsigned int*);

    const auto init = LoadFunction<InitFn>(module, "nvmlInit_v2");
    const auto shutdown = LoadFunction<ShutdownFn>(module, "nvmlShutdown");
    const auto driver_version =
        LoadFunction<DriverVersionFn>(module, "nvmlSystemGetDriverVersion");
    const auto get_count = LoadFunction<CountFn>(module, "nvmlDeviceGetCount_v2");
    const auto get_handle =
        LoadFunction<HandleFn>(module, "nvmlDeviceGetHandleByIndex_v2");
    const auto get_uuid = LoadFunction<StringFn>(module, "nvmlDeviceGetUUID");
    const auto get_name = LoadFunction<StringFn>(module, "nvmlDeviceGetName");
    const auto get_memory = LoadFunction<MemoryFn>(module, "nvmlDeviceGetMemoryInfo");
    const auto get_pci = LoadFunction<PciFn>(module, "nvmlDeviceGetPciInfo_v3");
    const auto get_current_gen =
        LoadFunction<UIntFn>(module, "nvmlDeviceGetCurrPcieLinkGeneration");
    const auto get_max_gen =
        LoadFunction<UIntFn>(module, "nvmlDeviceGetMaxPcieLinkGeneration");
    const auto get_current_width =
        LoadFunction<UIntFn>(module, "nvmlDeviceGetCurrPcieLinkWidth");
    const auto get_max_width =
        LoadFunction<UIntFn>(module, "nvmlDeviceGetMaxPcieLinkWidth");
    const auto get_power_limit =
        LoadFunction<UIntFn>(module, "nvmlDeviceGetPowerManagementLimit");
    const auto get_power_usage =
        LoadFunction<UIntFn>(module, "nvmlDeviceGetPowerUsage");
    const auto get_temperature =
        LoadFunction<TemperatureFn>(module, "nvmlDeviceGetTemperature");
    const auto get_clock = LoadFunction<ClockFn>(module, "nvmlDeviceGetClockInfo");

    if (init == nullptr || shutdown == nullptr || get_count == nullptr ||
        get_handle == nullptr || get_uuid == nullptr) {
        FreeLibrary(module);
        warnings.emplace_back("NVML runtime is missing required discovery entry points");
        return;
    }
    if (init() != NVML_SUCCESS) {
        FreeLibrary(module);
        warnings.emplace_back("NVML initialization failed; NVIDIA telemetry fields are unknown");
        return;
    }

    struct NvmlGuard {
        ShutdownFn shutdown;
        HMODULE module;
        ~NvmlGuard() {
            shutdown();
            FreeLibrary(module);
        }
    } guard{shutdown, module};

    snapshot.nvml_available = true;
    if (driver_version != nullptr) {
        std::array<char, NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE> buffer{};
        if (driver_version(buffer.data(), static_cast<unsigned int>(buffer.size())) == NVML_SUCCESS) {
            snapshot.nvidia_driver_version = buffer.data();
        }
    }

    unsigned int count = 0;
    if (get_count(&count) != NVML_SUCCESS) {
        warnings.emplace_back("NVML could not enumerate NVIDIA devices");
        return;
    }

    for (unsigned int index = 0; index < count; ++index) {
        nvmlDevice_t handle = nullptr;
        if (get_handle(index, &handle) != NVML_SUCCESS) continue;

        std::array<char, NVML_DEVICE_UUID_BUFFER_SIZE> uuid_buffer{};
        if (get_uuid(handle, uuid_buffer.data(),
                     static_cast<unsigned int>(uuid_buffer.size())) != NVML_SUCCESS) {
            continue;
        }
        const std::string uuid = uuid_buffer.data();
        const std::string comparable_uuid = Lower(uuid);
        auto match = std::find_if(snapshot.gpus.begin(), snapshot.gpus.end(),
                                  [&](const GpuInfo& gpu) {
                                      return Lower(gpu.persistent_id) == comparable_uuid;
                                  });
        if (match == snapshot.gpus.end()) {
            GpuInfo gpu;
            gpu.persistent_id = uuid;
            gpu.enriched_by_nvml = true;
            snapshot.gpus.push_back(std::move(gpu));
            match = std::prev(snapshot.gpus.end());
        }
        GpuInfo& gpu = *match;
        gpu.enriched_by_nvml = true;

        if (get_name != nullptr) {
            std::array<char, NVML_DEVICE_NAME_BUFFER_SIZE> name{};
            if (get_name(handle, name.data(), static_cast<unsigned int>(name.size())) ==
                NVML_SUCCESS && gpu.model.empty()) {
                gpu.model = name.data();
            }
        }
        if (get_memory != nullptr) {
            nvmlMemory_t memory{};
            if (get_memory(handle, &memory) == NVML_SUCCESS) {
                if (!gpu.vram_bytes.has_value()) gpu.vram_bytes = memory.total;
                gpu.vram_used_bytes = memory.used;
            }
        }
        if (get_pci != nullptr) {
            nvmlPciInfo_t pci{};
            if (get_pci(handle, &pci) == NVML_SUCCESS) {
                gpu.pci_domain = pci.domain;
                gpu.pci_bus = pci.bus;
                gpu.pci_device = pci.device;
            }
        }

        const auto read_uint = [&](UIntFn function, std::optional<std::uint32_t>& target) {
            if (function == nullptr) return;
            unsigned int value = 0;
            if (function(handle, &value) == NVML_SUCCESS) target = value;
        };
        read_uint(get_current_gen, gpu.negotiated_pcie_generation);
        read_uint(get_max_gen, gpu.maximum_pcie_generation);
        read_uint(get_current_width, gpu.negotiated_lane_width);
        read_uint(get_max_width, gpu.maximum_lane_width);

        if (get_power_limit != nullptr) {
            unsigned int milliwatts = 0;
            if (get_power_limit(handle, &milliwatts) == NVML_SUCCESS) {
                gpu.power_limit_watts = static_cast<double>(milliwatts) / 1000.0;
            }
        }
        if (get_power_usage != nullptr) {
            unsigned int milliwatts = 0;
            if (get_power_usage(handle, &milliwatts) == NVML_SUCCESS) {
                gpu.power_draw_watts = static_cast<double>(milliwatts) / 1000.0;
            }
        }
        if (get_temperature != nullptr) {
            unsigned int temperature = 0;
            if (get_temperature(handle, NVML_TEMPERATURE_GPU, &temperature) == NVML_SUCCESS) {
                gpu.temperature_c = static_cast<double>(temperature);
            }
        }
        if (get_clock != nullptr) {
            unsigned int clock = 0;
            if (get_clock(handle, NVML_CLOCK_GRAPHICS, &clock) == NVML_SUCCESS)
                gpu.graphics_clock_mhz = clock;
            if (get_clock(handle, NVML_CLOCK_MEM, &clock) == NVML_SUCCESS)
                gpu.memory_clock_mhz = clock;
        }
    }
}

}  // namespace sidecar::hardware
