#include "sidecar/hardware/identity.hpp"

#include "sidecar/core/sha256.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_set>

namespace sidecar::hardware {
namespace {

constexpr std::string_view kMissing = "<missing>";

std::string NormalizeText(std::string_view input) {
    std::string output;
    output.reserve(input.size());
    bool pending_space = false;
    for (const unsigned char character : input) {
        if (std::isspace(character) != 0) {
            pending_space = !output.empty();
            continue;
        }
        if (pending_space) {
            output.push_back(' ');
            pending_space = false;
        }
        output.push_back(static_cast<char>(std::tolower(character)));
    }
    return output;
}

bool IsPlaceholder(std::string_view normalized) {
    static const std::unordered_set<std::string> kPlaceholders{
        "", "to be filled by o.e.m.", "to be filled by oem", "default string",
        "default", "unknown", "none", "n/a", "not applicable", "not specified",
        "system manufacturer", "system product name", "base board product name",
        "o.e.m.", "oem", "xxxxxxxx", "123456789", "0123456789",
    };
    return kPlaceholders.contains(std::string(normalized));
}

std::string Field(const std::optional<std::string>& value) {
    return value.value_or(std::string(kMissing));
}

}  // namespace

const char* ToString(IdentityQuality quality) noexcept {
    switch (quality) {
        case IdentityQuality::Strong:
            return "STRONG";
        case IdentityQuality::Moderate:
            return "MODERATE";
        case IdentityQuality::Fallback:
            return "FALLBACK";
    }
    return "FALLBACK";
}

const char* ToString(DiscoveryConfidence confidence) noexcept {
    switch (confidence) {
        case DiscoveryConfidence::DirectlyReported:
            return "DIRECTLY_REPORTED";
        case DiscoveryConfidence::Derived:
            return "DERIVED";
        case DiscoveryConfidence::Unknown:
            return "UNKNOWN";
    }
    return "UNKNOWN";
}

std::optional<std::string> NormalizeIdentityValue(const std::optional<std::string>& value) {
    if (!value.has_value()) {
        return std::nullopt;
    }
    const std::string normalized = NormalizeText(*value);
    if (IsPlaceholder(normalized)) {
        return std::nullopt;
    }
    return normalized;
}

std::optional<std::string> NormalizeSystemUuid(const std::optional<std::string>& value) {
    const auto normalized_value = NormalizeIdentityValue(value);
    if (!normalized_value.has_value()) {
        return std::nullopt;
    }

    std::string hex;
    for (const unsigned char character : *normalized_value) {
        if (std::isxdigit(character) != 0) {
            hex.push_back(static_cast<char>(std::tolower(character)));
        } else if (character != '-' && character != '{' && character != '}') {
            return std::nullopt;
        }
    }
    if (hex.size() != 32U ||
        std::all_of(hex.begin(), hex.end(), [](char value) { return value == '0'; }) ||
        std::all_of(hex.begin(), hex.end(), [](char value) { return value == 'f'; })) {
        return std::nullopt;
    }

    return hex.substr(0, 8) + "-" + hex.substr(8, 4) + "-" + hex.substr(12, 4) +
           "-" + hex.substr(16, 4) + "-" + hex.substr(20, 12);
}

MachineIdentity BuildMachineIdentity(const MachineIdentityInput& input) {
    const auto system_uuid = NormalizeSystemUuid(input.system_uuid);
    const auto system_vendor = NormalizeIdentityValue(input.system_manufacturer);
    const auto system_product = NormalizeIdentityValue(input.system_product);
    const auto board_vendor = NormalizeIdentityValue(input.board_manufacturer);
    const auto board_product = NormalizeIdentityValue(input.board_product);
    const auto system_serial = NormalizeIdentityValue(input.system_serial);
    const auto board_serial = NormalizeIdentityValue(input.board_serial);
    const auto fallback_id = NormalizeIdentityValue(input.fallback_installation_id);

    std::optional<std::string> platform_serial_hash;
    if (system_serial.has_value() || board_serial.has_value()) {
        const std::string serial_material =
            "SIDECAR-PLATFORM-SERIAL-V1\n"
            "system_serial=" + Field(system_serial) + "\n"
            "board_serial=" + Field(board_serial) + "\n";
        platform_serial_hash = core::Sha256Hex(serial_material);
    }

    std::optional<std::string> fallback_hash;
    if (!system_uuid.has_value() && !platform_serial_hash.has_value() && fallback_id.has_value()) {
        fallback_hash = core::Sha256Hex("SIDECAR-FALLBACK-INSTALLATION-V1\nvalue=" +
                                        *fallback_id + "\n");
    }

    MachineIdentity identity;
    const std::size_t descriptor_count =
        static_cast<std::size_t>(system_vendor.has_value()) +
        static_cast<std::size_t>(system_product.has_value()) +
        static_cast<std::size_t>(board_vendor.has_value()) +
        static_cast<std::size_t>(board_product.has_value());

    if (system_uuid.has_value() && descriptor_count >= 2U) {
        identity.quality = IdentityQuality::Strong;
        identity.basis.emplace_back("smbios_system_uuid");
        identity.basis.emplace_back("platform_descriptors");
        if (platform_serial_hash.has_value()) {
            identity.basis.emplace_back("hashed_platform_serial");
        }
    } else if (system_uuid.has_value() ||
               (platform_serial_hash.has_value() && descriptor_count >= 1U) ||
               descriptor_count >= 3U) {
        identity.quality = IdentityQuality::Moderate;
        if (system_uuid.has_value()) {
            identity.basis.emplace_back("smbios_system_uuid");
        }
        if (platform_serial_hash.has_value()) {
            identity.basis.emplace_back("hashed_platform_serial");
        }
        if (descriptor_count > 0U) {
            identity.basis.emplace_back("platform_descriptors");
        }
    } else {
        identity.quality = IdentityQuality::Fallback;
        if (fallback_hash.has_value()) {
            identity.basis.emplace_back("hashed_windows_installation_id");
            identity.warnings.emplace_back(
                "SMBIOS identity was insufficient; stability now depends on the Windows installation identity");
        } else if (platform_serial_hash.has_value()) {
            identity.basis.emplace_back("hashed_platform_serial");
            identity.basis.emplace_back("insufficient_platform_identity");
            identity.warnings.emplace_back(
                "A platform serial was available without enough corroborating SMBIOS identity; stability confidence is reduced");
        } else {
            identity.basis.emplace_back("insufficient_platform_identity");
            identity.warnings.emplace_back(
                "No confident stable platform identity was available; collision and stability risk is elevated");
        }
    }

    identity.canonical_material =
        "SIDECAR-MACHINE-ID-V1\n"
        "system_uuid=" + Field(system_uuid) + "\n"
        "system_vendor=" + Field(system_vendor) + "\n"
        "system_product=" + Field(system_product) + "\n"
        "board_vendor=" + Field(board_vendor) + "\n"
        "board_product=" + Field(board_product) + "\n"
        "platform_serial_hash=" + Field(platform_serial_hash) + "\n"
        "fallback_installation_hash=" + Field(fallback_hash) + "\n";
    identity.machine_hash = core::Sha256Hex(identity.canonical_material);
    return identity;
}

HardwareDiscoveryService::HardwareDiscoveryService(IHardwareDiscoveryProvider& provider) noexcept
    : provider_(provider) {}

DiscoveryReport HardwareDiscoveryService::Discover() const {
    RawDiscovery raw = provider_.Discover();
    DiscoveryReport report;
    report.identity = BuildMachineIdentity(raw.identity_input);
    report.snapshot = std::move(raw.snapshot);
    report.warnings = std::move(raw.warnings);
    report.warnings.insert(report.warnings.end(), report.identity.warnings.begin(),
                           report.identity.warnings.end());
    return report;
}

}  // namespace sidecar::hardware
