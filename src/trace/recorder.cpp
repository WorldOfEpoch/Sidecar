#include "sidecar/trace/recorder.hpp"

#include <ctime>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace sidecar::trace {

std::uint64_t CurrentThreadCpuTimeNs() noexcept {
#ifdef _WIN32
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user)) {
        ULARGE_INTEGER kernel_time{}, user_time{};
        kernel_time.LowPart = kernel.dwLowDateTime;
        kernel_time.HighPart = kernel.dwHighDateTime;
        user_time.LowPart = user.dwLowDateTime;
        user_time.HighPart = user.dwHighDateTime;
        return (kernel_time.QuadPart + user_time.QuadPart) * 100ULL;
    }
#endif
    return static_cast<std::uint64_t>(
        (static_cast<long double>(std::clock()) * 1000000000.0L) / CLOCKS_PER_SEC);
}

}  // namespace sidecar::trace
