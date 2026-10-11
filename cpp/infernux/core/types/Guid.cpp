#include "Guid.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <random>

namespace infernux
{
std::string GenerateGuid()
{
    static std::mutex mutex;
    static std::mt19937_64 generator = [] {
        std::random_device device;
        std::array<uint32_t, 10> entropy{};
        for (auto &value : entropy)
            value = device();
        std::seed_seq seed(entropy.begin(), entropy.end());
        return std::mt19937_64(seed);
    }();

    std::array<uint64_t, 2> words;
    {
        std::lock_guard lock(mutex);
        words = {generator(), generator()};
    }
    // UUID v4 version/variant bits also exclude the reserved all-zero identity.
    words[0] = (words[0] & 0xffffffffffff0fffULL) | 0x4000ULL;
    words[1] = (words[1] & 0x3fffffffffffffffULL) | 0x8000000000000000ULL;
    constexpr char digits[] = "0123456789abcdef";
    std::string result(32, '0');
    for (size_t word = 0; word < words.size(); ++word)
        for (size_t digit = 0; digit < 16; ++digit)
            result[word * 16 + digit] = digits[(words[word] >> ((15 - digit) * 4)) & 0xf];
    return result;
}
} // namespace infernux
