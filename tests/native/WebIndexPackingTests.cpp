#include "WebIndexPacking.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
template <typename Exception, typename Callback> void RequireRejected(Callback callback)
{
    bool rejected = false;
    try {
        callback();
    } catch (const Exception &) {
        rejected = true;
    }
    assert(rejected);
}
} // namespace

int main()
{
    using infernux::MeshIndexFormat;
    using infernux::web::PackWebIndexRange;
    using infernux::web::WebPackedIndexStreams;

    WebPackedIndexStreams streams;
    const std::vector<uint32_t> small{0, 2, 1};
    const auto first16 = PackWebIndexRange(streams, MeshIndexFormat::UInt16, 70000, 3, small, 0, small.size());
    assert(first16.format == MeshIndexFormat::UInt16);
    assert(first16.firstIndex == 0 && first16.indexCount == 3 && first16.baseVertex == 70000);
    assert((streams.uint16Indices == std::vector<uint16_t>{0, 2, 1, 0}));
    assert(streams.uint32Indices.empty());

    const auto second16 = PackWebIndexRange(streams, MeshIndexFormat::Auto, 4, 3, small, 1, 2);
    assert(second16.format == MeshIndexFormat::UInt16);
    assert(second16.firstIndex == 4 && second16.indexCount == 2 && second16.baseVertex == 4);
    assert((streams.uint16Indices == std::vector<uint16_t>{0, 2, 1, 0, 2, 1}));
    assert((streams.uint16Indices.size() * sizeof(uint16_t)) % 4U == 0);

    std::vector<uint32_t> large(65537);
    large.back() = 65536;
    const auto first32 = PackWebIndexRange(streams, MeshIndexFormat::Auto, 8, large.size(), large, large.size() - 1, 1);
    assert(first32.format == MeshIndexFormat::UInt32);
    assert(first32.firstIndex == 0 && first32.indexCount == 1 && first32.baseVertex == 8);
    assert(streams.uint32Indices == std::vector<uint32_t>{65536});

    // A hot-reloaded draw that changes format publishes into the other stream;
    // neither its offset nor its bytes can alias the prior UInt16 publication.
    const auto reloaded32 = PackWebIndexRange(streams, MeshIndexFormat::UInt32, 12, 3, small, 0, small.size());
    assert(reloaded32.format == MeshIndexFormat::UInt32);
    assert(reloaded32.firstIndex == 1 && reloaded32.indexCount == 3);
    assert((streams.uint32Indices == std::vector<uint32_t>{65536, 0, 2, 1}));
    assert((streams.uint16Indices == std::vector<uint16_t>{0, 2, 1, 0, 2, 1}));

    const auto before16 = streams.uint16Indices;
    const auto before32 = streams.uint32Indices;
    RequireRejected<std::invalid_argument>(
        [&] { PackWebIndexRange(streams, MeshIndexFormat::UInt16, 0, large.size(), large, large.size() - 1, 1); });
    assert(streams.uint16Indices == before16 && streams.uint32Indices == before32);

    RequireRejected<std::invalid_argument>(
        [&] { PackWebIndexRange(streams, MeshIndexFormat::UInt32, 0, 3, small, 2, 2); });
    assert(streams.uint16Indices == before16 && streams.uint32Indices == before32);

    const std::vector<uint32_t> outOfRange{3};
    RequireRejected<std::invalid_argument>(
        [&] { PackWebIndexRange(streams, MeshIndexFormat::Auto, 0, 3, outOfRange, 0, 1); });
    assert(streams.uint16Indices == before16 && streams.uint32Indices == before32);

    RequireRejected<std::overflow_error>([&] {
        PackWebIndexRange(streams, MeshIndexFormat::UInt32,
                          static_cast<size_t>(std::numeric_limits<int32_t>::max()) + 1U, 3, small, 0, 3);
    });
    assert(streams.uint16Indices == before16 && streams.uint32Indices == before32);

    const std::vector<uint32_t> maximumIndex{std::numeric_limits<uint32_t>::max()};
    RequireRejected<std::overflow_error>([&] {
        PackWebIndexRange(streams, MeshIndexFormat::UInt32, 1,
                          static_cast<size_t>(std::numeric_limits<uint32_t>::max()) + 1U, maximumIndex, 0, 1);
    });
    assert(streams.uint16Indices == before16 && streams.uint32Indices == before32);

    std::cout << "Web index packing tests passed\n";
    return 0;
}
