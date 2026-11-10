#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace infernux
{
/// A pure description of one selected native IO attempt, not an execution plan.
/// bytes is its admitted byte allowance; zero denotes one indivisible action.
/// remainingBytes describes the currently available span, not the whole request.
/// parameter separates backend costs such as codec level or parser workspace.
struct IoWork
{
    std::string_view operation;
    size_t bytes = 0;
    std::optional<uint64_t> remainingBytes;
    uint64_t parameter = 0;
};

inline IoWork IoByteWork(std::string_view operation, uint64_t remaining, size_t maximum, uint64_t parameter = 0)
{
    return {operation, static_cast<size_t>(std::min<uint64_t>(remaining, maximum)), remaining, parameter};
}

/// Pure remaining-work facts. bytes/input/output are exact extents;
/// upper_bytes is a byte bound. A codec's outputBytes is an optional bound,
/// never its predicted callback count. Disengaged extents mean missing facts;
/// zero is known, and a zero-byte state transition remains an atom.
/// entryIndex identifies the original immutable archive directory entry.
/// finishOperation is a byte span's terminal action, or a codec's possible
/// terminal callback. Codec completion/output handoffs use observed callbacks;
/// a possible finish is not an unconditional extra callback or frame.
/// entryBytes is the immutable full decoded extent for a codec's entry. In
/// particular an empty-input final drain still depends on that complete entry.
/// worker is a measured multi-callback span, with no host output handoffs. Its
/// callback count is not bounded by its byte extent (including empty entries).
/// bytes_fragment/codec_fragment describe one supplied input block. Their cost
/// class uses that original block, separately from a tiny final input block.
struct IoSpan
{
    std::string_view kind;
    std::string_view operation;
    std::optional<size_t> entryIndex;
    uint64_t parameter = 0;
    std::optional<uint64_t> inputBytes;
    std::optional<uint64_t> outputBytes;
    std::string_view finishOperation;
    std::optional<uint64_t> entryBytes;
};

/// A repeated group describes input/codec or decode/output interleaving;
/// repeats is a count of these groups, not a codec callback count.
using IoRecipe = std::vector<std::pair<std::vector<IoSpan>, uint64_t>>;
} // namespace infernux
