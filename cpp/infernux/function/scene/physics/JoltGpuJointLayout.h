#pragma once

#include <algorithm>
#include <cstring>
#include <function/scene/physics/gpu/PhysicsTypes.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace infernux
{

// Immutable address facts for one accepted compact stream generation. Word
// offsets are resource-local; stable identities never retain these addresses.
class JoltGpuJointLayout final
{
  public:
    using Directory = JPH::GPU::JointDirectory;
    using Author = JPH::GPU::JointAuthor;
    using Word = JPH::GPU::QueryRecordWord;
    using UInt4 = JPH::GPU::UInt4;
    struct Slice
    {
        uint32_t first = 0, count = 0;
        uint32_t authorFirst = 0, authorWords = 0, stateFirst = 0, stateWords = 0;
    };

    static uint32_t AuthorWords(uint32_t kind)
    {
        using namespace JPH::GPU;
        switch (kind) {
        case JointFixed:
            return JointFixedAuthorWords;
        case JointHinge:
            return JointHingeAuthorWords;
        case JointSlider:
            return JointSliderAuthorWords;
        case JointDistance:
            return JointDistanceAuthorWords;
        case JointSwingTwist:
            return JointSwingAuthorWords;
        case JointSixDOF:
            return JointSixAuthorWords;
        default:
            throw std::invalid_argument("Unknown compact joint author kind");
        }
    }
    static uint32_t StateWords(uint32_t kind)
    {
        using namespace JPH::GPU;
        switch (kind) {
        case JointFixed:
            return JointFixedStateWords;
        case JointHinge:
            return JointHingeStateWords;
        case JointSlider:
            return JointSliderStateWords;
        case JointDistance:
            return JointDistanceStateWords;
        case JointSwingTwist:
            return JointSwingStateWords;
        case JointSixDOF:
            return JointSixStateWords;
        default:
            throw std::invalid_argument("Unknown compact joint state kind");
        }
    }
    explicit JoltGpuJointLayout(const std::vector<Author> &authors)
    {
        const uint64_t count = authors.size();
        uint64_t authorCursor = 1u + count * JPH::GPU::JointDirectoryWords;
        Check(authorCursor);
        m_rows.reserve(authors.size());
        for (const auto &a : authors) {
            if (!a.HasMatchingKind())
                throw std::invalid_argument("Compact joint kind does not match its typed author");
            m_rows.push_back({a.bodies, a.identity, {Checked(authorCursor), 0u, 0u, 0u}});
            authorCursor += AuthorWords(a.bodies.z);
            Check(authorCursor);
        }
        m_authorEnd = Checked(authorCursor);
        uint64_t stateCursor = authorCursor;
        for (auto &row : m_rows) {
            row.layout.y = Checked(stateCursor);
            stateCursor += StateWords(row.bodies.z);
            Check(stateCursor);
        }
        m_total = Checked(stateCursor);
    }
    [[nodiscard]] UInt4 Header() const
    {
        return {static_cast<uint32_t>(m_rows.size()), m_authorEnd, m_total, 0u};
    }
    [[nodiscard]] uint64_t ByteSize() const
    {
        return uint64_t(m_total) * 16u;
    }
    [[nodiscard]] uint64_t AuthorEndBytes() const
    {
        return uint64_t(m_authorEnd) * 16u;
    }
    [[nodiscard]] const std::vector<Directory> &Rows() const
    {
        return m_rows;
    }
    [[nodiscard]] const Directory &Row(uint32_t row) const
    {
        return m_rows.at(row);
    }

    template <class Changes>
    [[nodiscard]] std::shared_ptr<const JoltGpuJointLayout> WithAuthorUpdates(const Changes &changes) const
    {
        auto result = std::shared_ptr<JoltGpuJointLayout>(new JoltGpuJointLayout(*this));
        for (const auto &change : changes) {
            auto &row = result->m_rows.at(change.index);
            if (!change.author.HasMatchingKind() || row.bodies.z != change.author.bodies.z)
                throw std::invalid_argument("In-place joint update cannot change payload kind or size");
            row.bodies = change.author.bodies;
            row.identity = change.author.identity;
        }
        return result;
    }

    static std::vector<uint8_t> AuthorBytes(const Author &a)
    {
        if (!a.HasMatchingKind())
            throw std::invalid_argument("Compact joint author kind mismatch");
        return std::visit(
            [](const auto &payload) {
                std::vector<uint8_t> bytes(sizeof(payload));
                std::memcpy(bytes.data(), &payload, bytes.size());
                return bytes;
            },
            a.payload);
    }
    [[nodiscard]] std::vector<uint8_t> MetadataBytes() const
    {
        std::vector<uint8_t> bytes(16u + m_rows.size() * sizeof(Directory));
        const auto header = Header();
        std::memcpy(bytes.data(), &header, sizeof(header));
        if (!m_rows.empty())
            std::memcpy(bytes.data() + 16u, m_rows.data(), m_rows.size() * sizeof(Directory));
        return bytes;
    }
    [[nodiscard]] std::vector<uint8_t> PrefixBytes(const std::vector<Author> &authors) const
    {
        if (authors.size() != m_rows.size())
            throw std::invalid_argument("Joint prefix requires its exact author catalog");
        auto bytes = MetadataBytes();
        bytes.resize(static_cast<size_t>(AuthorEndBytes()));
        for (uint32_t i = 0; i < authors.size(); ++i) {
            const auto &a = authors[i];
            const auto &r = m_rows[i];
            if (std::memcmp(&a.bodies, &r.bodies, sizeof(UInt4)) ||
                std::memcmp(&a.identity, &r.identity, sizeof(UInt4)))
                throw std::invalid_argument("Joint prefix metadata differs from captured layout");
            const auto payload = AuthorBytes(a);
            if (payload.size() != uint64_t(AuthorWords(a.bodies.z)) * 16u)
                throw std::logic_error("Shared typed joint author size mismatch");
            std::memcpy(bytes.data() + uint64_t(r.layout.x) * 16u, payload.data(), payload.size());
        }
        return bytes;
    }
    [[nodiscard]] Slice WorldSlice(uint32_t first, uint32_t count) const
    {
        if (first > m_rows.size() || count > m_rows.size() - first)
            throw std::out_of_range("Joint snapshot range exceeds accepted catalog");
        if (!count)
            return {first, 0u, m_authorEnd, 0u, m_total, 0u};
        const auto &begin = m_rows[first];
        const auto &last = m_rows[first + count - 1u];
        return {first,          count,
                begin.layout.x, last.layout.x + AuthorWords(last.bodies.z) - begin.layout.x,
                begin.layout.y, last.layout.y + StateWords(last.bodies.z) - begin.layout.y};
    }
    [[nodiscard]] std::shared_ptr<const JoltGpuJointLayout> LocalLayout(Slice slice) const
    {
        // Only metadata is rebased. The snapshot copies unchanged author and
        // numerical payloads from the exact selected-world live generation.
        auto result = std::shared_ptr<JoltGpuJointLayout>(new JoltGpuJointLayout());
        result->m_rows.assign(m_rows.begin() + slice.first, m_rows.begin() + slice.first + slice.count);
        const uint32_t authorFirst = 1u + JPH::GPU::JointDirectoryWords * slice.count;
        result->m_authorEnd = authorFirst + slice.authorWords;
        result->m_total = result->m_authorEnd + slice.stateWords;
        for (auto &row : result->m_rows) {
            row.layout.x = authorFirst + (row.layout.x - slice.authorFirst);
            row.layout.y = result->m_authorEnd + (row.layout.y - slice.stateFirst);
        }
        return result;
    }

  private:
    JoltGpuJointLayout() = default;
    static void Check(uint64_t words)
    {
        if (words > std::numeric_limits<uint32_t>::max() || words > std::numeric_limits<size_t>::max() / 16u)
            throw std::length_error("Compact joint stream exceeds word/address range");
    }
    static uint32_t Checked(uint64_t value)
    {
        Check(value);
        return static_cast<uint32_t>(value);
    }
    std::vector<Directory> m_rows;
    uint32_t m_authorEnd = 1, m_total = 1;
};

} // namespace infernux
