#pragma once

#include "RayGeometry.h"
#include <cstring>

namespace infernux::raytracing
{

struct RayEmissiveSource
{
    uint32_t surfaceId;
    uint32_t triangleStart;
    uint32_t triangleCount;
    float power;
};

// O(instances + emissive triangles), entirely on the existing budgeted build.
// A surface owns one contiguous BLAS slice, giving reciprocal lookup O(1).
class RayEmissiveBuild
{
  public:
    RayEmissiveBuild(const std::vector<RayEmissiveSource> &sources, const std::vector<float> &areas,
                     RayInstancePage &page, size_t surfaceCount)
        : m_sources(sources), m_areas(areas), m_page(page), m_done(sources.empty())
    {
        if (!m_done) {
            page.emissiveRanges.resize(surfaceCount);
            size_t count = 0;
            for (const auto &source : sources)
                count += source.triangleCount;
            page.emissiveTriangles.reserve(count);
        }
    }

    bool Advance(uint32_t budget)
    {
        while (budget && !m_done) {
            if (m_instance < m_page.instances.size()) {
                m_page.emissiveRanges[m_page.instances[m_instance].surfaceId][3] = static_cast<uint32_t>(m_instance);
                ++m_instance;
            } else if (m_source < m_sources.size()) {
                const auto &source = m_sources[m_source];
                auto &range = m_page.emissiveRanges[source.surfaceId];
                if (!m_triangle) {
                    range[0] = static_cast<uint32_t>(m_page.emissiveTriangles.size());
                    range[1] = source.triangleCount;
                    range[2] = source.triangleStart;
                }
                const auto triangle = source.triangleStart + static_cast<uint32_t>(m_triangle);
                m_total += double(source.power) * m_areas[triangle];
                m_page.emissiveTriangles.push_back({range[3], triangle, source.surfaceId, 0u});
                if (++m_triangle == source.triangleCount) {
                    ++m_source;
                    m_triangle = 0;
                }
            } else if (m_normalize < m_page.emissiveTriangles.size()) {
                auto &record = m_page.emissiveTriangles[m_normalize];
                const auto &source = m_sources[m_normalizeSource];
                m_prefix += double(source.power) * m_areas[record[1]];
                const auto count = m_page.emissiveTriangles.size();
                const float cdf = m_total > 0.0 ? static_cast<float>(m_prefix / m_total)
                                                : static_cast<float>(double(m_normalize + 1) / count);
                record[3] = Bits(cdf);
                ++m_normalize;
                if (++m_normalizeTriangle == source.triangleCount) {
                    ++m_normalizeSource;
                    m_normalizeTriangle = 0;
                }
            } else {
                m_done = true;
            }
            --budget;
        }
        return m_done;
    }

  private:
    static uint32_t Bits(float value)
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    }
    const std::vector<RayEmissiveSource> &m_sources;
    const std::vector<float> &m_areas;
    RayInstancePage &m_page;
    size_t m_instance = 0, m_source = 0, m_triangle = 0, m_normalize = 0;
    size_t m_normalizeSource = 0, m_normalizeTriangle = 0;
    double m_total = 0.0, m_prefix = 0.0;
    bool m_done;
};

} // namespace infernux::raytracing
