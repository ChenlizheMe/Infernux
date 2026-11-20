// Internal pair-local numerical-range solves. No entry point or device storage.
// Rank is resolved from geometrically scaled allowed DOFs at float32 precision;
// it is not an exact rank certificate for every positive stored-float mode.
#ifndef JPH_GPU_RIGID_PSD_SLANG
#define JPH_GPU_RIGID_PSD_SLANG

struct JphPsdRange
{
    float3 axis[3];
    uint rank;
    bool valid;
};

JphPsdRange jph_psd_range()
{
    JphPsdRange basis;
    basis.axis[0] = basis.axis[1] = basis.axis[2] = float3(0.0f);
    basis.rank = 0u;
    basis.valid = true;
    return basis;
}

struct JphPsdFactors
{
    float3 column[12];
    uint count;
    uint upperBound;
    bool valid;
};

JphPsdFactors jph_psd_factors()
{
    JphPsdFactors factors;
    factors.count = 0u;
    factors.upperBound = 3u;
    factors.valid = true;
    return factors;
}

// Positive weights do not affect exact range. Resolve geometric directions
// before applying mass/inertia weights, preserving independent low-response DOFs.
// The caller normalizes arms by their own body arm scale; do not turn a
// cancellation-sized cross product into a reliable unit direction. The reduced
// solve still uses the original weighted physical mobility matrix.
void jph_psd_add(inout JphPsdFactors factors, float3 column)
{
    if (!all(isfinite(column))) { factors.valid = false; return; }
    float magnitude = max(abs(column.x), max(abs(column.y), abs(column.z)));
    if (magnitude == 0.0f) return;
    // Exact duplicate directions need not participate in pivoting twice.
    for (uint i = 0u; i < factors.count; ++i)
        if (all(column == factors.column[i]) || all(column == -factors.column[i])) return;
    if (factors.count == 12u) { factors.valid = false; return; }
    factors.column[factors.count++] = column;
}

JphPsdRange jph_psd_finish(JphPsdFactors factors, uint dimension)
{
    JphPsdRange basis = jph_psd_range();
    basis.valid = factors.valid;
    // Cheap exact mask/count bounds remain useful. Within that bound the
    // pivoted, twice-reorthogonalized QR resolves a numerical response space;
    // sub-resolution dependence is ordinary rank reduction, not device failure.
    bool3 support = bool3(false);
    for (uint i = 0u; i < factors.count; ++i)
        support = support || (factors.column[i] != float3(0.0f));
    uint support_count = uint(support.x) + uint(support.y) + uint(support.z);
    uint upper_bound = min(factors.upperBound, min(dimension, min(factors.count, support_count)));
    for (uint rank = 0u; rank < upper_bound; ++rank)
    {
        float best_length = 0.0f;
        float3 best = float3(0.0f);
        // Pivoting avoids constructing a poor basis from two nearly parallel
        // early columns when a later actual DOF is well independent.
        for (uint candidate = 0u; candidate < factors.count; ++candidate)
        {
            float3 residual = factors.column[candidate];
            for (uint pass = 0u; pass < 2u; ++pass)
                for (uint i = 0u; i < basis.rank; ++i)
                    residual -= dot(basis.axis[i], residual) * basis.axis[i];
            float residual_length = length(residual);
            if (residual_length > best_length) { best_length = residual_length; best = residual; }
        }
        // Dimensionless direction resolution for these scaled float32
        // factors. This can omit a real near-dependent mode below resolution;
        // it must not be described as an exact geometric nullspace proof.
        // Validate retained-space residual, physical response and energy in
        // ordinary fixtures; an unresolved RHS component is not repaired here.
        if (best_length <= 8.0f * 1.1920928955078125e-7f) break;
        basis.axis[basis.rank++] = best / best_length;
    }
    return basis;
}

bool jph_psd_full2(float a, float b, float c, float2 rhs, out float2 value)
{
    float2 d = rsqrt(float2(a, c));
    float off = b * d.x * d.y;
    float det = 1.0f - off * off;
    float2 scaled_rhs = rhs * d;
    value = d * float2(scaled_rhs.x - off * scaled_rhs.y, scaled_rhs.y - off * scaled_rhs.x) / det;
    return a > 0.0f && c > 0.0f && det > 0.0f && all(isfinite(value));
}

bool jph_psd_full3(float3 c0, float3 c1, float3 c2, float3 rhs, out float3 value)
{
    float3 diagonal = float3(c0.x, c1.y, c2.z);
    if (any(diagonal <= 0.0f) || !all(isfinite(diagonal)))
    { value = float3(0.0f); return false; }
    float3 d = rsqrt(diagonal);
    c0 *= d * d.x;
    c1 *= d * d.y;
    c2 *= d * d.z;
    float3 row0 = cross(c1, c2);
    float3 row1 = cross(c2, c0);
    float3 row2 = cross(c0, c1);
    float det = dot(c0, row0);
    float3 scaled_rhs = d * rhs;
    value = d * float3(dot(row0, scaled_rhs), dot(row1, scaled_rhs), dot(row2, scaled_rhs)) / det;
    return det > 0.0f && all(isfinite(value));
}

float3 jph_psd_product(float3 c0, float3 c1, float3 c2, float3 value)
{
    return c0 * value.x + c1 * value.y + c2 * value.z;
}

bool jph_psd_solve3(float3 c0, float3 c1, float3 c2, float3 rhs, JphPsdRange basis, out float3 value)
{
    value = float3(0.0f);
    if (!basis.valid || !all(isfinite(c0)) || !all(isfinite(c1)) || !all(isfinite(c2)) || !all(isfinite(rhs)))
        return false;
    if (basis.rank == 3u) return jph_psd_full3(c0, c1, c2, rhs, value);
    // Rank zero can be numerical zero at the documented direction resolution.
    // Input finiteness was checked above; no reachable impulse is applied.
    if (basis.rank == 0u) return true;
    float3 u = basis.axis[0];
    float3 ku = jph_psd_product(c0, c1, c2, u);
    if (basis.rank == 1u)
    {
        float response = dot(u, ku);
        if (!(response > 0.0f)) return false;
        value = u * (dot(u, rhs) / response);
        return all(isfinite(value));
    }
    float3 v = basis.axis[1];
    float3 kv = jph_psd_product(c0, c1, c2, v);
    float2 reduced;
    if (!jph_psd_full2(dot(u, ku), dot(u, kv), dot(v, kv), float2(dot(u, rhs), dot(v, rhs)), reduced))
        return false;
    value = u * reduced.x + v * reduced.y;
    return all(isfinite(value));
}

bool jph_psd_solve2(float a, float b, float c, float2 rhs, JphPsdRange basis, out float2 value)
{
    value = float2(0.0f);
    if (!basis.valid || !all(isfinite(float3(a, b, c))) || !all(isfinite(rhs))) return false;
    if (basis.rank == 2u) return jph_psd_full2(a, b, c, rhs, value);
    if (basis.rank == 0u) return true;
    float2 u = basis.axis[0].xy;
    float response = dot(u, float2(a * u.x + b * u.y, b * u.x + c * u.y));
    if (!(response > 0.0f)) return false;
    value = u * (dot(u, rhs) / response);
    return all(isfinite(value));
}

#endif
