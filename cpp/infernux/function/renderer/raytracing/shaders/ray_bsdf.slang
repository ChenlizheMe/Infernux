import inx_lit_scattering;

static const float ray_bsdf_pi = 3.14159265358979323846;
static const float ray_bsdf_inverse_pi = 0.31830988618379067154;

public uint inx_ray_sample_hash(uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint3 ray_random_bits3(uint seed)
{
    return uint3(inx_ray_sample_hash(seed), inx_ray_sample_hash(seed + 1u), inx_ray_sample_hash(seed + 2u));
}

public float3 inx_ray_uniform3(uint seed)
{
    return float3(ray_random_bits3(seed) >> 8u) * (1.0 / 16777216.0);
}

// The path seed contains the pixel scramble XOR the sample-index hash.
// Each coordinate visits all 32 strata once before choosing another permutation.
public float3 ray_reflected_uniform3(uint path_seed, uint index)
{
    uint pixel_seed = path_seed ^ inx_ray_sample_hash(index);
    uint block = inx_ray_sample_hash(pixel_seed ^ inx_ray_sample_hash(index >> 5u));
    uint3 permutation = ray_random_bits3(block);
    uint3 strata = (((permutation >> 5u) | 1u) * (index & 31u) + permutation) & 31u;
    uint3 jitter = ray_random_bits3(path_seed) >> 13u;
    return float3((strata << 19u) | jitter) * (1.0 / 16777216.0);
}

struct RayBsdfFrame
{
    float3 tangent;
    float3 bitangent;
    float3 normal;

    float3 to_world(float3 direction)
    {
        return tangent * direction.x + bitangent * direction.y + normal * direction.z;
    }

    float3 to_local(float3 direction)
    {
        return float3(dot(tangent, direction), dot(bitangent, direction), dot(normal, direction));
    }
};

RayBsdfFrame ray_bsdf_frame(float3 normal)
{
    float3 axis = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    float3 tangent = normalize(cross(axis, normal));
    return {tangent, cross(normal, tangent), normal};
}

/// Directions and both normals are unit length. The geometric normal faces the outgoing side.
public struct RayDirectionSample
{
    public float3 direction;
    public float pdf;
    public float3 weight;
    public bool active;
    public bool delta;
};

/// Proposal outcomes outside the geometric hemisphere are counted with zero weight.
public RayDirectionSample ray_sample_irradiance(float2 uniform, float3 shading_normal, float3 geometric_normal)
{
    float radius = sqrt(uniform.x);
    float phi = 2.0 * ray_bsdf_pi * uniform.y;
    float cosine = sqrt(1.0 - uniform.x);
    float3 direction =
        ray_bsdf_frame(shading_normal).to_world(float3(radius * cos(phi), radius * sin(phi), cosine));
    bool active = cosine > 0.0 && dot(direction, geometric_normal) > 0.0;
    return {direction, cosine * ray_bsdf_inverse_pi, float3(active ? ray_bsdf_pi : 0.0), active, false};
}

/// Lit uses its authored roughness range. An explicit alpha of zero represents an ideal reflection event.
public struct RayGgxMaterial
{
    public float alpha;
    public float3 f0;
    public float f90;
    public float strength;
};

public RayGgxMaterial ray_lit_ggx(float3 albedo, float metallic, float smoothness, float normal_variance,
                                  float specular_highlights)
{
    let scattering = inx_prepare_lit_scattering(albedo, metallic, smoothness, normal_variance);
    return {scattering.ggx_alpha, scattering.f0, scattering.f90, saturate(specular_highlights)};
}

float3 ray_ggx_fresnel(RayGgxMaterial material, float cosine)
{
    float tail = 1.0 - saturate(cosine);
    float tail_squared = tail * tail;
    return lerp(material.f0, float3(material.f90), tail * tail_squared * tail_squared) * material.strength;
}

struct RayGgxTerms
{
    float pdf;
    float3 weight;
    bool active;
};

RayGgxTerms ray_ggx_terms(float3 shading_normal, float3 geometric_normal, float3 outgoing, float3 incoming,
                          float3 facet, RayGgxMaterial material)
{
    float nov = saturate(dot(shading_normal, outgoing));
    float noh = saturate(dot(shading_normal, facet));
    float voh = saturate(dot(outgoing, facet));
    if (!(nov > 0.0 && noh > 0.0 && voh > 0.0 && dot(geometric_normal, outgoing) > 0.0))
        return {0.0, float3(0.0), false};

    float alpha_squared = material.alpha * material.alpha;
    float nv = inx_ggx_projected_roughness(nov, alpha_squared);
    float d = inx_ggx_ndf_denominator(noh, alpha_squared);
    float distribution = ray_bsdf_inverse_pi * alpha_squared / (d * d);
    // Visible normal density times the reflection Jacobian. G1(v)/(4 * nov) simplifies analytically.
    float pdf = distribution / (2.0 * (nov + nv));
    float nol = saturate(dot(shading_normal, incoming));
    if (!(nol > 0.0 && dot(geometric_normal, incoming) > 0.0))
        return {pdf, float3(0.0), false};

    float nl = inx_ggx_projected_roughness(nol, alpha_squared);
    // In f*cos/pdf, only the ratio of correlated Smith G2 to visible-normal G1 remains.
    float masking = (nol * (nov + nv)) / (nol * nv + nov * nl);
    return {pdf, ray_ggx_fresnel(material, voh) * masking, true};
}

public struct RayGgxEvaluation
{
    public float pdf;
    public float3 projected_bsdf;
};

/// Continuous BSDF times the incident cosine. Delta reflection has no solid-angle density.
public RayGgxEvaluation ray_evaluate_ggx(float3 shading_normal, float3 geometric_normal, float3 outgoing,
                                          float3 incoming, RayGgxMaterial material)
{
    float3 half_vector = outgoing + incoming;
    float half_length_squared = dot(half_vector, half_vector);
    if (material.alpha == 0.0 || half_length_squared == 0.0)
        return {0.0, float3(0.0)};
    RayGgxTerms terms = ray_ggx_terms(shading_normal, geometric_normal, outgoing, incoming,
                                      half_vector * rsqrt(half_length_squared), material);
    return {terms.pdf, terms.weight * terms.pdf};
}

float3 ray_visible_ggx_facet(float3 outgoing, float alpha, float2 uniform)
{
    // Heitz (2018), projected disk parameterization for the unit-roughness hemisphere.
    float3 projected_view = normalize(float3(alpha * outgoing.xy, outgoing.z));
    float3 azimuth = float3(-projected_view.y, projected_view.x, 0.0);
    float azimuth_length_squared = dot(azimuth, azimuth);
    azimuth = azimuth_length_squared > 0.0 ? azimuth * rsqrt(azimuth_length_squared) : float3(1.0, 0.0, 0.0);

    float phi = 2.0 * ray_bsdf_pi * uniform.y;
    float2 disk = sqrt(uniform.x) * float2(cos(phi), sin(phi));
    float rim = sqrt(max(0.0, 1.0 - disk.x * disk.x));
    disk.y = lerp(rim, disk.y, 0.5 * (1.0 + projected_view.z));
    float z = sqrt(max(0.0, 1.0 - dot(disk, disk)));
    float3 projected_normal =
        disk.x * azimuth + disk.y * cross(projected_view, azimuth) + z * projected_view;
    return normalize(float3(alpha * projected_normal.xy, max(0.0, projected_normal.z)));
}

/// Sample a reflected direction about the shading normal and reject geometry-crossing directions without resampling.
public RayDirectionSample ray_sample_ggx(float2 uniform, float3 shading_normal, float3 geometric_normal,
                                          float3 outgoing, RayGgxMaterial material)
{
    float cosine = dot(shading_normal, outgoing);
    if (!(cosine > 0.0 && dot(geometric_normal, outgoing) > 0.0))
        return {float3(0.0), 0.0, float3(0.0), false, material.alpha == 0.0};

    if (material.alpha == 0.0)
    {
        float3 direction = reflect(-outgoing, shading_normal);
        bool active = dot(geometric_normal, direction) > 0.0;
        return {direction, 0.0, active ? ray_ggx_fresnel(material, cosine) : float3(0.0), active, true};
    }

    let frame = ray_bsdf_frame(shading_normal);
    float3 facet = frame.to_world(ray_visible_ggx_facet(frame.to_local(outgoing), material.alpha, uniform));
    float3 direction = reflect(-outgoing, facet);
    RayGgxTerms terms = ray_ggx_terms(shading_normal, geometric_normal, outgoing, direction, facet, material);
    return {direction, terms.pdf, terms.weight, terms.active, false};
}

/// A continuous proposal has no material throughput. Evaluate the full BSDF with this density.
public struct RayDirectionProposal
{
    public float3 direction;
    public float pdf;
    public bool active;
};

/// diffuse_probability is in [0, 1]. Incoming geometry rejection does not renormalize this density.
public float ray_continuous_pdf(float3 shading_normal, float3 geometric_normal, float3 outgoing,
                                  float3 incoming, RayGgxMaterial material, float diffuse_probability)
{
    if (!(dot(shading_normal, outgoing) > 0.0 && dot(geometric_normal, outgoing) > 0.0))
        return 0.0;
    float cosine_pdf = max(dot(shading_normal, incoming), 0.0) * ray_bsdf_inverse_pi;
    // Delta reflection is a separate event, never a solid-angle component.
    if (material.alpha == 0.0)
        return diffuse_probability > 0.0 ? cosine_pdf : 0.0;
    float reflection_pdf = ray_evaluate_ggx(shading_normal, geometric_normal, outgoing, incoming, material).pdf;
    return diffuse_probability * cosine_pdf + (1.0 - diffuse_probability) * reflection_pdf;
}

/// uniform.xy samples a direction; uniform.z selects its proposal with the declared probability.
public RayDirectionProposal ray_sample_continuous(float3 uniform, float3 shading_normal, float3 geometric_normal,
                                                    float3 outgoing, RayGgxMaterial material,
                                                    float diffuse_probability)
{
    if (!(dot(shading_normal, outgoing) > 0.0 && dot(geometric_normal, outgoing) > 0.0) ||
        (material.alpha == 0.0 && diffuse_probability == 0.0))
        return {float3(0.0), 0.0, false};
    RayDirectionSample sample;
    if (material.alpha == 0.0 || uniform.z < diffuse_probability)
        sample = ray_sample_irradiance(uniform.xy, shading_normal, geometric_normal);
    else
        sample = ray_sample_ggx(uniform.xy, shading_normal, geometric_normal, outgoing, material);
    float pdf = ray_continuous_pdf(shading_normal, geometric_normal, outgoing, sample.direction, material,
                                    diffuse_probability);
    return {sample.direction, pdf, sample.active};
}

/// The previous surface viewed from a directional light. A zero width disables this proposal.
public struct RayReflectedLightLobe
{
    public float3 normal;
    public float alpha;
    public float3 toward_light;
};

public float ray_reflected_light_pdf(float3 normal, float3 geometric, float3 outgoing, float3 incoming,
    RayGgxMaterial material, float diffuse_probability, RayReflectedLightLobe guide)
{
    float bsdf_pdf = ray_continuous_pdf(normal, geometric, outgoing, incoming, material, diffuse_probability);
    if (guide.alpha <= 0.0)
        return bsdf_pdf;
    RayGgxMaterial reflector = {guide.alpha, float3(1.0), 1.0, 1.0};
    float3 reflected = -incoming;
    float light_pdf = dot(reflected, guide.normal) > 0.0
        ? ray_evaluate_ggx(guide.normal, guide.normal, guide.toward_light, reflected, reflector).pdf : 0.0;
    return 0.5 * (bsdf_pdf + light_pdf);
}

/// The two proposals share solid-angle measure at the receiver. A rejected hemisphere sample has zero weight.
public RayDirectionProposal ray_sample_reflected_light(float3 uniform, float3 normal, float3 geometric,
    float3 outgoing, RayGgxMaterial material, float diffuse_probability, RayReflectedLightLobe guide)
{
    RayGgxMaterial reflector = {guide.alpha, float3(1.0), 1.0, 1.0};
    if (uniform.z < 0.5)
    {
        let light_sample = ray_sample_ggx(uniform.xy, guide.normal, guide.normal, guide.toward_light, reflector);
        float3 direction = -light_sample.direction;
        bool active = light_sample.active && dot(direction, normal) > 0.0 && dot(direction, geometric) > 0.0 &&
                      dot(outgoing, normal) > 0.0 && dot(outgoing, geometric) > 0.0;
        float bsdf_pdf = ray_continuous_pdf(normal, geometric, outgoing, direction, material, diffuse_probability);
        return {direction, 0.5 * (light_sample.pdf + bsdf_pdf), active};
    }

    uniform.z = 2.0 * uniform.z - 1.0;
    let bsdf_sample = ray_sample_continuous(uniform, normal, geometric, outgoing, material, diffuse_probability);
    float3 reflected = -bsdf_sample.direction;
    float light_pdf = dot(reflected, guide.normal) > 0.0
        ? ray_evaluate_ggx(guide.normal, guide.normal, guide.toward_light, reflected, reflector).pdf : 0.0;
    return {bsdf_sample.direction, 0.5 * (bsdf_sample.pdf + light_pdf), bsdf_sample.active};
}
