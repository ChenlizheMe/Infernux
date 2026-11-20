// The authored secondary-surface dispatch is declared by the including material module.
// Matches RayMaterialParticipation in the material table producer.
static const uint inx_ray_material_lit = 1u;
static const uint inx_ray_material_excluded = 2u;
static const uint inx_ray_material_unlit = 3u;

bool inx_ray_has_surface(uint kind)
{
    return kind == inx_ray_material_lit || kind == inx_ray_material_unlit;
}

float3 inx_ray_unlit_radiance(RaySurfaceHit hit, InxSurfaceData surface)
{
    float3 geometric = hit.front_face ? hit.geometric_normal_ws : -hit.geometric_normal_ws;
    let context = inx_shading_context(hit.position_ws, geometric, 0.0, float2(0.0), 0xffffffffu, false, 0.0);
    return inx_ray_unlit_response::shade<false>(surface, context);
}

bool inx_evaluate_ray_lit(InxMeshInput input, bool front_facing, InxSecondarySurfaceSampling footprint,
                          out InxSurfaceData surface)
{
    InxRayLitSampling sampling = {footprint};
    surface = inx_ray_author_surface(input, sampling);
    surface = inx_prepare_surface(surface, input, front_facing);
    return inx_surface_covered(surface, inx_ray_alpha_clip(input.surface_id));
}

bool inx_evaluate_ray_lit_hit(RaySurfaceHit attributes, uint surface_id, float lod, out InxSurfaceData surface)
{
    InxMeshInput input = (InxMeshInput)0;
    input.surface_id = surface_id;
    input.position_ws = attributes.position_ws;
    input.normal_ws = attributes.normal_ws;
    input.tangent_ws = attributes.tangent_ws;
    input.color = attributes.color.rgb;
    input.uv0 = attributes.uv0;
    input.uv1 = attributes.uv1;
    input.line_color = float4(1.0);
    InxSecondarySurfaceSampling footprint = {lod, attributes.edge_u_ws, attributes.edge_v_ws,
                                             attributes.edge_u_uv1, attributes.edge_v_uv1};
    return inx_evaluate_ray_lit(input, attributes.front_face, footprint, surface);
}

struct InxRayLitHitPolicy : RayHitPolicy
{
    float lod;

    bool accepts(RayPage page, TraceRay ray, RayHit candidate)
    {
        uint participation = inx_ray_participation(candidate.surface_id);
        if (!inx_ray_has_surface(participation))
            return participation != inx_ray_material_excluded;
        if (!(inx_ray_alpha_clip(candidate.surface_id) > 0.0))
            return true;
        InxSurfaceData surface;
        return inx_evaluate_ray_lit_hit(ray_surface_at_hit(page, ray, candidate), candidate.surface_id, lod, surface);
    }
};

RayHit inx_trace_lit_closest(RayPage page, TraceRay ray, uint layer_mask, float lod)
{
    InxRayLitHitPolicy policy = {lod};
    return trace_closest(page, ray, layer_mask, policy);
}

bool inx_trace_lit_visibility(RayPage page, TraceRay ray, uint layer_mask, float lod)
{
    InxRayLitHitPolicy policy = {lod};
    return trace_visibility(page, ray, layer_mask, policy);
}

public InxRaySurfaceHit inx_world_ray_closest(float3 origin_ws, float3 direction_ws, float min_distance,
                                            float max_distance, uint layer_mask, float lod)
{
    TraceRay ray = {origin_ws, min_distance, direction_ws, max_distance};
    RayPage page = ray_page();
    InxRaySurfaceHit result = (InxRaySurfaceHit)0;
    result.distance = max_distance;
    result.available = page.status == ray_page_ready;
    if (!result.available)
        return result;
    RayHit hit = inx_trace_lit_closest(page, ray, layer_mask, lod);
    if (hit.triangle_index != ray_invalid_index)
    {
        if (!inx_ray_has_surface(inx_ray_participation(hit.surface_id)))
        {
            result.available = false;
            return result;
        }
        result.found = true;
        result.distance = hit.distance;
        RaySurfaceHit attributes = ray_surface_at_hit(page, ray, hit);
        result.position_ws = attributes.position_ws;
        result.surface_id = hit.surface_id;
        inx_evaluate_ray_lit_hit(attributes, hit.surface_id, lod, result.surface);
    }
    return result;
}

public bool inx_world_ray_visibility(float3 origin_ws, float3 direction_ws, float min_distance,
                                     float max_distance, uint layer_mask, float lod)
{
    TraceRay ray = {origin_ws, min_distance, direction_ws, max_distance};
    return inx_trace_lit_visibility(ray_page(), ray, layer_mask, lod);
}

uint2 inx_ray_sobol_bits(uint index)
{
    uint x = reversebits(index);
    uint y = x;
    y ^= (y & 0x55555555u) << 1u;
    y ^= (y & 0x33333333u) << 2u;
    y ^= (y & 0x0f0f0f0fu) << 4u;
    y ^= (y & 0x00ff00ffu) << 8u;
    y ^= y << 16u;
    return uint2(x, y);
}

float2 inx_ray_primary_uniform(uint index, uint pixel_seed)
{
    // Sobol dimension 3 (x^2 + x + 1, initial numerators 1, 3) stratifies
    // camera phase. These masks invert its leading three bits for this block.
    // Only the order changes. A surface visible at a single jitter phase
    // still receives directions from the entire hemisphere over time.
    uint phase = (countbits(index & 0x79e79e78u) & 1u)
               | ((countbits(index & 0xa28a28a0u) & 1u) << 1u)
               | ((countbits(index & 0x14514510u) & 1u) << 2u);
    index ^= phase;
    // Two base-2 Sobol dimensions. The second direction matrix is Pascal's triangle mod 2.
    uint2 shift = uint2(inx_ray_sample_hash(pixel_seed), inx_ray_sample_hash(pixel_seed + 1u));
    return float2((inx_ray_sobol_bits(index) ^ shift) >> 8u) * (1.0 / 16777216.0);
}

float inx_ray_offset(float position, float normal)
{
    if (abs(position) < 1.0 / 32.0)
        return position + normal * (1.0 / 65536.0);
    int offset = int(256.0 * normal);
    return asfloat(asint(position) + (position < 0.0 ? -offset : offset));
}

float3 inx_ray_origin(float3 position, float3 normal)
{
    return float3(inx_ray_offset(position.x, normal.x), inx_ray_offset(position.y, normal.y),
                  inx_ray_offset(position.z, normal.z));
}

float inx_ray_round_outside(float position, float displacement)
{
    if (displacement == 0.0)
        return position;
    if (position == 0.0)
        return asfloat(displacement > 0.0 ? 1u : 0x80000001u);
    uint bits = asuint(position);
    return asfloat((position > 0.0) == (displacement > 0.0) ? bits + 1u : bits - 1u);
}

float3 inx_ray_origin(float3 position, float3 normal, float3 error)
{
    float3 displacement = normal * dot(abs(normal), error);
    float3 outside = position + displacement;
    return float3(inx_ray_round_outside(outside.x, displacement.x),
                  inx_ray_round_outside(outside.y, displacement.y),
                  inx_ray_round_outside(outside.z, displacement.z));
}

struct InxRayShadowHitPolicy : RayHitPolicy
{
    InxBuffer_uint4 records;
    InxRayLitHitPolicy material;

    bool accepts(RayPage page, TraceRay ray, RayHit candidate)
    {
        return inx_buffer_load(records, candidate.surface_id * 3u).w != 0u &&
               material.accepts(page, ray, candidate);
    }
};

TraceRay inx_ray_open_segment(float3 origin, float3 delta)
{
    // The endpoint is at t = 1 for every length, including after an instance transform.
    return {origin, 0.0, delta, asfloat(0x3f7fffffu)};
}

TraceRay inx_ray_light_visibility(float3 origin, InxTransportLight light)
{
    if (light.directional)
        return {origin, 0.0, light.sample.direction, 3.402823e38};
    float3 to_light = light.endpoint - origin;
    if (dot(to_light, light.sample.direction) <= 0.0)
        return {origin, 0.0, light.sample.direction, 0.0};
    return inx_ray_open_segment(origin, to_light);
}

float3 inx_ray_direct_radiance(RayPage page, RaySurfaceHit attributes, InxSurfaceData surface,
                               uint4 metadata, InxBuffer_uint4 records, float3 outgoing, uint seed)
{
    uint2 counts = inx_transport_light_counts();
    uint count = counts.x + counts.y;
    if (count == 0u)
        return float3(0.0);
    float3 geometric = attributes.front_face ? attributes.geometric_normal_ws : -attributes.geometric_normal_ws;
    InxPbrLightResponse response = inx_secondary_light_response(surface, surface.normal_ws, outgoing);
    uint strata = min(count, 4u);
    float3 radiance = float3(0.0);
    float3 total = float3(0.0);
    float3 orientation = float3(0.0);
    uint3 selected = uint3(ray_invalid_index);
    for (uint i = 0u; i < strata; ++i)
    {
        // Each stratum estimates its own sum. Only the shadow-dependent part needs a ray.
        uint stream = inx_ray_sample_hash(seed + i);
        float sample_u = float(inx_ray_sample_hash(stream ^ 0x91e10da5u) >> 8u) * (1.0 / 16777216.0);
        InxTransportLightSelection candidate = inx_sample_transport_light(
            count, strata, i, sample_u, attributes.position_ws, geometric, surface.normal_ws, metadata.x);
        uint index = candidate.index;
        uint corner = inx_ray_sample_hash(stream ^ 0x68bc21ebu) & 3u;
        InxTransportLight light = inx_transport_light(index, attributes.position_ws, corner);
        if ((light.layer_mask & metadata.x) == 0u || light.sample.attenuation <= 0.0 ||
            dot(geometric, light.sample.direction) <= 0.0 || dot(surface.normal_ws, light.sample.direction) <= 0.0)
            continue;
        float3 contribution = candidate.inverse_pdf * response.evaluate(light.sample);
        float strength = metadata.z != 0u ? light.shadow_strength : 0.0;
        radiance += (1.0 - strength) * contribution;
        float3 shadowed = strength * contribution;
        float3 weight = abs(shadowed);
        total += weight;
        // A shared random number correlates RGB without changing each channel's selection PDF.
        float replace_u = float(inx_ray_sample_hash(stream ^ 0xc2b2ae35u) >> 8u) * (1.0 / 16777216.0);
        uint key = (index << 2u) | corner;
        for (uint channel = 0u; channel < 3u; ++channel)
        {
            if (weight[channel] > replace_u * total[channel])
            {
                selected[channel] = key;
                orientation[channel] = sign(shadowed[channel]);
            }
        }
    }
    float3 origin = inx_ray_origin(attributes.position_ws, geometric, attributes.position_error_ws);
    InxRayLitHitPolicy material = {0.0};
    InxRayShadowHitPolicy caster = {records, material};
    // Channels that selected the same light endpoint share one visibility query.
    for (uint channel = 0u; channel < 3u; ++channel)
    {
        uint key = selected[channel];
        if (key == ray_invalid_index)
            continue;
        InxTransportLight light = inx_transport_light(key >> 2u, attributes.position_ws, key & 3u);
        TraceRay shadow_ray = inx_ray_light_visibility(origin, light);
        bool visible = shadow_ray.maximum <= 0.0 || trace_visibility(page, shadow_ray, light.layer_mask, caster);
        for (uint match = channel; match < 3u; ++match)
        {
            if (selected[match] == key)
            {
                if (visible)
                    radiance[match] += total[match] * orientation[match];
                selected[match] = ray_invalid_index;
            }
        }
    }
    return radiance;
}

struct InxRayTransportTerm
{
    float3 radiance;
    bool available;
    float distance;
};

// Proposal selection only. Both lobes are evaluated with the same mixture PDF below.
float inx_ray_diffuse_probability(InxSurfaceData surface, RayGgxMaterial material, float nov)
{
    const float3 luminance = float3(0.2126, 0.7152, 0.0722);
    float diffuse = dot(saturate(surface.albedo), luminance) * (1.0 - saturate(surface.metallic));
    float grazing = 1.0 - saturate(nov);
    float grazing2 = grazing * grazing;
    float glossy = lerp(dot(material.f0, luminance), material.f90, grazing2 * grazing2 * grazing) * material.strength;
    if (diffuse == 0.0) return 0.0;
    if (glossy == 0.0) return 1.0;
    // Keep both proposals represented for mixed materials and coloured incident light.
    return clamp(diffuse / (diffuse + glossy), 0.05, 0.95);
}

RayReflectedLightLobe inx_ray_reflected_light(float4 reflector, uint seed)
{
    RayReflectedLightLobe guide = (RayReflectedLightLobe)0;
    // Narrow lobes need a fresh light connection. Broad lobes share the existing world-cache budget.
    float importance = 1.0 - smoothstep(0.04, 0.16, reflector.w);
    if (reflector.w <= 0.0 || float(inx_ray_sample_hash(seed ^ 0x103bf8d3u) >> 8u) * (1.0 / 16777216.0) >= importance)
        return guide;
    uint count = inx_transport_light_counts().x;
    if (count > 0u)
    {
        uint index = inx_ray_sample_hash(seed ^ 0x459e1d09u) % count;
        let light = inx_transport_light(index, float3(0.0), 0u);
        if (light.layer_mask != 0u && dot(reflector.xyz, light.sample.direction) > 0.0)
            return {reflector.xyz, reflector.w, light.sample.direction};
    }
    return guide;
}

float3 inx_ray_secondary_weight(InxSurfaceData surface, float3 geometric, float3 outgoing, uint seed, uint sample_index,
                                 out RayDirectionProposal proposal, RayReflectedLightLobe guide)
{
    float3 normal = normalize(surface.normal_ws);
    RayGgxMaterial material = ray_lit_ggx(surface.albedo, surface.metallic, surface.smoothness,
                                          0.0, surface.specular_highlights);
    float diffuse_probability = inx_ray_diffuse_probability(surface, material, dot(normal, outgoing));
    if (guide.alpha > 0.0)
        proposal = ray_sample_reflected_light(ray_reflected_uniform3(seed, sample_index), normal, geometric, outgoing,
                                              material, diffuse_probability, guide);
    else
        proposal = ray_sample_continuous(inx_ray_uniform3(seed), normal, geometric, outgoing, material, diffuse_probability);
    if (!proposal.active)
        return float3(0.0);
    return inx_secondary_projected_bsdf(surface, normal, outgoing, proposal.direction) / proposal.pdf;
}

struct InxRayBounce
{
    RayHit hit;
    RaySurfaceHit attributes;
    InxSurfaceData surface;
    float3 outgoing;
    float3 throughput;
};

float inx_ray_mis_weight(float own_pdf, float other_pdf)
{
    if (!(own_pdf > 0.0))
        return 0.0;
    float relative = other_pdf / own_pdf;
    return 1.0 / (1.0 + relative * relative);
}

struct InxRayEmitterSample
{
    RaySurfaceHit attributes;
    uint2 endpoint;
    float3 direction;
    float3 emission;
    float pdf;
    bool active;
};

InxRayEmitterSample inx_ray_emitter_sample(RayPage page, float3 position, float3 uniform, float maximum)
{
    InxRayEmitterSample sample = (InxRayEmitterSample)0;
    let selection = ray_sample_emissive(page, uniform);
    if (!selection.active)
        return sample;
    sample.endpoint = uint2(selection.hit.instance_index, selection.hit.triangle_index);
    TraceRay ray = {position, 0.0, float3(0.0), maximum};
    sample.attributes = ray_surface_at_hit(page, ray, selection.hit);
    float3 delta = sample.attributes.position_ws - position;
    float distance = length(delta);
    if (!(distance > 0.0 && distance <= maximum))
        return sample;
    sample.direction = delta / distance;
    sample.attributes.front_face = dot(sample.direction, sample.attributes.geometric_normal_ws) < 0.0;
    sample.pdf = ray_emissive_solid_angle_pdf(selection.pmf, sample.attributes, position);
    if (!(sample.pdf > 0.0))
        return sample;
    InxSurfaceData surface;
    if (!inx_evaluate_ray_lit_hit(sample.attributes, selection.hit.surface_id, 0.0, surface))
        return sample;
    sample.emission = inx_ray_participation(selection.hit.surface_id) == inx_ray_material_unlit
        ? inx_ray_unlit_radiance(sample.attributes, surface) : surface.emission;
    sample.active = true;
    return sample;
}

struct InxRayEndpointHitPolicy : RayHitPolicy
{
    uint2 endpoint;

    bool accepts(RayPage page, TraceRay ray, RayHit candidate)
    {
        // Rounding can put the sampled triangle just inside the open segment.
        // A straight segment ending on that triangle cannot cross it earlier.
        if (all(uint2(candidate.instance_index, candidate.triangle_index) == endpoint))
            return false;
        InxRayLitHitPolicy material = {0.0};
        return material.accepts(page, ray, candidate);
    }
};

bool inx_ray_emitter_visible(RayPage page, float3 origin, InxRayEmitterSample sample)
{
    float3 segment = sample.attributes.position_ws - origin;
    if (all(segment == float3(0.0)))
        return false;
    InxRayEndpointHitPolicy material = {sample.endpoint};
    return trace_visibility(page, inx_ray_open_segment(origin, segment), 0xffffffffu, material);
}

float3 inx_ray_emissive_direct(RayPage page, RaySurfaceHit attributes, InxSurfaceData surface,
    float3 outgoing, float3 geometric, uint seed, float maximum, RayReflectedLightLobe guide)
{
    let sample = inx_ray_emitter_sample(page, attributes.position_ws, inx_ray_uniform3(seed ^ 0x39aa145du), maximum);
    if (!sample.active || dot(geometric, sample.direction) <= 0.0)
        return float3(0.0);
    float3 normal = normalize(surface.normal_ws);
    RayGgxMaterial material = ray_lit_ggx(surface.albedo, surface.metallic, surface.smoothness,
                                          0.0, surface.specular_highlights);
    float probability = inx_ray_diffuse_probability(surface, material, dot(normal, outgoing));
    float bsdf_pdf = ray_reflected_light_pdf(normal, geometric, outgoing, sample.direction, material,
                                              probability, guide);
    float3 response = inx_secondary_projected_bsdf(surface, normal, outgoing, sample.direction);
    if (!inx_ray_emitter_visible(page, inx_ray_origin(attributes.position_ws, geometric,
                                                     attributes.position_error_ws), sample))
        return float3(0.0);
    return sample.emission * response * (inx_ray_mis_weight(sample.pdf, bsdf_pdf) / sample.pdf);
}

// D owns analytic NEE and MIS-combined emissive/BSDF endpoint samples.
// It contains no further reflected scattering at the endpoint.
InxRayTransportTerm inx_ray_direct_sample(RayPage page, RaySurfaceHit attributes, InxSurfaceData surface,
    uint4 metadata, InxBuffer_uint4 records, float3 outgoing, uint seed, float maximum, float light_roughness,
    RayReflectedLightLobe guide, uint sample_index, out InxRayBounce bounce)
{
    bounce = (InxRayBounce)0;
    bounce.hit.triangle_index = ray_invalid_index;
    InxSurfaceData light_surface = surface;
    light_surface.smoothness = min(surface.smoothness, 1.0 - light_roughness);
    float3 radiance = inx_ray_direct_radiance(page, attributes, light_surface, metadata, records, outgoing, seed);
    float3 geometric = attributes.front_face ? attributes.geometric_normal_ws : -attributes.geometric_normal_ws;
    radiance += inx_ray_emissive_direct(page, attributes, surface, outgoing, geometric, seed, maximum, guide);
    RayDirectionProposal proposal;
    float3 weight = inx_ray_secondary_weight(surface, geometric, outgoing, seed ^ 0xb5297a4du, sample_index, proposal, guide);
    if (!proposal.active || max(weight.r, max(weight.g, weight.b)) <= 0.0)
        return {radiance, true, 0.0};
    TraceRay endpoint = {inx_ray_origin(attributes.position_ws, geometric, attributes.position_error_ws),
                         0.0, proposal.direction, maximum};
    RayHit hit = inx_trace_lit_closest(page, endpoint, 0xffffffffu, 0.0);
    if (hit.triangle_index == ray_invalid_index)
        return {radiance + weight * inx_environment_radiance(endpoint.direction), true, 0.0};
    uint kind = inx_ray_participation(hit.surface_id);
    if (!inx_ray_has_surface(kind))
        return {float3(0.0), false, hit.distance};
    bounce.attributes = ray_surface_at_hit(page, endpoint, hit);
    inx_evaluate_ray_lit_hit(bounce.attributes, hit.surface_id, 0.0, bounce.surface);
    float emission_weight = inx_ray_mis_weight(proposal.pdf,
        ray_emissive_pdf(hit, bounce.attributes, attributes.position_ws));
    if (kind == inx_ray_material_unlit)
        return {radiance + emission_weight * weight * inx_ray_unlit_radiance(bounce.attributes, bounce.surface), true, hit.distance};
    bounce.hit = hit;
    bounce.outgoing = -endpoint.direction;
    bounce.throughput = weight;
    return {radiance + emission_weight * weight * bounce.surface.emission, true, hit.distance};
}

InxRayTransportTerm inx_ray_direct_fresh(RayPage page, RaySurfaceHit attributes, InxSurfaceData surface,
                                          uint4 metadata, InxBuffer_uint4 records, float3 outgoing,
                                          uint seed, float maximum, float light_roughness = 0.0)
{
    InxRayBounce endpoint;
    return inx_ray_direct_sample(page, attributes, surface, metadata, records, outgoing, seed, maximum,
                                 light_roughness, (RayReflectedLightLobe)0, 0u, endpoint);
}

// Fixed additional depth one: sample reflected D(z), excluding Le(z) and an environment miss.
// No previous I1 value participates in this integrand.
InxRayTransportTerm inx_ray_tail_sample(RayPage page, RaySurfaceHit attributes, InxSurfaceData surface,
                                         InxBuffer_uint4 records, float3 outgoing, uint seed, float maximum)
{
    float3 geometric = attributes.front_face ? attributes.geometric_normal_ws : -attributes.geometric_normal_ws;
    RayDirectionProposal proposal;
    float3 weight = inx_ray_secondary_weight(surface, geometric, outgoing, seed ^ 0x1b56c4e9u, 0u,
                                             proposal, (RayReflectedLightLobe)0);
    if (!proposal.active || max(weight.r, max(weight.g, weight.b)) <= 0.0)
        return {float3(0.0), true, 0.0};
    TraceRay ray = {inx_ray_origin(attributes.position_ws, geometric, attributes.position_error_ws),
                    0.0, proposal.direction, maximum};
    RayHit hit = inx_trace_lit_closest(page, ray, 0xffffffffu, 0.0);
    if (hit.triangle_index == ray_invalid_index)
        return {float3(0.0), true, 0.0};
    uint kind = inx_ray_participation(hit.surface_id);
    if (kind == inx_ray_material_unlit)
        return {float3(0.0), true, hit.distance};
    if (kind != inx_ray_material_lit)
        return {float3(0.0), false, hit.distance};
    RaySurfaceHit next_attributes = ray_surface_at_hit(page, ray, hit);
    InxSurfaceData next_surface;
    inx_evaluate_ray_lit_hit(next_attributes, hit.surface_id, 0.0, next_surface);
    InxRayTransportTerm next = inx_ray_direct_fresh(page, next_attributes, next_surface,
        inx_buffer_load(records, hit.surface_id * 3u), records, -ray.direction,
        inx_ray_sample_hash(seed ^ 0x68e31da4u), maximum);
    return {weight * next.radiance, next.available, hit.distance};
}

public float4 inx_ray_radiance_parts(float3 origin, float3 direction, InxBuffer_uint4 records,
                                      uint seed, int part, float maximum)
{
    RayPage page = ray_page();
    if (page.status != ray_page_ready)
        return float4(0.0);
    TraceRay ray = {origin, 0.0, direction, maximum};
    RayHit hit = inx_trace_lit_closest(page, ray, 0xffffffffu, 0.0);
    if (hit.triangle_index == ray_invalid_index)
        return float4(part == 0 ? inx_environment_radiance(direction) : float3(0.0), 1.0);
    uint kind = inx_ray_participation(hit.surface_id);
    if (!inx_ray_has_surface(kind))
        return float4(0.0);
    RaySurfaceHit attributes = ray_surface_at_hit(page, ray, hit);
    InxSurfaceData surface;
    inx_evaluate_ray_lit_hit(attributes, hit.surface_id, 0.0, surface);
    if (part == 4)
        return float4(float(hit.surface_id), float(hit.primitive), hit.distance, 1.0);
    if (kind == inx_ray_material_unlit)
        return float4(part <= 1 ? inx_ray_unlit_radiance(attributes, surface) : float3(0.0), 1.0);
    if (part == 1)
        return float4(surface.emission, 1.0);
    InxRayTransportTerm term;
    if (part == 3)
        term = inx_ray_tail_sample(page, attributes, surface, records, -direction, seed, maximum);
    else
        term = inx_ray_direct_fresh(page, attributes, surface, inx_buffer_load(records, hit.surface_id * 3u),
                                     records, -direction, seed, maximum);
    return float4(term.radiance + (part == 0 ? surface.emission : float3(0.0)), term.available ? 1.0 : 0.0);
}

InxRaySignal inx_ray_signal_output(float3 signal, uint2 witness)
{
    return {float4(signal, 1.0), witness};
}

float inx_ray_tail_cone(InxSurfaceData surface)
{
    RayGgxMaterial scattering = ray_lit_ggx(surface.albedo, surface.metallic, surface.smoothness,
                                            0.0, surface.specular_highlights);
    if (scattering.strength > 0.0 && scattering.alpha < 0.1)
        return 0.0;
    float cone = scattering.strength > 0.0 ? min(0.06, 0.2 * scattering.alpha) : 0.06;
    // Request admission and lookup use exactly the same packed angular tolerance.
    return unpackHalf2x16ToFloat(packHalf2x16(float2(cone, 0.0))).x;
}

float3 inx_ray_cached_tail(RayPage page, RayHit hit, RaySurfaceHit attributes, InxBuffer_uint4 cache,
                           float3 outgoing, float cone)
{
    uint4 header = inx_buffer_load(cache, 0u);
    if (header.x == 0u)
        return float3(0.0);
    uint4 metric = inx_buffer_load(cache, 1u);
    float cell_size = asfloat(metric.x);
    int3 cell = ray_tail_cell(attributes.position_ws, cell_size);
    uint direction = ray_tail_direction(outgoing, metric.y);
    uint hash = ray_tail_hash(cell, 0u, hit.surface_id, hit.primitive, direction);
    float3 normal = attributes.front_face ? attributes.geometric_normal_ws : -attributes.geometric_normal_ws;
    for (uint probe = 0u; probe < 4u; ++probe)
    {
        uint slot = (hash + probe) % header.x;
        uint base = 2u + slot * 8u;
        uint4 key = inx_buffer_load(cache, base);
        uint4 identity = inx_buffer_load(cache, base + 1u);
        if (!ray_tail_same(key, identity, cell, hit.surface_id, hit.primitive, header.y))
            continue;
        uint4 anchor = inx_buffer_load(cache, base + 2u);
        uint4 wo = inx_buffer_load(cache, base + 3u);
        uint4 ng = inx_buffer_load(cache, base + 6u);
        if (wo.w != direction || anchor.w == 0u ||
            !ray_tail_representative(asfloat(anchor.xyz), asfloat(wo.xyz), asfloat(ng.xyz),
                                     attributes.position_ws, outgoing, normal, cell_size, cone))
            continue;
        uint4 exact = inx_buffer_load(cache, base + 7u);
        if (exact.x != hit.instance_index)
            continue;
        uint4 bary = inx_buffer_load(cache, base + 4u);
        TraceRay at_anchor = {asfloat(anchor.xyz), 0.0, -asfloat(wo.xyz), 0.0};
        RayHit anchor_hit = {0.0, asfloat(bary.xy), exact.x, exact.y, hit.primitive, hit.surface_id};
        RaySurfaceHit anchor_surface = ray_surface_at_hit(page, at_anchor, anchor_hit);
        float3 origin = inx_ray_origin(attributes.position_ws, normal, attributes.position_error_ws);
        float3 endpoint = inx_ray_origin(anchor_surface.position_ws, asfloat(ng.xyz), anchor_surface.position_error_ws);
        float3 link = endpoint - origin;
        if (any(link != float3(0.0)))
        {
            TraceRay visibility = inx_ray_open_segment(origin, link);
            if (!inx_trace_lit_visibility(page, visibility, 0xffffffffu, 0.0))
                continue;
        }
        return asfloat(inx_buffer_load(cache, base + 5u).xyz);
    }
    return float3(0.0);
}

InxRaySignal inx_ray_radiance_at_hit(RayPage page, TraceRay ray, RayHit hit, InxBuffer_uint4 records,
                                     InxBuffer_uint4 cache, uint seed, float3 weight, float light_roughness,
                                     float4 reflector, uint sample_index, float emission_pdf, float3 emitter_origin)
{
    if (hit.triangle_index == ray_invalid_index)
        return inx_ray_signal_output(inx_environment_radiance(ray.direction) * weight, uint2(0u));
    uint kind = inx_ray_participation(hit.surface_id);
    if (!inx_ray_has_surface(kind))
        return (InxRaySignal)0;
    RaySurfaceHit attributes = ray_surface_at_hit(page, ray, hit);
    InxSurfaceData secondary;
    inx_evaluate_ray_lit_hit(attributes, hit.surface_id, 0.0, secondary);
    // A delta ray has no competing continuous area-sampling path.
    float emission_weight = emission_pdf > 0.0 ? inx_ray_mis_weight(emission_pdf,
        ray_emissive_pdf(hit, attributes, emitter_origin)) : 1.0;
    if (kind == inx_ray_material_unlit)
        return inx_ray_signal_output(emission_weight * weight * inx_ray_unlit_radiance(attributes, secondary),
                                     uint2(hit.surface_id, asuint(hit.distance)));
    float maximum = asfloat(inx_buffer_load(cache, 1u).z);
    InxRayBounce bounce;
    let guide = inx_ray_reflected_light(reflector, seed);
    let direct = inx_ray_direct_sample(page, attributes, secondary, inx_buffer_load(records, hit.surface_id * 3u),
        records, -ray.direction, seed, maximum, light_roughness, guide, sample_index, bounce);
    if (!direct.available)
        return (InxRaySignal)0;
    float3 tail = float3(0.0);
    float cone = guide.alpha > 0.0 ? 0.0 : inx_ray_tail_cone(secondary);
    if (cone > 0.0)
        tail = inx_ray_cached_tail(page, hit, attributes, cache, -ray.direction, cone);
    else if (bounce.hit.triangle_index != ray_invalid_index)
    {
        // Reuse the endpoint already traced for emission. Only reflected D(z) belongs to the tail.
        let next = inx_ray_direct_fresh(page, bounce.attributes, bounce.surface,
            inx_buffer_load(records, bounce.hit.surface_id * 3u), records, bounce.outgoing,
            inx_ray_sample_hash(seed ^ 0x68e31da4u), maximum);
        if (next.available)
            tail = bounce.throughput * next.radiance;
    }
    return inx_ray_signal_output((emission_weight * secondary.emission + direct.radiance + tail) * weight,
                                  uint2(hit.surface_id, asuint(hit.distance)));
}

InxRaySignal inx_ray_radiance(RayPage page, TraceRay ray, InxBuffer_uint4 records, InxBuffer_uint4 cache,
                              uint seed, float3 weight, float light_roughness, float4 reflector, uint sample_index,
                              float emission_pdf = 0.0, float3 emitter_origin = float3(0.0))
{
    RayHit hit = inx_trace_lit_closest(page, ray, 0xffffffffu, 0.0);
    return inx_ray_radiance_at_hit(page, ray, hit, records, cache, seed, weight, light_roughness,
                                   reflector, sample_index, emission_pdf, emitter_origin);
}

public float4 inx_ray_thin_reflection(float3 position_ws, float3 normal_ws, InxBuffer_uint4 records,
                                     float2 pixel_position, InxBuffer_uint4 cache)
{
    RayPage page = ray_page();
    if (page.status != ray_page_ready)
        return float4(0.0);
    float3 toward_eye = normalize(inx_camera_toward_eye(position_ws));
    float3 normal = dot(normal_ws, toward_eye) < 0.0 ? -normal_ws : normal_ws;
    TraceRay ray = {inx_ray_origin(position_ws, normal), 0.0, reflect(-toward_eye, normal),
                    inx_view.projection_params.y};
    uint2 pixel = uint2(pixel_position);
    uint seed = inx_ray_sample_hash(pixel.x ^ inx_ray_sample_hash(pixel.y) ^ inx_view_sample_index());
    return inx_ray_radiance(page, ray, records, cache, seed, float3(1.0), 0.0, float4(0.0), 0u).radiance;
}

uint2 inx_ray_path_key(float2 position, bool specular, bool temporal)
{
    uint2 pixel = uint2(position);
    uint index = temporal ? inx_view_sample_index() : 0u;
    if (specular)
    {
        uint seed = inx_ray_sample_hash(pixel.x ^ inx_ray_sample_hash(pixel.y));
        return uint2(inx_ray_sample_hash(seed), index);
    }
    // Interleave a 16-point set on the trace grid. The reservoir carries the
    // original rank and scramble so a neighbor replays exactly the same sample.
    uint2 lane = pixel & 3u;
    uint rank = ((lane.x & 1u) << 3u) | (((lane.x ^ lane.y) & 1u) << 2u) |
                (lane.x & 2u) | ((lane.x ^ lane.y) >> 1u);
    uint2 tile = pixel >> 2u;
    uint scramble = inx_ray_sample_hash(tile.x ^ inx_ray_sample_hash(tile.y));
    return uint2((scramble & ~15u) | rank, index);
}

float3 inx_screen_ray_normal(uint word)
{
    float2 oct = unpackHalf2x16ToFloat(word);
    float3 normal = float3(oct, 1.0 - abs(oct.x) - abs(oct.y));
    if (normal.z < 0.0)
        normal.xy = (1.0 - abs(normal.yx)) * select(normal.xy >= 0.0, float2(1.0), float2(-1.0));
    return normalize(normal);
}

bool inx_inside_screen_ray(float3 ndc)
{
    return all(abs(ndc.xy) < 1.0) && ndc.z > 0.0 && ndc.z <= 1.0;
}

struct InxScreenDiffuseTrace
{
    float distance;
    float2 uv;
    bool hit;
};

InxRaySignal inx_reproject_diffuse_radiance(TraceRay ray, InxScreenDiffuseTrace hit, InxUInt2D primary,
                                          InxUInt2D reprojection, InxHistory2D history, uint2 full)
{
    uint2 pixel = min(uint2(hit.uv * float2(full)), full - 1u);
    uint2 reproj = reprojection.texture.Load(int3(pixel, 0)).xy;
    if (reproj.y == 0u)
        return (InxRaySignal)0;
    uint4 surface = primary.texture.Load(int3(pixel, 0));
    float3 normal = inx_screen_ray_normal(surface.z);
    if (dot(normal, ray.direction) >= 0.0)
        return (InxRaySignal)0;

    float2 previous_uv = hit.uv - unpackHalf2x16ToFloat(reproj.x);
    if (any(previous_uv < 0.0) || any(previous_uv >= 1.0))
        return (InxRaySignal)0;
    uint2 at = min(uint2(previous_uv * float2(full)), full - 1u);
    return {float4(history.texture.Load(int3(at, 0)).rgb, 1.0), uint2(surface.x, asuint(hit.distance))};
}

float inx_load_screen_depth(InxUInt2D depths, int2 pixel, int2 extent)
{
    return asfloat(depths.texture.Load(int3(clamp(pixel, int2(0), extent - 1), 0)).x);
}

struct InxDiffuseDepthTest
{
    float clearance;
    float penetration;
};

InxDiffuseDepthTest inx_diffuse_screen_depth(InxUInt2D depths, int2 extent, float3 ndc, bool linear_test)
{
    float2 pixel = (ndc.xy * 0.5 + 0.5) * float2(extent);
    float nearest;
    float linear;
    // R32Float is not filterable on all WebGPU adapters.
    if (linear_test)
    {
        float2 center = pixel - 0.5;
        int2 base = int2(floor(center));
        float2 f = frac(center);
        float a = inx_load_screen_depth(depths, base, extent);
        float b = inx_load_screen_depth(depths, base + int2(1, 0), extent);
        float c = inx_load_screen_depth(depths, base + int2(0, 1), extent);
        float d = inx_load_screen_depth(depths, base + int2(1, 1), extent);
        nearest = f.y < 0.5 ? (f.x < 0.5 ? a : b) : (f.x < 0.5 ? c : d);
        linear = lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
    }
    else
    {
        nearest = inx_load_screen_depth(depths, int2(floor(pixel)), extent);
        linear = nearest;
    }
    float nearest_inv = rcp(max(nearest, 1.0e-12));
    float linear_inv = rcp(max(linear, 1.0e-12));
    float ray_inv = rcp(max(ndc.z, 1.0e-12));
    return {max(nearest_inv, linear_inv) * 1.000002 - ray_inv, ray_inv - min(nearest_inv, linear_inv)};
}

float3 inx_clip_diffuse_end(float3 start, float3 end, float limit)
{
    float3 delta = end - start;
    float3 lower = float3(-1.0, -1.0, 0.0);
    float3 advance = select(delta < 0.0, start - lower, 1.0 - start);
    float3 boundary = select(abs(delta) > 1.0e-10, advance / max(abs(delta), 1.0e-10), float3(1.0e20));
    return start + delta * max(0.0, min(limit, min(boundary.x, min(boundary.y, boundary.z))));
}

bool inx_march_diffuse_segment(float3 start, float3 end, InxUInt2D depths, int2 extent, float jitter,
                               float thickness, bool near_segment, out float3 crossing)
{
    float2 pixels = (end.xy - start.xy) * (0.5 * float2(extent));
    int count = int(clamp(length(pixels), 2.0, 8.0));
    float stride = rcp(float(count - int(end.z <= 9.5367432e-7)));
    float before = 0.0;
    float clearance = 0.0;
    [loop] for (int i = 0; i < count; ++i)
    {
        float after = (float(i) + jitter) * stride;
        if (!near_segment)
            after *= after;
        after = min(after, 0.99998474);
        let behind = inx_diffuse_screen_depth(depths, extent, lerp(start, end, after), near_segment);
        if (behind.clearance >= 0.0)
        {
            before = after;
            clearance = behind.clearance;
            continue;
        }
        float penetration = behind.penetration;
        float back_clearance = behind.clearance;
        [unroll] for (int j = 0; j < 2; ++j)
        {
            float middle = 0.5 * (before + after);
            let sample = inx_diffuse_screen_depth(depths, extent, lerp(start, end, middle), near_segment);
            if (sample.clearance < 0.0)
            {
                after = middle;
                back_clearance = sample.clearance;
                penetration = sample.penetration;
            }
            else
            {
                before = middle;
                clearance = sample.clearance;
            }
        }
        float fraction = clearance / (clearance - back_clearance);
        float candidate = lerp(before, after, fraction);
        let sample = inx_diffuse_screen_depth(depths, extent, lerp(start, end, candidate), near_segment);
        if (abs(sample.clearance) < 0.9 * clearance)
        {
            after = candidate;
            back_clearance = sample.clearance;
            penetration = sample.penetration;
        }
        if (penetration < thickness && back_clearance < thickness)
        {
            crossing = lerp(start, end, after);
            return true;
        }
        break;
    }
    crossing = lerp(start, end, before);
    return false;
}

InxScreenDiffuseTrace inx_march_diffuse_screen(TraceRay ray, float4x4 vp, InxUInt2D depths,
                                              uint2 extent, uint2 pixel, uint frame)
{
    float4 clip = mul(vp, float4(ray.origin, 1.0));
    if (clip.w <= 0.0 || !inx_inside_screen_ray(clip.xyz / clip.w))
        return (InxScreenDiffuseTrace)0;
    float3 start = clip.xyz / clip.w;
    float4 travel = mul(vp, float4(ray.direction, 0.0));
    float4 probe = clip + 0.5 * travel;
    float w = probe.w >= 0.0 ? max(probe.w, 1.0e-10) : min(probe.w, -1.0e-10);
    float3 near_end = probe.xyz / w;
    near_end = start + (near_end - start) * (near_end.z < 0.0 ? -1.0 : 1.0);
    near_end = inx_clip_diffuse_end(start, near_end, 1.0);
    float jitter = frac(frac(52.982918 * frac(dot(float2(pixel), float2(0.06711056, 0.00583715)))) +
                        float(frame & 3u) * 0.25);
    float thickness = 0.5 * abs(inx_view.depth_params.z);
    float3 crossing;
    bool hit = inx_march_diffuse_segment(start, near_end, depths, int2(extent), jitter, thickness, true, crossing);
    if (!hit)
    {
        start = crossing;
        float4 projected = float4(start, 1.0) + travel;
        float denominator = projected.w >= 0.0 ? max(projected.w, 1.0e-10) : min(projected.w, -1.0e-10);
        float3 far_end = inx_clip_diffuse_end(start, projected.xyz / denominator, 1.0e20);
        hit = inx_march_diffuse_segment(start, far_end, depths, int2(extent), jitter, thickness, false, crossing);
    }

    float4 world = mul(inx_view.inverse_view_projection, float4(crossing, 1.0));
    float distance = length(world.xyz / world.w - ray.origin);
    return {min(distance, ray.maximum), saturate(crossing.xy * 0.5 + 0.5),
            hit && distance > ray.minimum && distance < ray.maximum};
}

InxRaySignal inx_ray_diffuse_incident(RayPage page, TraceRay ray, InxUInt2D primary, InxUInt2D depths,
                                     InxHistory2D history, InxUInt2D reprojection,
                                     InxBuffer_uint4 records, InxBuffer_uint4 cache,
                                     uint seed, uint index, float pdf, float3 receiver, uint2 pixel, bool temporal)
{
    float4x4 vp = mul(inx_view.projection, inx_view.view);
    uint2 extent = inx_image_dimensions(depths, 0u);
    let screen = inx_march_diffuse_screen(ray, vp, depths, extent, pixel, index);
    if (screen.hit && temporal && history.valid)
    {
        uint2 full = inx_image_dimensions(primary, 0u);
        let light = inx_reproject_diffuse_radiance(ray, screen, primary, reprojection, history, full);
        if (light.radiance.a > 0.0)
            return light;
    }
    // Advance the triangle intersection interval while keeping all hit distances relative to the receiver.
    ray.minimum = min(max(0.1, screen.distance), ray.maximum);
    RayHit hit = inx_trace_lit_closest(page, ray, 0xffffffffu, 0.0);
    float roughness = rsqrt(sqrt(max(402.1238597 * pdf - 1.0, 1.0)));
    return inx_ray_radiance_at_hit(page, ray, hit, records, cache, seed, float3(1.0), roughness,
                                   float4(0.0), index, 0.0, receiver);
}

// The per-view sequence advances over the Hilbert order of each 64x64 ray tile.
uint inx_diffuse_hilbert(uint2 pixel)
{
    uint2 point = pixel & 63u;
    uint index = 0u;
    for (uint step = 32u; step > 0u; step >>= 1u)
    {
        uint2 bits = uint2((point & step) != uint2(0u));
        index += step * step * ((3u * bits.x) ^ bits.y);
        if (bits.y == 0u)
        {
            if (bits.x != 0u)
                point ^= 63u;
            point = point.yx;
        }
    }
    return index;
}

public InxRaySignal inx_ray_diffuse_sample(float3 position, float3 geometric_normal, float3 shading_normal,
                                         uint2 cell, bool temporal,
                                         InxUInt2D primary, InxUInt2D screen_depth, InxHistory2D history,
                                         InxUInt2D reprojection, InxBuffer_uint4 records, InxBuffer_uint4 cache)
{
    RayPage page = ray_page();
    if (page.status != ray_page_ready)
        return (InxRaySignal)0;
    uint frame = temporal ? inx_view_sample_index() : 0u;
    uint index = inx_diffuse_hilbert(cell) + 288u * (frame & 63u);
    float2 u = frac(0.5 + float2(0.7548776865, 0.5698403120) * float(index));
    float z = 1.0 - u.x;
    float hemisphere_radius = sqrt(max(0.0, 1.0 - z*z));
    float angle = 6.28318530718 * u.y;

    float hemisphere_sign = shading_normal.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (hemisphere_sign + shading_normal.z);
    float b = shading_normal.x * shading_normal.y * a;
    float3 tangent = float3(1.0 + hemisphere_sign * shading_normal.x * shading_normal.x * a,
                            hemisphere_sign * b, -hemisphere_sign * shading_normal.x);
    float3 bitangent = float3(b, hemisphere_sign + shading_normal.y * shading_normal.y * a, -shading_normal.y);
    float3 direction = shading_normal * z + (tangent * cos(angle) + bitangent * sin(angle)) * hemisphere_radius;
    uint3 packed = uint3(clamp(direction, -1.0, 1.0) * 511.0 + 511.5);
    uint axis = packed.x | (packed.y << 10u) | (packed.z << 20u);
    // Keep blocked directions in the sample count so the hemisphere PDF stays unchanged.
    if (dot(direction, geometric_normal) <= 0.0)
        return inx_ray_signal_output(float3(0.0), uint2(axis, 0u));

    TraceRay ray = {position, 0.0, direction, inx_view.projection_params.y};
    const float pdf = 0.15915494309;
    let hit = inx_ray_diffuse_incident(page, ray, primary, screen_depth, history, reprojection, records, cache,
                                       inx_ray_sample_hash(index), frame, pdf, position, cell * 2u, temporal);
    if (hit.radiance.a <= 0.0)
        return (InxRaySignal)0;

    float3 color = max(hit.radiance.rgb, 0.0);
    float peak = max(color.r, max(color.g, color.b));
    color *= min(1.0, 10.0 / max(peak, 1.0e-20));
    return {float4(color / pdf, 1.0), uint2(axis, hit.witness.y)};
}

RayDirectionSample inx_ray_primary_direction(InxSurfaceData surface, InxShadingContext context,
                                              float3 geometric_normal, bool specular, bool temporal, uint2 key,
                                              out uint seed, out float4 reflector)
{
    uint rank = key.x & 15u;
    // Shift both coordinates over the unit square. A receiver visible only on
    // a thin edge can still sample the whole hemisphere without its neighbors.
    float2 offset = float2(float(rank) * (1.0 / 16.0), float(rank) * 0.61803398875);
    float2 uniform = specular ? inx_ray_primary_uniform(key.y, key.x) :
                                frac(inx_ray_primary_uniform(key.y, key.x >> 4u) + offset);
    seed = specular ? key.x : inx_ray_sample_hash(key.x);
    if (temporal)
        seed ^= inx_ray_sample_hash(key.y);
    float3 shading_normal = normalize(surface.normal_ws);
    reflector = float4(0.0);
    RayDirectionSample sample;
    if (specular)
    {
        float3 outgoing = normalize(inx_camera_toward_eye(context.position_ws));
        RayGgxMaterial material = ray_lit_ggx(surface.albedo, surface.metallic, surface.smoothness,
                                              context.normal_variance, surface.specular_highlights);
        reflector = float4(shading_normal, material.alpha);
        sample = ray_sample_ggx(uniform, shading_normal, geometric_normal, outgoing, material);
    }
    else
        sample = ray_sample_irradiance(uniform, shading_normal, geometric_normal);
    return sample;
}

float3 inx_ray_primary_emissive(RayPage page, InxSurfaceData surface, InxShadingContext context,
    float3 geometric, bool specular, uint seed, uint index)
{
    float3 normal = normalize(surface.normal_ws);
    float3 outgoing = normalize(inx_camera_toward_eye(context.position_ws));
    RayGgxMaterial material = ray_lit_ggx(surface.albedo, surface.metallic, surface.smoothness,
                                          context.normal_variance, surface.specular_highlights);
    if (specular && material.alpha == 0.0)
        return float3(0.0);
    let sample = inx_ray_emitter_sample(page, context.position_ws,
        ray_reflected_uniform3(seed ^ 0xa9d53e47u, index), inx_view.projection_params.y);
    if (!sample.active || dot(geometric, sample.direction) <= 0.0)
        return float3(0.0);
    float cosine = max(dot(normal, sample.direction), 0.0);
    float bsdf_pdf = cosine * 0.31830988618379067154;
    float3 response = float3(cosine);
    if (specular)
    {
        let terms = ray_evaluate_ggx(normal, geometric, outgoing, sample.direction, material);
        bsdf_pdf = terms.pdf;
        response = terms.projected_bsdf;
    }
    if (!inx_ray_emitter_visible(page, inx_ray_origin(context.position_ws, geometric), sample))
        return float3(0.0);
    return sample.emission * response * (inx_ray_mis_weight(sample.pdf, bsdf_pdf) / sample.pdf);
}

public InxRaySignal inx_ray_replay_path(InxSurfaceData surface, InxShadingContext context, float3 geometric_normal,
                                        InxBuffer_uint4 records, uint model_id, bool specular, bool temporal,
                                        InxBuffer_uint4 cache, uint2 key)
{
    if (model_id != INX_PBR_MODEL_ID)
        return (InxRaySignal)0;
    uint seed;
    float4 reflector;
    RayDirectionSample sample = inx_ray_primary_direction(surface, context, geometric_normal, specular,
                                                          temporal, key, seed, reflector);
    RayPage page = ray_page();
    if (page.status != ray_page_ready)
        return (InxRaySignal)0;
    uint index = key.y;
    float3 emission = inx_ray_primary_emissive(page, surface, context, geometric_normal, specular, seed, index);
    if (!sample.active)
        return inx_ray_signal_output(emission, uint2(ray_invalid_index, 0u));
    TraceRay ray = {inx_ray_origin(context.position_ws, geometric_normal), 0.0, sample.direction, inx_view.projection_params.y};
    // One of N directions represents solid angle 1/(N*pdf). Project the cap's
    // half-angle onto microfacet slopes for the next analytic light connection.
    // alpha = tan(theta/2), perceptual roughness = sqrt(alpha). Emission and
    // BSDF-sampled endpoints retain the authored material and its matching PDF.
    // Match the footprint to the reconstruction budget. After geometry changes,
    // at most eight reflected samples can contribute to the displayed history.
    bool steady_visibility = world_ray_geometry[1].y == inx_previous_ray_geometry_serial();
    float sample_count = steady_visibility ? (specular ? 96.0 : 64.0) : (specular ? 8.0 : 4.0);
    float cap = max(4.0 * 3.14159265359 * sample_count * sample.pdf - 1.0, 1.0);
    float light_roughness = temporal && !sample.delta ? rsqrt(sqrt(cap)) : 0.0;
    InxRaySignal signal = inx_ray_radiance(page, ray, records, cache, seed, sample.weight, light_roughness, reflector,
                                           index, sample.delta ? 0.0 : sample.pdf, context.position_ws);
    signal.radiance.rgb += emission;
    return signal;
}

public InxRaySignal inx_ray_indirect_signal(InxSurfaceData surface, InxShadingContext context, float3 geometric_normal,
                                            InxBuffer_uint4 records, uint model_id, bool specular, bool temporal,
                                            InxBuffer_uint4 cache, uint2 trace_pixel)
{
    float2 position = specular || !temporal ? context.pixel_position : float2(trace_pixel);
    return inx_ray_replay_path(surface, context, geometric_normal, records, model_id, specular, temporal, cache,
                               inx_ray_path_key(position, specular, temporal));
}

// A positive target leaves support for directions that were black before a lighting or visibility change.
float inx_ray_path_target(float3 radiance)
{
    float3 magnitude = abs(radiance);
    float peak = max(magnitude.r, max(magnitude.g, magnitude.b));
    return 0.125 + peak / (1.0 + peak);
}

public InxRayPathEstimate inx_ray_path_temporal(InxSurfaceData surface, InxShadingContext context, float3 normal,
    InxBuffer_uint4 records, uint model_id, bool specular, InxBuffer_uint4 cache, uint4 history, uint2 trace_pixel,
    float4 fresh_radiance, uint2 fresh_witness)
{
    uint2 key = inx_ray_path_key(specular ? context.pixel_position : float2(trace_pixel), specular, true);
    InxRaySignal fresh = {fresh_radiance, fresh_witness};
    if (fresh.radiance.w <= 0.0)
        return {fresh, uint4(0), float3(0.0), float3(0.0)};

    if ((history.w & 7u) == 0u)
    {
        if (specular)
            return {fresh, uint4(key, asuint(1.0), 1u), fresh.radiance.rgb, float3(0.0)};
        // A disoccluded diffuse pixel uses the complementary Sobol direction as a unit-mass proposal.
        history = uint4(key ^ uint2(0u, 1u), asuint(1.0), 1u);
    }

    InxRaySignal replay = inx_ray_replay_path(surface, context, normal, records, model_id, specular, true,
                                              cache, history.xy);
    uint represented = min(history.w & 7u, 4u);
    uint count = 1u + represented;
    float mass = float(represented) * asfloat(history.z);
    float3 initial = fresh.radiance.rgb;
    float initial_target = inx_ray_path_target(initial);
    float replay_target = inx_ray_path_target(replay.radiance.rgb);
    float replay_score = mass * replay_target;
    float total = initial_target + replay_score;
    uint bits = inx_ray_sample_hash(key.x ^ inx_ray_sample_hash(key.y) ^ (2u * 0xb92367d5u));
    bool take_replay = float(bits >> 8u) * (1.0 / 16777216.0) * total < replay_score;
    float target = take_replay ? replay_target : initial_target;
    uint4 reservoir = uint4(take_replay ? history.xy : key, asuint(total / (float(count) * target)), count);
    fresh.radiance.rgb = (initial + mass * replay.radiance.rgb) / float(count);
    return {fresh, reservoir, take_replay ? replay.radiance.rgb : initial, replay.radiance.rgb};
}

public int2 inx_ray_path_neighbor(uint2 cell)
{
    // A shifted 2x2 tile is paired by an involution. Both pixels read the same temporal publication.
    uint shift = inx_ray_sample_hash(inx_view_sample_index() ^ 0x39faa78du);
    uint2 q = cell + uint2(shift & 1u, (shift >> 1u) & 1u);
    uint2 tile = q >> 1u;
    uint axis = inx_ray_sample_hash(tile.x ^ inx_ray_sample_hash(tile.y ^ shift)) & 3u;
    uint2 mask = axis == 0u ? uint2(1, 0) : (axis == 1u ? uint2(0, 1) : uint2(1, 1));
    return int2(cell) + int2(q ^ mask) - int2(q);
}

public uint2 inx_ray_tail_pixel(uint index, uint2 extent)
{
    uint seed = inx_ray_sample_hash(index ^ inx_ray_sample_hash(inx_view_sample_index()));
    return uint2(seed % extent.x, inx_ray_sample_hash(seed) % extent.y);
}

InxRayTailRequest inx_ray_tail_request(RayPage page, TraceRay ray)
{
    InxRayTailRequest request = (InxRayTailRequest)0;
    if (page.status != ray_page_ready)
        return request;
    RayHit hit = inx_trace_lit_closest(page, ray, 0xffffffffu, 0.0);
    if (hit.triangle_index == ray_invalid_index ||
        inx_ray_participation(hit.surface_id) != inx_ray_material_lit)
        return request;
    RaySurfaceHit attributes = ray_surface_at_hit(page, ray, hit);
    InxSurfaceData surface;
    inx_evaluate_ray_lit_hit(attributes, hit.surface_id, 0.0, surface);
    float cone = inx_ray_tail_cone(surface);
    if (cone == 0.0)
        return request;
    float3 geometric = attributes.front_face ? attributes.geometric_normal_ws : -attributes.geometric_normal_ws;
    request.identity = uint4(hit.instance_index, hit.triangle_index, asuint(hit.barycentrics));
    request.anchor = uint4(asuint(attributes.position_ws), packHalf2x16(float2(0.0, cone)) | 1u);
    request.outgoing = uint4(asuint(-ray.direction), hit.surface_id);
    request.normal = uint4(asuint(geometric), hit.primitive);
    return request;
}

public InxRayTailRequest inx_ray_tail_primary_request(InxSurfaceData surface, InxShadingContext context,
    float3 geometric_normal, uint model_id, bool specular, bool temporal)
{
    if (model_id != INX_PBR_MODEL_ID)
        return (InxRayTailRequest)0;
    uint seed;
    float4 reflector;
    uint2 key = inx_ray_path_key(context.pixel_position, specular, temporal);
    RayDirectionSample sample = inx_ray_primary_direction(surface, context, geometric_normal, specular,
                                                          temporal, key, seed, reflector);
    if (!sample.active)
        return (InxRayTailRequest)0;
    TraceRay ray = {inx_ray_origin(context.position_ws, geometric_normal), 0.0, sample.direction,
                    inx_view.projection_params.y};
    return inx_ray_tail_request(ray_page(), ray);
}

public InxRayTailRequest inx_ray_tail_thin_request(float3 position, float3 normal)
{
    float3 outgoing = normalize(inx_camera_toward_eye(position));
    normal = dot(normal, outgoing) < 0.0 ? -normal : normal;
    TraceRay ray = {inx_ray_origin(position, normal), 0.0, reflect(-outgoing, normal), inx_view.projection_params.y};
    return inx_ray_tail_request(ray_page(), ray);
}

public float4 inx_ray_tail_estimate(InxBuffer_uint4 requests, InxBuffer_uint4 cache,
                                     InxBuffer_uint4 records, uint index)
{
    // The native owner supplies one stable representative and one complete control header.
    uint4 identity = inx_buffer_load(requests, 4u + index * 4u);
    uint4 anchor = inx_buffer_load(requests, 5u + index * 4u);
    if ((anchor.w & 1u) == 0u)
        return float4(0.0, 0.0, 0.0, -1.0);
    uint4 outgoing = inx_buffer_load(requests, 6u + index * 4u);
    uint4 normal = inx_buffer_load(requests, 7u + index * 4u);
    RayPage page = ray_page();
    if (page.status != ray_page_ready)
        return float4(0.0, 0.0, 0.0, -1.0);
    uint slot = anchor.w >> 2u;
    uint count = 0u;
    if ((anchor.w & 2u) == 0u)
        count = inx_buffer_load(cache, 4u + slot * 8u).w;
    float3 wo = asfloat(outgoing.xyz);
    TraceRay at_anchor = {asfloat(anchor.xyz), 0.0, -wo, 0.0};
    RayHit hit = {0.0, asfloat(identity.zw), identity.x, identity.y, normal.w, outgoing.w};
    RaySurfaceHit attributes = ray_surface_at_hit(page, at_anchor, hit);
    InxSurfaceData surface;
    inx_evaluate_ray_lit_hit(attributes, hit.surface_id, 0.0, surface);
    uint4 world = inx_buffer_load(requests, 3u);
    uint seed = inx_ray_sample_hash(slot ^ inx_ray_sample_hash(count) ^ world.x ^ inx_ray_sample_hash(world.y));
    InxRayTransportTerm result = inx_ray_tail_sample(page, attributes, surface, records, wo, seed, asfloat(world.z));
    return float4(result.radiance, result.available ? result.distance : -1.0);
}
