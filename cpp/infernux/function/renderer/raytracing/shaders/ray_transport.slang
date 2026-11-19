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
    uint x = reversebits(index);
    uint y = x;
    y ^= (y & 0x55555555u) << 1u;
    y ^= (y & 0x33333333u) << 2u;
    y ^= (y & 0x0f0f0f0fu) << 4u;
    y ^= (y & 0x00ff00ffu) << 8u;
    y ^= y << 16u;
    uint2 shift = uint2(inx_ray_sample_hash(pixel_seed), inx_ray_sample_hash(pixel_seed + 1u));
    return float2((uint2(x, y) ^ shift) >> 8u) * (1.0 / 16777216.0);
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

TraceRay inx_ray_light_visibility(float3 origin, InxTransportLight light)
{
    if (light.directional)
        return {origin, 0.0, light.sample.direction, 3.402823e38};
    float3 to_light = light.endpoint - origin;
    if (dot(to_light, light.sample.direction) <= 0.0)
        return {origin, 0.0, light.sample.direction, 0.0};
    float distance = length(to_light);
    // Leave the point/spot/rectangle endpoint outside the occlusion interval.
    return {origin, 0.0, to_light / distance, asfloat(asuint(distance) - 1u)};
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

bool inx_ray_emitter_visible(RayPage page, float3 origin, InxRayEmitterSample sample)
{
    float3 normal = sample.attributes.front_face ? sample.attributes.geometric_normal_ws
                                                : -sample.attributes.geometric_normal_ws;
    float3 target = inx_ray_origin(sample.attributes.position_ws, normal, sample.attributes.position_error_ws);
    float3 delta = target - origin;
    float distance = length(delta);
    if (!(distance > 0.0))
        return false;
    TraceRay visibility = {origin, 0.0, delta / distance, asfloat(asuint(distance) - 1u)};
    return inx_trace_lit_visibility(page, visibility, 0xffffffffu, 0.0);
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
        float distance = length(link);
        if (distance > 0.0)
        {
            TraceRay visibility = {origin, 0.0, link / distance, asfloat(asuint(distance) - 1u)};
            if (!inx_trace_lit_visibility(page, visibility, 0xffffffffu, 0.0))
                continue;
        }
        return asfloat(inx_buffer_load(cache, base + 5u).xyz);
    }
    return float3(0.0);
}

InxRaySignal inx_ray_radiance(RayPage page, TraceRay ray, InxBuffer_uint4 records, InxBuffer_uint4 cache,
                              uint seed, float3 weight, float light_roughness, float4 reflector, uint sample_index,
                              float emission_pdf = 0.0, float3 emitter_origin = float3(0.0))
{
    RayHit hit = inx_trace_lit_closest(page, ray, 0xffffffffu, 0.0);
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
                                            InxBuffer_uint4 cache)
{
    return inx_ray_replay_path(surface, context, geometric_normal, records, model_id, specular, temporal, cache,
                               inx_ray_path_key(context.pixel_position, specular, temporal));
}

// A positive target leaves support for directions that were black before a lighting or visibility change.
float inx_ray_path_target(float3 radiance)
{
    float3 magnitude = abs(radiance);
    float peak = max(magnitude.r, max(magnitude.g, magnitude.b));
    return 0.125 + peak / (1.0 + peak);
}

public InxRayPathEstimate inx_ray_path_temporal(InxSurfaceData surface, InxShadingContext context, float3 normal,
    InxBuffer_uint4 records, uint model_id, bool specular, InxBuffer_uint4 cache, uint4 history, uint2 trace_pixel)
{
    uint2 key = inx_ray_path_key(specular ? context.pixel_position : float2(trace_pixel), specular, true);
    // The fourth reservoir word carries the count [0:2], selected luminance [3:16],
    // and a signed temporal gradient [17:31]. Both guides retain a float's exponent.
    uint previous_count = history.w & 7u;
    uint previous_luma = (history.w << 14u) & 0x7ffe0000u;
    float gradient = 0.0;
    uint selected_luma = 0u;
    // Without a reprojected diffuse proposal, evaluate the complementary Sobol
    // direction in the same two-candidate budget. A fresh proposal has unit mass.
    if (!specular && previous_count == 0u)
        history = uint4(key ^ uint2(0u, 1u), asuint(1.0), 1u);
    uint represented = min(history.w & 7u, 4u);
    uint count = 1u + represented;
    uint2 selected = key;
    float selected_target = 1.0;
    float total = 0.0;
    float3 estimate = float3(0.0);
    InxRaySignal result = (InxRaySignal)0;
    // Use one traversal call site for the bounded stream on both the native and browser GPU compilers.
    for (uint i = 0u; i < (represented == 0u ? 1u : 2u); ++i)
    {
        uint2 candidate = i == 0u ? key : history.xy;
        InxRaySignal sample = inx_ray_replay_path(surface, context, normal, records, model_id, specular, true,
                                                  cache, candidate);
        if (i == 0u)
        {
            result = sample;
            if (result.radiance.w <= 0.0)
                return {result, uint4(0)};
        }
        float mass = i == 0u ? 1.0 : float(represented) * asfloat(history.z);
        float target = inx_ray_path_target(sample.radiance.rgb);
        uint luma = 0u;
        if (!specular)
        {
            float brightness = max(dot(sample.radiance.rgb, float3(0.2126, 0.7152, 0.0722)), 0.0);
            luma = asuint(brightness) & 0x7ffe0000u;
            if (i == 1u && previous_count != 0u)
                gradient = asfloat(history.z) * (asfloat(luma) - asfloat(previous_luma));
        }
        float score = mass * target;
        total += score;
        uint bits = inx_ray_sample_hash(key.x ^ inx_ray_sample_hash(key.y) ^ ((i + 1u) * 0xb92367d5u));
        if (float(bits >> 8u) * (1.0 / 16777216.0) * total < score)
        {
            selected_luma = luma;
            selected = candidate;
            selected_target = target;
        }
        estimate += mass * sample.radiance.rgb;
    }
    // Only proposal keys recur. Keep the hit of the fresh ray available to filters and project code.
    result.radiance.rgb = estimate / float(count);
    uint evidence = (asuint(gradient) & 0xfffe0000u) | (selected_luma >> 14u) | count;
    return {result, uint4(selected, asuint(total / (float(count) * selected_target)), evidence)};
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
