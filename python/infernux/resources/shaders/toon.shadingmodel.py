import infernux.shader as shader


class Toon(shader.shadingmodel):
    shader_id = "infernux/shading/toon"
    hidden = True
    lighting_contract = "custom_response"

    def band(self, value: float, threshold: float, softness: float) -> float:
        width = shader.max(softness, 0.0001)
        return shader.smoothstep(threshold - width, threshold + width, value)

    def shade_light(self, surface: shader.SurfaceData, ctx: shader.ShadingContext, light: shader.LightData) -> shader.vec3:
        normal = shader.normalize(surface.normal_ws)
        view = shader.lighting.view_direction(ctx)
        ndot_l = shader.max(shader.dot(normal, light.direction), 0.0)
        threshold = shader.saturate(surface.shading_param0)
        softness = shader.clamp(surface.shading_param1, 0.0, 0.25)
        diffuse_band = self.band(ndot_l, threshold, softness)
        halfway = view + light.direction
        halfway = halfway * shader.rsqrt(shader.max(shader.dot(halfway, halfway), 0.00000001))
        exponent = shader.lerp(12.0, 256.0, shader.saturate(surface.smoothness))
        raw_specular = shader.pow(shader.max(shader.dot(normal, halfway), 0.0), exponent)
        specular_band = self.band(raw_specular, 0.5, softness * 0.5)
        diffuse = shader.saturate(surface.albedo) * shader.lerp(0.18, 1.0, diffuse_band)
        specular = shader.vec3(specular_band * shader.saturate(surface.specular_highlights))
        radiance = light.color * light.attenuation * light.shadow
        return (diffuse + specular) * radiance * shader.step(0.0001, ndot_l)

    def shade(self, surface: shader.SurfaceData, ctx: shader.ShadingContext) -> shader.vec3:
        normal = shader.normalize(surface.normal_ws)
        ambient = shader.lighting.ambient(ctx, normal) * surface.albedo * shader.saturate(surface.occlusion) * 0.65
        return shader.lighting.direct(surface, ctx) + ambient + surface.emission
