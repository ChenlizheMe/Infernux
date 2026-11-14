import infernux.shader as shader


class ParticleSixWaySmoke(shader.particle):
    shader_id = "infernux/particle/six_way_smoke"
    surface_type = "transparent"
    cull_mode = "none"
    blend = "premultiply"
    cast_shadows = False
    base_color = shader.material.color(default=(0.66, 0.66, 0.66, 1.0))
    emission_color = shader.material.color(default=(1.0, 0.28, 0.04, 1.0), hdr=True)
    lighting_intensity = shader.material.float(default=1.0, range=(0.0, 8.0))
    ambient_intensity = shader.material.float(range=(0.0, 0.002))
    ambient_saturation = shader.material.float(range=(0.0, 1.0))
    emission_intensity = shader.material.float(range=(0.0, 32.0))
    absorption = shader.material.float(default=0.5, range=(0.0, 1.0))
    alpha_scale = shader.material.float(default=1.0, range=(0.0, 4.0))
    frame_jitter = shader.material.float(default=3.0, range=(0.0, 16.0))
    frame_offset = shader.material.float(range=(0.0, 1024.0))
    fade_in_fraction = shader.material.float(default=0.08, range=(0.0, 0.5))
    fade_out_start = shader.material.float(default=0.68, range=(0.5, 1.0))
    density_clip_threshold = shader.material.float(default=0.025, range=(0.0, 0.5))
    positive_axes_map = shader.material.texture_2d()
    negative_axes_map = shader.material.texture_2d(default="black")

    def particle_hash(self, value: shader.uint) -> float:
        value = value ^ (value >> shader.uint(16))
        value = value * shader.uint(0x7FEB352D)
        value = value ^ (value >> shader.uint(15))
        value = value * shader.uint(0x846CA68B)
        value = value ^ (value >> shader.uint(16))
        return float(value) * (1.0 / 4294967295.0)

    def flipbook_uv(self, uv: shader.vec2, frame: float, grid: shader.vec2) -> shader.vec2:
        column = frame - shader.floor(frame / grid.x) * grid.x
        cell = shader.vec2(column, shader.floor(frame / grid.x))
        extent = shader.max(shader.vec2(shader.dimensions(self.positive_axes_map, 0)), shader.vec2(1.0))
        inset = 0.5 * grid / extent
        return (shader.clamp(uv, inset, shader.vec2(1.0) - inset) + cell) / grid

    def main(self, ctx: shader.ParticleContext) -> shader.vec4:
        grid = shader.max(ctx.flipbook_grid, shader.vec2(1.0))
        frame_count = grid.x * grid.y
        jitter = (self.particle_hash(ctx.particle_id) - 0.5) * self.frame_jitter
        frame = shader.clamp(ctx.normalized_age * (frame_count - 1.0) + self.frame_offset + jitter, 0.0, frame_count - 1.0)
        first = shader.floor(frame)
        second = shader.min(first + 1.0, frame_count - 1.0)
        blend = shader.frac(frame)
        first_uv = self.flipbook_uv(ctx.local_uv, first, grid)
        second_uv = self.flipbook_uv(ctx.local_uv, second, grid)
        gradient_x = shader.ddx(ctx.local_uv) / grid
        gradient_y = shader.ddy(ctx.local_uv) / grid
        positive = shader.lerp(
            shader.sample_grad(self.positive_axes_map, first_uv, gradient_x, gradient_y),
            shader.sample_grad(self.positive_axes_map, second_uv, gradient_x, gradient_y), blend)
        fade_in = shader.smoothstep(0.0, shader.max(self.fade_in_fraction, 0.0001), ctx.normalized_age)
        fade_out = 1.0 - shader.smoothstep(self.fade_out_start, 1.0, ctx.normalized_age)
        lifetime_fade = fade_in * fade_out
        density_alpha = shader.max(positive.a, 0.0) * self.alpha_scale * lifetime_fade
        shader.clip(density_alpha - self.density_clip_threshold)
        negative = shader.lerp(
            shader.sample_grad(self.negative_axes_map, first_uv, gradient_x, gradient_y),
            shader.sample_grad(self.negative_axes_map, second_uv, gradient_x, gradient_y), blend)
        dynamic_coverage = shader.saturate(self.alpha_scale * lifetime_fade * self.base_color.a)
        alpha = shader.saturate(density_alpha * self.base_color.a)
        rgb = shader.lighting.six_way_smoke(
            ctx, shader.max(positive.rgb, shader.vec3(0.0)) * dynamic_coverage,
            shader.max(negative.rgb, shader.vec3(0.0)) * dynamic_coverage,
            self.base_color.rgb * ctx.color, alpha, self.absorption, self.lighting_intensity,
            self.ambient_intensity, self.ambient_saturation)
        emission = self.emission_color.rgb * negative.a * self.emission_intensity
        return shader.vec4((rgb + emission * alpha) * ctx.alpha, alpha * ctx.alpha)
