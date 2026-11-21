// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// Adapted from MuJoCo Warp _src/util_misc.py, _poly_force_deriv.

float dampingDerivative(float3 coefficients, float velocity)
{
    const float speed = abs(velocity);
    return coefficients.x + 2.0 * coefficients.y * speed + 3.0 * coefficients.z * speed * speed;
}
