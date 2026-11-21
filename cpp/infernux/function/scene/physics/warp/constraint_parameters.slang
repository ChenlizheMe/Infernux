// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0
// MuJoCo Warp _src/constraint.py continuous-time row parameters.

static const float ConstraintMinValue = 1.0e-15;
static const float MinImpedance = 0.0001;
static const float MaxImpedance = 0.9999;

float3 impedanceParameters(float2 reference, float4 impedance, float power, float position, float timestep,
                           bool unclampedReference)
{
    const float time = unclampedReference ? reference.x : max(reference.x, 2.0 * timestep);
    const float lower = clamp(impedance.x, MinImpedance, MaxImpedance);
    const float upper = clamp(impedance.y, MinImpedance, MaxImpedance);
    const float mid = clamp(impedance.w, MinImpedance, MaxImpedance);
    const float x = saturate(abs(position) / max(impedance.z, ConstraintMinValue));
    const float denominator = upper * upper;
    const float k = reference.x <= 0.0 ? -reference.x / max(ConstraintMinValue, denominator) :
                1.0 / max(ConstraintMinValue, denominator * time * time * reference.y * reference.y);
    const float b = reference.y <= 0.0 ? -reference.y / max(ConstraintMinValue, upper) :
                2.0 / max(ConstraintMinValue, upper * time);
    const float exponent = max(1.0, power);
    float interpolation = 0.5;
    if (lower != upper && impedance.z > ConstraintMinValue)
    {
        if (exponent == 1.0)
            interpolation = x;
        else if (x <= mid)
            interpolation = mid * pow(x / mid, exponent);
        else
            interpolation = 1.0 - (1.0 - mid) * pow((1.0 - x) / (1.0 - mid), exponent);
    }
    return float3(k, b, lerp(lower, upper, interpolation));
}

float constraintDiagonal(float inverseWeight, float impedance)
{
    return 1.0 / max(ConstraintMinValue, inverseWeight * (1.0 - impedance) / impedance);
}

float2 constraintCoupling(float4 coefficients, float fourth, float x)
{
    const float value = coefficients.x + x * (coefficients.y + x * (coefficients.z + x * (coefficients.w + x * fourth)));
    const float derivative = coefficients.y + x * (2.0 * coefficients.z + x * (3.0 * coefficients.w + x * 4.0 * fourth));
    return float2(value, derivative);
}

struct ConstraintParameters
{
    float4 impedance;
    float4 reference; // time constant and damping ratio, impedance power, unused
};

float3 impedanceParameters(ConstraintParameters parameters, float position, float timestep, bool unclampedReference)
{
    return impedanceParameters(parameters.reference.xy, parameters.impedance, parameters.reference.z,
                               position, timestep, unclampedReference);
}

ConstraintRow makeConstraintRow(uint4 layout, uint kind, float3 kbi, float position, float margin,
                                float velocity, float inverseWeight, float friction, float accelerationShift)
{
    const float stiffness = isFrictionConstraint(kind) ? 0.0 : kbi.x;
    const float reference = -stiffness * kbi.z * position - kbi.y * velocity - accelerationShift;
    return {layout, float4(position + margin, margin, velocity, reference),
            constraintDiagonal(inverseWeight, kbi.z), friction, kind};
}
