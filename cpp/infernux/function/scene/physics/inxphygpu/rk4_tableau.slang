// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

static const uint Rk4FinalStage = 3;

float rk4Weight(uint stage)
{
    return stage == 0 || stage == Rk4FinalStage ? 1.0 / 6.0 : 1.0 / 3.0;
}

float rk4StepScale(uint stage)
{
    return stage < 2 ? 0.5 : 1.0;
}
