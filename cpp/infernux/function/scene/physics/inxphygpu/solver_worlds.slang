// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#ifdef INX_AWAKE_SOLVER_WORLDS
[[vk::binding(31, 0)]] StructuredBuffer<uint> awakeWorlds;
#endif

uint3 solverDispatchId(uint3 id)
{
#ifdef INX_AWAKE_SOLVER_WORLDS
    id.y = awakeWorlds[id.y];
#endif
    return id;
}
