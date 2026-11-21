// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

static const uint LinearDispatchGroupsPerRow = 65535;

uint4 linearDispatch<let Threads : uint>(uint work)
{
    const uint groups = work / Threads + uint(work % Threads != 0);
    return uint4(min(groups, LinearDispatchGroupsPerRow),
                 max(1, (groups + LinearDispatchGroupsPerRow - 1) / LinearDispatchGroupsPerRow), 1, 0);
}

uint linearWorkIndex<let Threads : uint>(uint3 id)
{
    return id.y * (LinearDispatchGroupsPerRow * Threads) + id.x;
}
