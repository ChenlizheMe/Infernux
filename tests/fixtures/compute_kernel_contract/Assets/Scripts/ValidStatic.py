"""Small cross-target GPU kernel contract fixture."""

import infernux as inx


class JellyKernel:
    @staticmethod
    @inx.compute.kernel
    def step(domain):
        index = inx.compute.index(domain)
        domain[index] = 1
