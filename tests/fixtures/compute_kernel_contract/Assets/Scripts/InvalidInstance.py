"""Intentional diagnostic fixture: an instance receiver is not serializable."""

import infernux as inx


class JellyKernel:
    @inx.compute.kernel
    def step(self, domain):
        index = inx.compute.index(domain)
        domain[index] = 1
