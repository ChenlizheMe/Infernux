"""Resource dependencies and last consumers, independent of devices and queues.

The caller supplies accesses from its compiled operations. Completion means the
actual work finished, never merely that a command was prepared or accepted.
"""
from dataclasses import dataclass
import heapq


@dataclass(frozen=True)
class Access:
    reads: frozenset
    writes: frozenset
    after: tuple = ()

    @property
    def resources(self):
        return self.reads | self.writes


class Dataflow:
    """A finite dependency graph over opaque resource identities.

    RAW/WAR/WAW edges preserve declared versions of mutable storage. Disjoint
    work stays independent. Logical values can use their value IDs as resources;
    lowered bindings can use allocation IDs for conservative alias ordering.
    """

    def __init__(self, accesses, *, initialized):
        self.accesses = tuple(accesses)
        writers, readers, uses = {}, {}, {}
        dependencies = []
        initialized = frozenset(initialized)
        for index, access in enumerate(self.accesses):
            needs = set(access.after)
            for resource in access.reads:
                writer = writers.get(resource)
                if writer is None and resource not in initialized:
                    raise ValueError("graph reads a private value before its producer")
                if writer is not None:
                    needs.add(writer)
            for resource in access.writes:
                if resource in writers:
                    needs.add(writers[resource])
                needs.update(readers.get(resource, ()))
            for resource in access.reads - access.writes:
                readers.setdefault(resource, set()).add(index)
            for resource in access.writes:
                writers[resource] = index
                readers.pop(resource, None)
            for resource in access.resources:
                uses.setdefault(resource, []).append(index)
            dependencies.append(frozenset(needs))
        count = len(self.accesses)
        if any(not 0 <= parent < count for needs in dependencies for parent in needs):
            raise ValueError("graph dependency refers to an absent operation")
        children = [[] for _ in self.accesses]
        for index, needs in enumerate(dependencies):
            for parent in needs:
                children[parent].append(index)
        self.dependencies = tuple(dependencies)
        self.children = tuple(tuple(nodes) for nodes in children)
        self.consumers = {resource: tuple(nodes) for resource, nodes in uses.items()}
        # Validate once at construction. Runtime completion only decrements
        # counts and opens successors; it never rebuilds the graph.
        remaining = [len(needs) for needs in dependencies]
        ready = [index for index, needs in enumerate(remaining) if not needs]
        heapq.heapify(ready)
        order = []
        while ready:
            index = heapq.heappop(ready)
            order.append(index)
            for child in self.children[index]:
                remaining[child] -= 1
                if not remaining[child]:
                    heapq.heappush(ready, child)
        if len(order) != count:
            raise ValueError("graph dependencies contain a cycle")
        self.order = tuple(order)

    def cursor(self, *, retained=()):
        return Cursor(self, retained=retained)


class Cursor:
    """One run's readiness and lifetime facts; scheduling remains with its owner."""

    def __init__(self, graph, *, retained):
        self.graph = graph
        self.remaining = [len(needs) for needs in graph.dependencies]
        self.ready = [index for index, needs in enumerate(self.remaining) if not needs]
        heapq.heapify(self.ready)
        self.uses = {resource: len(nodes) for resource, nodes in graph.consumers.items()}
        self.retained = frozenset(retained)
        self.completed = 0
        self.index = heapq.heappop(self.ready) if self.ready else len(graph.accesses)

    @property
    def done(self):
        return self.completed == len(self.graph.accesses)

    def complete(self):
        if self.done:
            raise RuntimeError("graph has no active operation to complete")
        index = self.index
        released = []
        for resource in self.graph.accesses[index].resources:
            self.uses[resource] -= 1
            if not self.uses[resource] and resource not in self.retained:
                released.append(resource)
        self.completed += 1
        for child in self.graph.children[index]:
            self.remaining[child] -= 1
            if not self.remaining[child]:
                heapq.heappush(self.ready, child)
        self.index = heapq.heappop(self.ready) if self.ready else len(self.graph.accesses)
        return tuple(released)
