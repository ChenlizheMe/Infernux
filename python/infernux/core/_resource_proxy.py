"""One live Python proxy per native resource, without owning its lifetime."""
from __future__ import annotations

from weakref import WeakValueDictionary


class ResourceProxy:
    """Intern proxies by the identity of their retained native binding.

    Pybind reuses the Python binding of a live shared native resource. Each
    proxy retains that binding; its address cannot be reused while the proxy
    is alive. Weak values let unused proxies and their native references go
    away. GUIDs deliberately do not key this map: copies and newly published
    resources must not accidentally reuse a proxy for another native object.
    """

    _proxies: WeakValueDictionary = WeakValueDictionary()

    def __new__(cls, native):
        if native is None:
            raise ValueError("Cannot wrap a None native resource")
        key = (cls, id(native))
        proxy = ResourceProxy._proxies.get(key)
        if proxy is None:
            proxy = super().__new__(cls)
            proxy._native = native
            ResourceProxy._proxies[key] = proxy
        return proxy
