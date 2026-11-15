"""Domain lifecycles share source construction and lifetime ownership."""


class ReloadAdapters:
    def __init__(self, adapter):
        self.items = [adapter]

    def add(self, adapter):
        self.items.insert(0, adapter)

    def remove(self, adapter):
        self.items.remove(adapter)

    def guid_for_path(self, path):
        return next((guid for item in self.items if (guid := item.guid_for_path(path))), "")

    def retire(self, paths):
        failures = []
        for item in self.items:
            try:
                item.retire(paths)
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ExceptionGroup("Source retirement failed", failures)

    def validate_sources(self, paths):
        for item in self.items:
            validate = getattr(item, "validate_sources", None)
            if validate is not None:
                validate(paths)

    def record_dependencies(self, graph):
        for item in self.items:
            record = getattr(item, "record_dependencies", None)
            if record is not None:
                record(graph)

    def prepare(self, modules, owner):
        return [(item, item.prepare(modules, owner)) for item in self.items]

    def bind(self, prepared):
        for item, value in prepared:
            item.bind(value)

    def initialize(self, prepared, owner):
        for item, value in prepared:
            item.initialize(value, owner)

    def phase(self, prepared, owner, name):
        for item, value in prepared:
            callback = getattr(item, name, None)
            if callback is not None:
                callback(value, owner)

    def activate(self, prepared):
        for item, value in prepared:
            item.activate(value)

    def fail(self, paths, error):
        for item in self.items:
            item.fail(paths, error)
