"""Symbolic enum values shared by persistence and authored field input."""

from enum import Flag


def resolve_enum_name(enum_type, name, path="value", *, casefold=False):
    members = getattr(enum_type, "__members__", {})
    if casefold:

        def lookup(key):
            exact = members.get(key)
            if exact is not None:
                return exact
            matches = [
                value
                for candidate, value in members.items()
                if candidate.casefold() == key.casefold()
            ]
            return matches[0] if len(matches) == 1 else None

    else:
        lookup = members.get
    member = lookup(name)
    if member is not None:
        return member
    is_flags = isinstance(enum_type, type) and issubclass(enum_type, Flag)
    if is_flags and name == "0":
        return enum_type(0)
    if is_flags and "|" in name:
        result = enum_type(0)
        for part in name.split("|"):
            member = lookup(part.strip())
            if member is None:
                break
            result |= member
        else:
            return result
    raise ValueError(f"{path}: unknown {enum_type.__qualname__} member {name!r}")


def enum_value_name(value, path="value"):
    kind = type(value)
    name = value.name
    if name in kind.__members__:
        return name
    if isinstance(value, Flag):
        name = "0" if value.value == 0 else name
        if name is not None and resolve_enum_name(kind, name, path) == value:
            return name
    raise ValueError(f"{path}: unknown {kind.__qualname__} enum member")
