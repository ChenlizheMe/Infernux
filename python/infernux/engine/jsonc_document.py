"""Edit top-level JSONC properties without reformatting unrelated source text."""
from __future__ import annotations

from dataclasses import dataclass
import json


def _reject_constant(value: str):
    raise ValueError(f"Invalid JSON constant: {value}")


_DECODER = json.JSONDecoder(parse_constant=_reject_constant)


def _skip_space(text: str, position: int) -> int:
    while position < len(text) and text[position] in " \t\r\n":
        position += 1
    return position


def _masked_source(source: str) -> tuple[str, str]:
    """Keep offsets while masking comments and then JSONC trailing commas."""
    masked = list(source)
    if source.startswith("\ufeff"):
        masked[0] = " "
    commas = []
    position = 0
    while position < len(source):
        if source[position] == '"':
            _, position = _DECODER.raw_decode(source, position)
            continue
        if source.startswith("//", position):
            end = position + 2
            while end < len(source) and source[end] not in "\r\n":
                end += 1
        elif source.startswith("/*", position):
            closing = source.find("*/", position + 2)
            if closing < 0:
                raise ValueError(f"Unterminated JSONC comment at offset {position}")
            end = closing + 2
        else:
            if source[position] == ",":
                commas.append(position)
            position += 1
            continue
        for index in range(position, end):
            if masked[index] not in "\r\n":
                masked[index] = " "
        position = end
    comments_masked = "".join(masked)
    for comma in commas:
        following = _skip_space(comments_masked, comma + 1)
        previous = comma - 1
        while previous >= 0 and comments_masked[previous] in " \t\r\n":
            previous -= 1
        if (following < len(source) and comments_masked[following] in "}]"
                and previous >= 0 and comments_masked[previous] not in "{[,:"):
            masked[comma] = " "
    return comments_masked, "".join(masked)


@dataclass(frozen=True)
class _Member:
    key: str
    start: int
    value_start: int
    end: int
    comma: int | None


class JsoncDocument:
    """A validated object with spans for surgical top-level property edits."""

    def __init__(self, source: str):
        self.source = source
        masked, normalized = _masked_source(source)
        self.value = _DECODER.decode(normalized)
        if not isinstance(self.value, dict):
            raise ValueError("IDE configuration must be a JSON object")
        self.members: list[_Member] = []
        position = _skip_space(masked, 0) + 1
        while True:
            position = _skip_space(masked, position)
            if masked[position] == "}":
                self.close = position
                break
            start = position
            key, position = _DECODER.raw_decode(normalized, position)
            colon = _skip_space(masked, position)
            value_start = _skip_space(masked, colon + 1)
            _, end = _DECODER.raw_decode(normalized, value_start)
            separator = _skip_space(masked, end)
            comma = separator if masked[separator] == "," else None
            self.members.append(_Member(key, start, value_start, end, comma))
            position = separator + 1 if comma is not None else separator

    def render(self, updated: dict) -> str:
        """Return original text with only changed properties edited."""
        edits: list[tuple[int, int, str]] = []
        remaining = []
        # JSON configuration readers use the final declaration of a key.
        # Preserve shadowed declarations when updating its effective value.
        effective = {member.key: member for member in self.members}
        for member in self.members:
            if member.key not in updated:
                edits.append((member.start, member.end, ""))
                if member.comma is not None:
                    edits.append((member.comma, member.comma + 1, ""))
            else:
                remaining.append(member)
                if member is effective[member.key] and updated[member.key] != self.value[member.key]:
                    encoded = json.dumps(updated[member.key], ensure_ascii=False, allow_nan=False)
                    edits.append((member.value_start, member.end, encoded))
        additions = [key for key in updated if key not in self.value]
        trailing_comma = bool(self.members and self.members[-1].comma is not None)
        if remaining:
            last = remaining[-1]
            if additions and last.comma is None:
                edits.append((last.end, last.end, ","))
            elif not additions and not trailing_comma and last.comma is not None:
                edits.append((last.comma, last.comma + 1, ""))
        if additions:
            newline = "\r\n" if "\r\n" in self.source else "\n"
            line_start = self.source.rfind("\n", 0, self.close) + 1
            prefix = self.source[line_start:self.close]
            insertion = line_start if not prefix.strip() else self.close
            base_indent = prefix if not prefix.strip() else ""
            indent = base_indent + "    "
            if self.members:
                first = self.members[0].start
                previous_line = self.source.rfind("\n", 0, first) + 1
                first_indent = self.source[previous_line:first]
                if not first_indent.strip() and len(first_indent) > len(base_indent):
                    indent = first_indent
            content = ("," + newline).join(
                indent + json.dumps(key, ensure_ascii=False) + ": "
                + json.dumps(updated[key], ensure_ascii=False, allow_nan=False)
                for key in additions
            )
            if trailing_comma:
                content += ","
            if insertion and self.source[insertion - 1] != "\n":
                content = newline + content
            content += newline
            edits.append((insertion, insertion, content))
        result = self.source
        # Equal-offset insertions retain their authored order: a separating
        # comma must precede additions even for a compact {"last": 1} object.
        for start, end, replacement in reversed(sorted(edits, key=lambda edit: (edit[0], edit[1]))):
            result = result[:start] + replacement + result[end:]
        return result
