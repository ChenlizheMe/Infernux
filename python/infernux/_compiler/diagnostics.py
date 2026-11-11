"""Source diagnostics for the statically compiled GPU language; no host imports."""

from __future__ import annotations

import ast


class ShaderSourceError(SyntaxError):
    def __init__(self, message, location, *, code="author.syntax"):
        super().__init__(message, location)
        self.diagnostics = [{
            "code": code, "severity": "error", "path": self.filename,
            "line": self.lineno, "column": self.offset,
            "end_line": self.end_lineno or self.lineno,
            "end_column": self.end_offset or self.offset,
            "message": message, "phase": "author", "target": "",
        }]


def parse_gpu_module(source: str, path: str) -> ast.Module:
    try:
        return ast.parse(source, filename=path)
    except SyntaxError as error:
        location = (path, error.lineno or 1, error.offset or 1, error.text,
                    error.end_lineno or error.lineno or 1, error.end_offset or error.offset or 1)
        raise ShaderSourceError(error.msg, location) from None
