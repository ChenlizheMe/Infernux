"""Type stubs for ShaderInfo resource operations."""

class Shader:
    """Query published shaders and check/reimport static GPU author assets."""

    @staticmethod
    def check(asset: str, *, source: str | None = None) -> dict: ...

    @staticmethod
    def property_to_id(name: str) -> int:
        """Return a process-local property ID shared by all material instances."""
        ...

    @staticmethod
    def get_schema(asset: str) -> dict:
        """Query logical fields and graph ports of a static GPU author asset."""
        ...

    @classmethod
    def is_loaded(cls, name: str, shader_type: str = ...) -> bool:
        """Query GPU publication of a standalone stage or linked material program."""
        ...

    @classmethod
    def reload(cls, shader_id: str, shader_type: str | None = ...) -> bool:
        """Reimport by ShaderInfo Name or asset path; failures raise an exception."""
        ...
