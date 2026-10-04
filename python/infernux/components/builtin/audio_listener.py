"""
AudioListener — Python BuiltinComponent wrapper for C++ AudioListener.

Represents the "ears" in the scene. Typically attached to the main camera.
Only one AudioListener stays active at a time. If another listener is
enabled, Infernux keeps the current one active, warns, and leaves the new
listener on standby so it can automatically take over later if needed.

Example::

    from infernux.components.builtin import AudioListener

    class MainCameraSetup(InxComponent):
        def start(self):
            listener = self.game_object.get_component(AudioListener)
            # AudioListener has no configurable properties;
            # its existence on a GameObject registers it as the scene listener.
"""

from __future__ import annotations

from infernux.components.builtin_component import BuiltinComponent


class AudioListener(BuiltinComponent):
    """Python wrapper for the C++ AudioListener component.

    The AudioListener marks a GameObject as the scene's audio receiver.
    Attach it to the main camera so that 3D audio is heard from that
    perspective.

    No CppProperty descriptors are needed — the component's presence
    is its only function. Serialization delegates to the C++ component.
    """

    _cpp_type_name = "AudioListener"
    _component_category_ = "Audio"

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def game_object_id(self) -> int:
        """Owning GameObject ID (for Wwise listener registration)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.game_object_id
        return 0

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> str:
        """Serialize AudioListener to JSON string (delegates to C++)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.serialize()
        return "{}"

    def deserialize(self, json_str: str) -> bool:
        """Deserialize AudioListener from JSON string (delegates to C++)."""
        cpp = self._cpp_component
        if cpp is not None:
            return cpp.deserialize(json_str)
        return False
