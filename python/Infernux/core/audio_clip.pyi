"""Type stubs for Infernux.core.audio_clip.

Agent note:
    The runtime wrapper currently documents and reliably supports WAV loading.
    Some older docs/UI filters may mention OGG/MP3, but the C++ decode path is
    WAV-only until additional decoders are implemented.
"""

from __future__ import annotations

from typing import Optional

from Infernux.lib import AudioClip as CppAudioClip


class AudioClip:
    """Pythonic wrapper around C++ AudioClip.

    Use ``AudioClip.load("Assets/Audio/foo.wav")`` and pass the returned
    wrapper (or ``clip.native``) to ``AudioSource.set_track_clip`` or
    ``AudioSource.play_one_shot``. Do not unload a clip while an AudioSource is
    still using it.
    """

    def __init__(self, native: CppAudioClip) -> None:
        """Wrap an existing C++ AudioClip."""
        ...

    # Factory methods
    @staticmethod
    def load(file_path: str) -> Optional[AudioClip]:
        """Load an audio clip from a file path.

        Args:
            file_path: Project or absolute path to a WAV file.

        Returns:
            ``AudioClip`` when decoding succeeds; otherwise ``None``.
        """
        ...
    @staticmethod
    def from_native(native: CppAudioClip) -> AudioClip:
        """Wrap an existing C++ AudioClip instance."""
        ...

    # Properties
    @property
    def native(self) -> CppAudioClip:
        """The underlying C++ AudioClip object."""
        ...
    @property
    def is_loaded(self) -> bool:
        """Whether the audio data is loaded in memory."""
        ...
    @property
    def duration(self) -> float:
        """Duration of the audio clip in seconds."""
        ...
    @property
    def sample_count(self) -> int:
        """Total number of audio samples."""
        ...
    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz (e.g. 44100)."""
        ...
    @property
    def channels(self) -> int:
        """Number of audio channels (1=mono, 2=stereo)."""
        ...
    @property
    def name(self) -> str:
        """The display name of the audio clip."""
        ...
    @property
    def file_path(self) -> str:
        """The file path the clip was loaded from."""
        ...

    # Context manager
    def __enter__(self) -> AudioClip:
        """Enter context manager for resource management."""
        ...
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager and unload audio data.

        Avoid context-manager lifetime for clips assigned to persistent
        AudioSource tracks, because the clip may unload while playback still
        references it.
        """
        ...
    def unload(self) -> None:
        """Unload the audio data from memory."""
        ...

    def __repr__(self) -> str: ...
