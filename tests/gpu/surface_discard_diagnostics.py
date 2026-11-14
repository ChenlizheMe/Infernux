"""Runtime-neutral evidence for the surface-discard GPU regression test."""
import json
from pathlib import Path
import struct
import zlib

import numpy as np


def coverage_metrics(images, *, depth=False):
    """Use the same masks as the regression assertions, without asserting."""
    result = {}
    if 'masked_on' not in images:
        return result
    masked = images['masked_on']
    visible = masked[..., 0] > .95
    if not depth:
        visible &= (masked[..., 1] < .5) & (masked[..., 2] > .95)
    result['visible'] = int(visible.sum())
    if 'solid' in images:
        solid = images['solid'][..., 0] > .95
        if not depth:
            solid &= (images['solid'][..., 1] < .5) & (images['solid'][..., 2] > .95)
        result['solid'] = int(solid.sum())
    if not depth and 'baseline' in images:
        delta = images['baseline'][..., 0] - masked[..., 0]
        result['shadow'] = int(((delta > .08) & ~visible).sum())
        outside = delta[~visible]
        finite = outside[np.isfinite(outside)]
        result['outside_nonfinite'] = int(outside.size - finite.size)
        if finite.size:
            result['outside_delta_min'] = float(finite.min())
            result['outside_delta_max'] = float(finite.max())
            result['outside_delta_percentiles'] = np.percentile(finite, [50, 90, 99]).tolist()
    return result


def _write_preview(path, pixels):
    """Write an RGB PNG for inspection; NPZ remains the numerical evidence."""
    rgb = np.nan_to_num(pixels[..., :3], nan=0., posinf=1., neginf=0.)
    rgb = (np.clip(rgb, 0., 1.) * 255).astype(np.uint8)
    height, width, _ = rgb.shape

    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data +
                struct.pack('>I', zlib.crc32(kind + data)))

    raw = b''.join(b'\0' + row.tobytes() for row in rgb)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' +
                     chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)) +
                     chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


class SurfaceDiscardEvidence:
    """Capture before assertions and export only on failure to limit CI I/O."""

    def __init__(self, directory, *, kind, pipeline, samples):
        self.directory = Path(directory)
        self.metadata = dict(kind=kind, pipeline=pipeline, samples=samples, frames=[])
        self.images = {}

    def record(self, step, pixels, frame):
        self.images[step] = np.array(pixels, copy=True)
        self.metadata['frames'].append(dict(step=step, **frame))

    def export_failure(self, error, logs):
        """Attach diagnostic errors as notes, never replace the original failure."""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            # Preserve all channels and floating-point precision, including NaNs.
            np.savez_compressed(self.directory / 'images.npz', **self.images)
            report = dict(self.metadata, error=repr(error), logs=logs,
                          metrics=coverage_metrics(self.images, depth=self.metadata['pipeline'] == 'depth'),
                          images={step: dict(shape=list(pixels.shape), dtype=str(pixels.dtype),
                                             nonfinite=int((~np.isfinite(pixels)).sum()))
                                  for step, pixels in self.images.items()})
            (self.directory / 'evidence.json').write_text(
                json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            for step, pixels in self.images.items():
                _write_preview(self.directory / f'{step}.png', pixels)
            error.add_note(f'Surface-discard evidence: {self.directory.resolve()}')
        except Exception as diagnostic_error:
            error.add_note(f'Surface-discard evidence export failed: {diagnostic_error!r}; '
                           f'directory={self.directory}')
