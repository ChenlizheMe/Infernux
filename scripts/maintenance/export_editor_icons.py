"""Export authored editor SVGs to the shipped PNGs (development tool only)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", help="SVG stems under editor_icons/")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    app = QGuiApplication([])
    source = Path(__file__).with_name("editor_icons")
    output = Path(__file__).resolve().parents[2] / "python/infernux/resources/icons"
    for name in args.names:
        renderer = QSvgRenderer(str(source / f"{name}.svg"))
        if not renderer.isValid():
            raise ValueError(f"Invalid SVG: {name}")
        image = QImage(args.size, args.size, QImage.Format.Format_RGBA8888)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        renderer.render(painter, QRectF(0, 0, args.size, args.size))
        painter.end()
        target = output / f"{name}.png"
        if not image.save(str(target)):
            raise OSError(f"Cannot write icon: {target}")
        print(target)
    app.quit()


if __name__ == "__main__":
    main()
