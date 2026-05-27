"""Sample the dominant pixel colors of the Synara logo.

Rasterizes the SVG to a high-res PNG, bins colors by hue/luma family,
and prints the top occupants. Useful for keeping the dashboard accent
palette semantically aligned with the brand mark.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

try:
    import cairosvg
except ImportError:
    sys.exit("Need cairosvg: uv pip install cairosvg")
try:
    from PIL import Image
except ImportError:
    sys.exit("Need Pillow: uv pip install Pillow")

SVG = Path(__file__).resolve().parent.parent / "assets" / "synara-logo.svg"
PNG = Path("/tmp/synara-logo-512.png")
SIZE = 512
MIN_ALPHA = 32  # drop near-transparent pixels (anti-aliased edges)
TOP_N = 12


def hex_of(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def quantize(rgb: tuple[int, int, int], step: int = 16) -> tuple[int, int, int]:
    return tuple((c // step) * step for c in rgb)  # type: ignore[return-value]


def main() -> None:
    cairosvg.svg2png(url=str(SVG), write_to=str(PNG), output_width=SIZE, output_height=SIZE)
    img = Image.open(PNG).convert("RGBA")
    total = 0
    bins: Counter[tuple[int, int, int]] = Counter()
    for r, g, b, a in img.getdata():
        if a < MIN_ALPHA:
            continue
        total += 1
        bins[quantize((r, g, b))] += 1

    print(f"sampled {total} opaque px at {SIZE}x{SIZE}\n")
    print(f"{'rank':>4}  {'hex':<8}  {'pct':>6}  swatch")
    for i, (rgb, count) in enumerate(bins.most_common(TOP_N), 1):
        pct = 100 * count / total
        print(f"{i:>4}  {hex_of(rgb):<8}  {pct:>5.1f}%")


if __name__ == "__main__":
    main()
