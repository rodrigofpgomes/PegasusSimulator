"""MATLAB 'parula' colour map for the reference speed (dark blue at rest -> yellow at cruise).

If utils/parula256.csv exists (exact table: in MATLAB, writematrix(parula(256), 'parula256.csv')) it is used;
otherwise the map is interpolated linearly between parula's anchor colours (the two end colours are exact, the
interior ones are approximate).
"""

from pathlib import Path

import numpy as np

_ANCHORS = np.array([
    [0.2422, 0.1504, 0.6603],
    [0.2810, 0.3228, 0.9579],
    [0.1786, 0.5289, 0.9682],
    [0.0689, 0.6948, 0.8394],
    [0.2161, 0.7843, 0.5923],
    [0.6720, 0.7793, 0.2227],
    [0.9970, 0.7659, 0.2199],
    [0.9769, 0.9839, 0.0805],
])

_TABLE_FILE = Path(__file__).resolve().parent / "parula256.csv"


def _table():
    if _TABLE_FILE.is_file():
        return np.loadtxt(_TABLE_FILE, delimiter=",")
    x = np.linspace(0.0, 1.0, len(_ANCHORS))
    xs = np.linspace(0.0, 1.0, 256)
    return np.stack([np.interp(xs, x, _ANCHORS[:, k]) for k in range(3)], axis=1)


_TABLE = _table()


def parula(value, vmin=0.0, vmax=12.0, alpha=1.0):
    """RGBA tuple of `value` on the parula scale [vmin, vmax] (clipped)."""
    u = float(np.clip((value - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0))
    r, g, b = _TABLE[int(round(u * (len(_TABLE) - 1)))]
    return (float(r), float(g), float(b), alpha)
