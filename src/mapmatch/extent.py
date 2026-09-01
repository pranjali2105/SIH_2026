"""The geographic extent the offline map must cover.

Derived from the GPS of the four Volvo test sessions (S-A5..S-A8) rather than
hard-coded, so the map and the split cannot drift apart: if the test role ever
changes, the extract is rebuilt from the sessions actually in it.

The four sessions span south-west England and the west Midlands --
lat 50.41..52.49, lon -5.03..-1.41, roughly 250 x 230 km. That is a large
box, but it is the box the benchmark actually drives in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = REPO_ROOT / "data" / "osm"
BBOX_JSON = MAP_DIR / "bbox.json"

# Sessions the offline map has to cover: the `test` role.
TEST_SESSIONS = ("S-A5", "S-A6", "S-A7", "S-A8")

# Padding on every side. Roads crossing the boundary are kept whole by
# osmium's complete_ways strategy, but a margin means a vehicle near the edge
# still sees the junctions ahead of it rather than a truncated stub.
DEFAULT_MARGIN_KM = 5.0

_KM_PER_DEG_LAT = 111.32


@dataclass(frozen=True)
class BBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def osmium_arg(self) -> str:
        """`--bbox` argument: left,bottom,right,top."""
        return (f"{self.min_lon:.6f},{self.min_lat:.6f},"
                f"{self.max_lon:.6f},{self.max_lat:.6f}")

    def contains(self, lat: float, lon: float) -> bool:
        return (self.min_lat <= lat <= self.max_lat
                and self.min_lon <= lon <= self.max_lon)

    def intersects(self, other: "BBox") -> bool:
        return not (other.min_lon > self.max_lon or other.max_lon < self.min_lon
                    or other.min_lat > self.max_lat or other.max_lat < self.min_lat)

    def padded(self, margin_km: float) -> "BBox":
        dlat = margin_km / _KM_PER_DEG_LAT
        mid = 0.5 * (self.min_lat + self.max_lat)
        dlon = margin_km / (_KM_PER_DEG_LAT * max(np.cos(np.radians(mid)), 1e-6))
        return BBox(self.min_lon - dlon, self.min_lat - dlat,
                    self.max_lon + dlon, self.max_lat + dlat)


def session_bbox(names=TEST_SESSIONS, margin_km: float = DEFAULT_MARGIN_KM,
                 data_root=None) -> BBox:
    """Padded bounding box of the GPS traces of `names`.

    Zero lat/lon are dropped: they are the sentinel for "no fix", not a
    position in the Gulf of Guinea.
    """
    from data.loader import load_session, _find, DATA_ROOT

    root = Path(data_root) if data_root is not None else DATA_ROOT
    lats, lons = [], []
    for name in names:
        s = load_session(name, data_root=root, check_rate=False)
        lat_c, lon_c = _find(s.df, r"LATITUDE"), _find(s.df, r"LONGITUDE")
        if lat_c is None or lon_c is None:
            raise ValueError(f"{name} has no GPS columns")
        la = pd.to_numeric(s.df[lat_c], errors="coerce").to_numpy(float)
        lo = pd.to_numeric(s.df[lon_c], errors="coerce").to_numpy(float)
        ok = np.isfinite(la) & np.isfinite(lo) & (la != 0) & (lo != 0)
        if not ok.any():
            raise ValueError(f"{name} has no usable GPS fixes")
        lats.append(la[ok])
        lons.append(lo[ok])
    la = np.concatenate(lats)
    lo = np.concatenate(lons)
    raw = BBox(float(lo.min()), float(la.min()), float(lo.max()), float(la.max()))
    return raw.padded(margin_km)


def load_bbox(path: Path = BBOX_JSON) -> BBox:
    """The bbox the installed map was actually built for."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build the offline map first:\n"
            f"    python -m mapmatch.build_map")
    return BBox(**json.loads(path.read_text()))


def save_bbox(box: BBox, path: Path = BBOX_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(box), indent=2))
