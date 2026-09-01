"""Build the offline OSRM map for the test-session extent.

One-time, network-using SETUP step. Nothing at inference touches the network:
this produces a local `.osrm` dataset that `mapmatch.osrm` serves from
127.0.0.1.

    python -m mapmatch.build_map

Pipeline:

1. Compute the bounding box from the four Volvo test sessions (mapmatch.extent).
2. Pick the Geofabrik UK county extracts whose own polygon intersects it, from
   the published region index. County granularity rather than
   `england-latest.osm.pbf` (1.7 GB) because the machine this was built on had
   5.3 GB free -- the counties covering the box total a small fraction of that,
   and selecting them from the index means the choice is derived, not a
   hand-written list that silently leaves a hole in the map.
3. `osmium merge` them (merge de-duplicates the objects the county extracts
   share along their boundaries), then `osmium extract --bbox` with the
   default complete_ways strategy so ways crossing the boundary stay whole.
4. `osrm-extract` with the stock car profile, then `osrm-partition` +
   `osrm-customize` for the MLD pipeline `osrm-routed --algorithm mld` serves.
5. Delete the per-county downloads and the merge, which are the bulk of the
   disk cost and are reproducible.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from .extent import BBox, MAP_DIR, TEST_SESSIONS, session_bbox, save_bbox

GEOFABRIK_INDEX = "https://download.geofabrik.de/index-v1.json"
UK_ROOT = "united-kingdom"

# Homebrew's osrm-backend ships the stock profiles here.
CAR_PROFILE_CANDIDATES = (
    Path("/opt/homebrew/share/osrm/profiles/car.lua"),
    Path("/usr/local/share/osrm/profiles/car.lua"),
    Path("/usr/share/osrm/profiles/car.lua"),
)


class BuildError(RuntimeError):
    """The offline map could not be built."""


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise BuildError(
            f"{binary} not found on PATH. Install the toolchain first:\n"
            f"    brew install osrm-backend osmium-tool")
    return path


def car_profile() -> Path:
    for p in CAR_PROFILE_CANDIDATES:
        if p.exists():
            return p
    found = list(Path("/opt/homebrew/Cellar/osrm-backend").rglob("profiles/car.lua"))
    if found:
        return found[0]
    raise BuildError("could not locate the OSRM car.lua profile")


def _run(cmd: list[str], **kw) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}", file=sys.stderr)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


# -- region selection ------------------------------------------------------

def _geometry_bbox(geom: dict) -> BBox:
    """Bounding box of a GeoJSON (Multi)Polygon."""
    lons, lats = [], []

    def walk(node):
        if (isinstance(node, (list, tuple)) and len(node) == 2
                and all(isinstance(v, (int, float)) for v in node)):
            lons.append(float(node[0]))
            lats.append(float(node[1]))
            return
        for child in node:
            walk(child)

    walk(geom["coordinates"])
    return BBox(min(lons), min(lats), max(lons), max(lats))


def select_regions(index: dict, box: BBox) -> list[tuple[str, str]]:
    """Leaf UK regions intersecting `box`, as (id, pbf_url).

    Leaves only: the index lists `england` alongside its counties, and taking
    both would download the 1.7 GB parent to no purpose. A region is a leaf
    when nothing names it as a parent.
    """
    feats = {}
    parents = set()
    for f in index["features"]:
        p = f["properties"]
        feats[p["id"]] = f
        if p.get("parent"):
            parents.add(p["parent"])

    def under_uk(rid: str) -> bool:
        seen = set()
        while rid and rid not in seen:
            if rid == UK_ROOT:
                return True
            seen.add(rid)
            rid = feats.get(rid, {}).get("properties", {}).get("parent")
        return False

    picked = []
    for rid, f in feats.items():
        if rid in parents or not under_uk(rid):
            continue                      # not a leaf, or not in the UK tree
        geom = f.get("geometry")
        if not geom:
            continue
        if _geometry_bbox(geom).intersects(box):
            url = f["properties"].get("urls", {}).get("pbf")
            if url:
                picked.append((rid, url))
    return sorted(picked)


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}", file=sys.stderr)
        return
    print(f"  get  {dest.name}", file=sys.stderr)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
        shutil.copyfileobj(r, out, length=1 << 20)
    tmp.rename(dest)


# -- pipeline --------------------------------------------------------------

def build(map_dir: Path = MAP_DIR, margin_km: float = 5.0,
          keep_downloads: bool = False) -> Path:
    """Build the OSRM dataset. Returns the path to the `.osrm` base."""
    for b in ("osmium", "osrm-extract", "osrm-partition", "osrm-customize"):
        _require(b)

    map_dir.mkdir(parents=True, exist_ok=True)
    dl = map_dir / "downloads"
    dl.mkdir(exist_ok=True)

    box = session_bbox(TEST_SESSIONS, margin_km=margin_km)
    save_bbox(box)
    print(f"extent from {', '.join(TEST_SESSIONS)}: {box.osmium_arg()}",
          file=sys.stderr)

    print("selecting Geofabrik regions...", file=sys.stderr)
    with urllib.request.urlopen(GEOFABRIK_INDEX, timeout=120) as r:
        index = json.load(r)
    regions = select_regions(index, box)
    if not regions:
        raise BuildError("no Geofabrik region intersects the test extent")
    print(f"  {len(regions)}: {', '.join(r for r, _ in regions)}", file=sys.stderr)

    parts = []
    for rid, url in regions:
        dest = dl / f"{rid}.osm.pbf"
        _download(url, dest)
        parts.append(dest)

    merged = map_dir / "merged.osm.pbf"
    crop = map_dir / "extent.osm.pbf"
    if len(parts) == 1:
        merged = parts[0]
    else:
        print("merging...", file=sys.stderr)
        _run(["osmium", "merge", "--overwrite", *parts, "-o", merged])

    print("cropping to extent...", file=sys.stderr)
    _run(["osmium", "extract", "--overwrite", "--bbox", box.osmium_arg(),
          "--strategy", "complete_ways", merged, "-o", crop])

    if merged != parts[0] and merged.exists():
        merged.unlink()
    if not keep_downloads:
        shutil.rmtree(dl, ignore_errors=True)

    print("osrm-extract...", file=sys.stderr)
    _run(["osrm-extract", "-p", car_profile(), crop])
    base = crop.with_suffix("").with_suffix("")      # extent.osm.pbf -> extent
    osrm = map_dir / f"{base.name}.osrm"
    print("osrm-partition / osrm-customize...", file=sys.stderr)
    _run(["osrm-partition", osrm])
    _run(["osrm-customize", osrm])

    print(f"\nbuilt {osrm}", file=sys.stderr)
    return osrm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--margin-km", type=float, default=5.0)
    ap.add_argument("--keep-downloads", action="store_true",
                    help="keep the per-county .osm.pbf downloads (disk cost)")
    args = ap.parse_args(argv)
    try:
        build(margin_km=args.margin_km, keep_downloads=args.keep_downloads)
    except (BuildError, subprocess.CalledProcessError) as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
