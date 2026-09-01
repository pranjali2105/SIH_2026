"""Client for a LOCAL osrm-routed process.

Offline by construction: the server is a child process of this one, bound to
127.0.0.1, serving a dataset built ahead of time by `mapmatch.build_map`. A
non-loopback host is refused rather than used, so "no network at inference"
is enforced by the code and not merely intended.

Only two services are needed. `/nearest` answers "which road segment is this
position on, and what are the plausible alternatives" -- both the match and
the ambiguity test. `/match` is exposed for snapping a whole predicted
trajectory, which is what the every-10-seconds re-match uses when it has a
run of positions rather than a single point.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .extent import MAP_DIR

DEFAULT_DATASET = MAP_DIR / "extent.osrm"
LOOPBACK = ("127.0.0.1", "localhost", "::1")

STARTUP_TIMEOUT_S = 120.0
REQUEST_TIMEOUT_S = 10.0


class OSRMUnavailable(RuntimeError):
    """The local routing engine is not usable."""


class OfflineViolation(RuntimeError):
    """Something asked this client to talk to a non-local host."""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class OSRMClient:
    """Owns an `osrm-routed` child process and queries it over loopback.

    Usable as a context manager. Starting the engine costs a few seconds
    (memory-mapping the dataset), so one client should be shared across a
    whole evaluation rather than created per session.
    """

    def __init__(self, dataset: Path = DEFAULT_DATASET, port: int | None = None,
                 algorithm: str = "mld", threads: int | None = None,
                 start: bool = True):
        self.dataset = Path(dataset)
        self.algorithm = algorithm
        self.threads = threads or max(1, (os.cpu_count() or 4) // 2)
        self.port = port or _free_port()
        self.host = "127.0.0.1"
        self.proc: subprocess.Popen | None = None
        self._log = None
        if start:
            self.start()

    # -- lifecycle ---------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> "OSRMClient":
        if self.proc is not None:
            return self
        if shutil.which("osrm-routed") is None:
            raise OSRMUnavailable(
                "osrm-routed not on PATH — `brew install osrm-backend`")
        if not self.dataset.with_suffix(".osrm.fileIndex").exists() \
                and not self.dataset.exists():
            raise OSRMUnavailable(
                f"no OSRM dataset at {self.dataset} — build it first:\n"
                f"    python -m mapmatch.build_map")
        log_path = self.dataset.parent / "osrm-routed.log"
        self._log = open(log_path, "wb")
        cmd = ["osrm-routed", "--algorithm", self.algorithm,
               "--ip", self.host, "--port", str(self.port),
               "--threads", str(self.threads), str(self.dataset)]
        print(f"starting: {' '.join(cmd)}", file=sys.stderr)
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=self._log)
        atexit.register(self.close)
        self._await_ready(log_path)
        return self

    def _await_ready(self, log_path: Path) -> None:
        """Poll a trivial query until the engine answers or the process dies."""
        deadline = time.time() + STARTUP_TIMEOUT_S
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                tail = log_path.read_bytes()[-2000:].decode("utf-8", "replace")
                raise OSRMUnavailable(
                    f"osrm-routed exited with {self.proc.returncode}:\n{tail}")
            try:
                self._get("nearest", "0,0", {})
                return
            except OSRMUnavailable:
                time.sleep(0.25)
            except urllib.error.HTTPError:
                return          # answering, even if it dislikes 0,0
        raise OSRMUnavailable(
            f"osrm-routed did not become ready within {STARTUP_TIMEOUT_S:.0f}s")

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> "OSRMClient":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- transport ---------------------------------------------------------

    def _get(self, service: str, coords: str, params: dict) -> dict:
        if self.host not in LOOPBACK:
            raise OfflineViolation(
                f"OSRM host is {self.host!r}; inference must not leave the "
                "machine. Build the extract offline instead.")
        qs = urllib.parse.urlencode({k: v for k, v in params.items()
                                     if v is not None})
        url = f"{self.base_url}/{service}/v1/driving/{coords}"
        if qs:
            url += f"?{qs}"
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                raise OSRMUnavailable(f"{url} -> HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, ConnectionError, socket.timeout) as exc:
            raise OSRMUnavailable(f"{url} unreachable: {exc}") from exc

    # -- services ----------------------------------------------------------

    def nearest(self, lat: float, lon: float, number: int = 1,
                bearing: float | None = None, bearing_range: int = 60,
                radius_m: float | None = None) -> list[dict]:
        """Snap one position. Returns up to `number` candidate waypoints.

        Each waypoint carries `location` ([lon, lat]), `distance` (metres from
        the query), `name` (street name) and `nodes` (the two OSM node ids of
        the segment snapped to) -- the last is what anchors the match onto an
        exact edge of our own graph.

        `bearing` (degrees clockwise from north) restricts candidates to
        segments running roughly the way the vehicle is going, which is the
        difference between snapping to a carriageway and snapping to the one
        beside it going the other way.
        """
        params: dict = {"number": max(1, int(number))}
        if bearing is not None:
            params["bearings"] = f"{int(round(bearing)) % 360},{int(bearing_range)}"
        if radius_m is not None:
            params["radiuses"] = f"{radius_m:g}"
        res = self._get("nearest", f"{lon:.7f},{lat:.7f}", params)
        if res.get("code") != "Ok":
            return []
        return res.get("waypoints", [])

    def match(self, coords: list[tuple[float, float]],
              radius_m: float | None = None,
              timestamps: list[int] | None = None) -> dict:
        """Map-match a trajectory of (lat, lon). Returns the raw response."""
        s = ";".join(f"{lon:.7f},{lat:.7f}" for lat, lon in coords)
        params: dict = {"geometries": "geojson", "overview": "full",
                        "annotations": "nodes"}
        if radius_m is not None:
            params["radiuses"] = ";".join([f"{radius_m:g}"] * len(coords))
        if timestamps is not None:
            params["timestamps"] = ";".join(str(int(t)) for t in timestamps)
        return self._get("match", s, params)
