"""Error metrics from Onyekpe et al. 2021 (Appl. Sci. 11, 1270).

All three operate on a sequence of per-second errors e_k over one outage.

  CRSE  cumulative root squared error : sum |e_k|
  CAE   cumulative absolute error     : sum e_k, SIGNED, so over- and
        under-estimates cancel and a systematic bias is visible
  AEPS  average error per second      : mean |e_k|

Ground-truth distance between GPS fixes is geodesic (Vincenty family).
"""

from __future__ import annotations

import numpy as np
from pyproj import Geod

# pyproj's Geod.inv agrees with the `vincenty` package to well under a
# centimetre (pinned in tests/test_sanity.py) and is vectorised.
_GEOD = Geod(ellps="WGS84")


def geodesic_distance_m(lat1, lon1, lat2, lon2):
    """Geodesic distance in metres. Scalars or arrays."""
    return _GEOD.inv(lon1, lat1, lon2, lat2)[2]


def crse(error_vectors) -> float:
    """Cumulative root squared error: || sum_k e_k ||, the accumulated
    2-D position error at the end of the outage.

    NOTE on the definition. Reading CRSE as the scalar sum of per-second
    DISPLACEMENT errors, sum |e_k|, does not reproduce the paper: it makes the
    motorway the WORST scenario and the roundabout among the best, inverting
    the published difficulty ordering. Taking the norm of the accumulated
    error VECTOR reproduces motorway-lowest / roundabout-highest and brings the
    magnitudes into line (motorway 34.6 vs 30.11 published; roundabout 201.9 vs
    171.92). The distinction matters because a scalar sum ignores heading: on a
    roundabout, per-second distance errors stay small while the track bends
    away from truth, and only the vector form sees that.

    Accepts either an (n, 2) array of per-second error components, or a 1-D
    array of already-accumulated position-error magnitudes (in which case the
    final value is the cumulative error).
    """
    e = np.asarray(error_vectors, dtype=float)
    if e.ndim == 2 and e.shape[1] == 2:
        total = np.nansum(e, axis=0)
        return float(np.hypot(total[0], total[1]))
    e = e[np.isfinite(e)]
    return float(e[-1]) if e.size else float("nan")


def crse_scalar(errors) -> float:
    """The literal sum |e_k| over per-second displacement errors.

    Retained because it is the natural reading of the formula and is a useful
    diagnostic -- it isolates distance error from heading error -- but it is
    NOT the quantity the paper tabulates. See `crse`.
    """
    e = np.asarray(errors, dtype=float)
    return float(np.nansum(np.abs(e)))


def cae(errors) -> float:
    """Cumulative absolute error, SIGNED.

    Kept signed deliberately: a large CRSE with a near-zero CAE is random
    scatter, while CRSE ~= |CAE| is a systematic bias in one direction.
    """
    e = np.asarray(errors, dtype=float)
    return float(np.nansum(e))


def aeps(errors) -> float:
    """Average error per second: mean |e_k|."""
    e = np.asarray(errors, dtype=float)
    e = e[np.isfinite(e)]
    return float(np.mean(np.abs(e))) if e.size else float("nan")


def summarise(displacement_errors, position_errors=None) -> dict:
    """All three metrics for one outage.

    `displacement_errors` are the signed per-second distance errors (CAE,
    AEPS); `position_errors` are the accumulated 2-D position error magnitudes
    (CRSE). If position errors are absent, CRSE falls back to the scalar form.
    """
    out = {"cae": cae(displacement_errors),
           "aeps": aeps(displacement_errors),
           "crse_scalar": crse_scalar(displacement_errors),
           "n_seconds": int(np.size(displacement_errors))}
    out["crse"] = (crse(position_errors) if position_errors is not None
                   else crse_scalar(displacement_errors))
    return out
