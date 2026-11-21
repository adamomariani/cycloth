"""Geocoding (PDOK Locatieserver) and bicycle routing (OSRM bike profile).

Both services are free and keyless:
  * PDOK Locatieserver - official Dutch address / place geocoder (Kadaster)
  * routing.openstreetmap.de/routed-bike - FOSSGIS-hosted OSRM, bicycle profile
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import requests

PDOK_URL = "https://api.pdok.nl/bzk/locatieserver/search/v3_1/suggest"
OSRM_URL = "https://routing.openstreetmap.de/routed-bike/route/v1/driving/{coords}"
TIMEOUT = 20
NL_BBOX = (50.70, 3.20, 53.70, 7.30)   # lat_min, lon_min, lat_max, lon_max


@dataclass
class Place:
    label: str
    lat: float
    lon: float


def in_netherlands(lat: float, lon: float) -> bool:
    a, b, c, d = NL_BBOX
    return a <= lat <= c and b <= lon <= d


def geocode(query: str, rows: int = 6) -> list[Place]:
    """Free-text search; returns best matches (addresses, streets, towns, postcodes)."""
    query = (query or "").strip()
    if not query:
        return []
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", query)
    if m:  # raw "lat, lon"
        return [Place(f"{float(m[1]):.5f}, {float(m[2]):.5f}", float(m[1]), float(m[2]))]
    r = requests.get(
        PDOK_URL,
        params={"q": query, "rows": rows, "fl": "weergavenaam,centroide_ll,type",
                "fq": "type:(gemeente OR woonplaats OR weg OR postcode OR adres OR hectometerpaal)"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    out = []
    for d in r.json().get("response", {}).get("docs", []):
        mm = re.match(r"POINT\(([-\d.]+) ([-\d.]+)\)", d.get("centroide_ll", ""))
        if mm:
            out.append(Place(d["weergavenaam"], float(mm[2]), float(mm[1])))
    return out


def bike_route(waypoints: list[tuple[float, float]]) -> dict:
    """Route through (lat, lon) waypoints. Returns coords [(lat, lon)], distance_m, duration_s."""
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in waypoints)
    r = requests.get(OSRM_URL.format(coords=coords),
                     params={"overview": "full", "geometries": "geojson"}, timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != "Ok" or not j.get("routes"):
        raise RuntimeError(f"Routing failed: {j.get('code')} {j.get('message', '')}")
    rt = j["routes"][0]
    pts = [(lat, lon) for lon, lat in rt["geometry"]["coordinates"]]
    return {"coords": pts, "distance_m": rt["distance"], "duration_s": rt["duration"]}


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
R_EARTH = 6_371_000.0


def haversine(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def bearing(lat1, lon1, lat2, lon2) -> float:
    """Initial compass bearing (deg, 0 = north, 90 = east) from point 1 to 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def segments(coords: list[tuple[float, float]], seg_len_m: float = 400.0) -> list[dict]:
    """Chop a polyline into ~seg_len_m pieces with distance-weighted mean heading.

    Returns dicts: start_m, length_m, mid_m, lat, lon (segment midpoint), bearing.
    """
    out, acc, start_m, cum = [], [], 0.0, 0.0
    for (a_lat, a_lon), (b_lat, b_lon) in zip(coords[:-1], coords[1:]):
        d = haversine(a_lat, a_lon, b_lat, b_lon)
        if d <= 0:
            continue
        acc.append((d, bearing(a_lat, a_lon, b_lat, b_lon), a_lat, a_lon, b_lat, b_lon))
        cum += d
        if cum - start_m >= seg_len_m:
            out.append(_close(acc, start_m))
            start_m, acc = cum, []
    if acc:
        out.append(_close(acc, start_m))
    return out


def _close(acc, start_m) -> dict:
    length = sum(a[0] for a in acc)
    u = sum(a[0] * math.sin(math.radians(a[1])) for a in acc)
    v = sum(a[0] * math.cos(math.radians(a[1])) for a in acc)
    mid = acc[len(acc) // 2]
    return {"start_m": start_m, "length_m": length, "mid_m": start_m + length / 2,
            "lat": (mid[2] + mid[4]) / 2, "lon": (mid[3] + mid[5]) / 2,
            "bearing": (math.degrees(math.atan2(u, v)) + 360) % 360}


def loop_segments(lat: float, lon: float, distance_m: float, seg_len_m: float = 400.0) -> list[dict]:
    """Synthetic loop with unknown route: headings rotate uniformly through 360 deg,
    so half the ride is into the wind, all at the start location's weather."""
    n = max(8, int(distance_m // seg_len_m))
    L = distance_m / n
    return [{"start_m": i * L, "length_m": L, "mid_m": (i + 0.5) * L, "lat": lat, "lon": lon,
             "bearing": (360.0 * (i + 0.5) / n) % 360} for i in range(n)]


# --------------------------------------------------------------------------
# GPX import
# --------------------------------------------------------------------------
def parse_gpx(data: bytes | str, min_spacing_m: float = 10.0) -> dict:
    """Read a GPX 1.0/1.1 file into the same route dict as bike_route().

    Uses track points (<trkpt>, all segments in order); falls back to route points
    (<rtept>) and then waypoints (<wpt>). Points closer than min_spacing_m to the
    previous kept point are dropped (GPS jitter would distort headings).

    Returns coords [(lat, lon)], distance_m, duration_s (None), name, kind
    ("track"/"route"/"waypoints"), n_points, elev_gain_m, is_loop.
    """
    import xml.etree.ElementTree as ET

    if isinstance(data, str):
        data = data.encode("utf-8")
    if b"<!ENTITY" in data[:4096]:
        raise ValueError("GPX files with XML entity declarations are not supported.")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise ValueError(f"Not a valid GPX/XML file ({e}).") from e

    def tag(el):
        return el.tag.rsplit("}", 1)[-1]

    def child_text(el, name):
        for c in el:
            if tag(c) == name and c.text:
                return c.text.strip()
        return None

    if tag(root) != "gpx":
        raise ValueError("File is XML but not GPX (root element is not <gpx>).")

    groups = {"trkpt": [], "rtept": [], "wpt": []}
    for el in root.iter():
        t = tag(el)
        if t in groups:
            try:
                lat, lon = float(el.get("lat")), float(el.get("lon"))
            except (TypeError, ValueError):
                continue
            ele = child_text(el, "ele")
            groups[t].append((lat, lon, float(ele) if ele not in (None, "") else None))
    kind, raw = next(((k, groups[k]) for k in ("trkpt", "rtept", "wpt") if len(groups[k]) >= 2), (None, []))
    if kind is None:
        raise ValueError("The GPX file contains no track, route or at least two waypoints.")

    name = None
    for el in root.iter():
        if tag(el) in ("metadata", "trk", "rte"):
            name = child_text(el, "name")
            if name:
                break

    kept = [raw[0]]
    for p in raw[1:]:
        if haversine(kept[-1][0], kept[-1][1], p[0], p[1]) >= min_spacing_m:
            kept.append(p)
    if haversine(kept[-1][0], kept[-1][1], raw[-1][0], raw[-1][1]) > 0:
        kept.append(raw[-1])
    if len(kept) < 2:
        raise ValueError("The GPX track is too short.")

    dist = sum(haversine(a[0], a[1], b[0], b[1]) for a, b in zip(kept[:-1], kept[1:]))
    eles = [p[2] for p in raw if p[2] is not None]
    gain = None
    if len(eles) >= 2:
        # hysteresis: a climb only counts once it exceeds 3 m above the last low point,
        # so GPS / barometric noise does not accumulate into fake climbing
        gain, ref, climbing = 0.0, eles[0], False
        for e in eles[1:]:
            if e >= ref + 3.0 or (climbing and e > ref):
                gain += e - ref
                ref, climbing = e, True
            elif e <= ref - 3.0 or (not climbing and e < ref):
                ref, climbing = e, False
    coords = [(p[0], p[1]) for p in kept]
    return {"coords": coords, "distance_m": dist, "duration_s": None,
            "name": name or "GPX route", "kind": {"trkpt": "track", "rtept": "route", "wpt": "waypoints"}[kind],
            "n_points": len(coords), "elev_gain_m": gain,
            "is_loop": haversine(*coords[0], *coords[-1]) < 300 and dist > 1000}


def reverse_route(route: dict) -> dict:
    r = dict(route)
    r["coords"] = list(reversed(route["coords"]))
    return r


def sample_points(coords: list[tuple[float, float]], max_points: int = 5,
                  spacing_m: float = 8000.0) -> list[tuple[float, float, float]]:
    """Weather sample locations along the route: (lat, lon, distance_along_m).
    One point per ~spacing_m, capped at max_points (start and end always included)."""
    cum = [0.0]
    for (a, b), (c, d) in zip(coords[:-1], coords[1:]):
        cum.append(cum[-1] + haversine(a, b, c, d))
    total = cum[-1]
    n = int(min(max_points, max(1, math.ceil(total / spacing_m) + 1)))
    if n == 1 or total == 0:
        return [(coords[0][0], coords[0][1], 0.0)]
    targets = [total * k / (n - 1) for k in range(n)]
    out, j = [], 0
    for t in targets:
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        out.append((coords[j][0], coords[j][1], t))
    return out
