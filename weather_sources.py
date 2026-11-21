"""Real-time weather from several independent, keyless sources.

Forecast sources (hourly, normalised to one common frame per location):
  * Open-Meteo, one request returning several numerical weather models:
        KNMI HARMONIE-AROME (2.5 km, Dutch met office)  knmi_seamless
        DWD ICON-D2 / ICON-EU (Germany)                  icon_seamless
        Meteo-France AROME / ARPEGE                      meteofrance_seamless
        ECMWF IFS 0.25 deg                               ecmwf_ifs025
        NOAA GFS                                         gfs_seamless
  * MET Norway Locationforecast 2.0 (ECMWF-based, post-processed)

Real-time / observation sources:
  * Buienradar rain nowcast  - radar extrapolation, 5-min steps, next ~2 h
  * Buienradar station feed  - current KNMI station measurements (+ weather report)
  * MeteoAlarm               - official KNMI weather warnings (code yellow/orange/red)

Normalised hourly frame (index = UTC hour H, tz-aware):
  temp [C], feels [C], precip [mm in hour H..H+1], pop [%], wind [km/h],
  gust [km/h], wind_dir [deg, FROM], humidity [%], cloud [%], uv [-],
  thunder / freezing / snow [0/1 flags]
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

TZ_NL = ZoneInfo("Europe/Amsterdam")
TIMEOUT = 20
USER_AGENT = "cycloth-cycling-clothing-advisor/1.0 (personal, non-commercial)"
COLUMNS = ["temp", "feels", "precip", "pop", "wind", "gust", "wind_dir",
           "humidity", "cloud", "uv", "thunder", "freezing", "snow"]

OPEN_METEO_MODELS = {
    "knmi_seamless": "KNMI HARMONIE",
    "icon_seamless": "DWD ICON",
    "meteofrance_seamless": "Meteo-France AROME",
    "ecmwf_ifs025": "ECMWF IFS",
    "gfs_seamless": "NOAA GFS",
}
MET_NO = "MET Norway"

# Default ensemble weights: high-resolution regional models (HARMONIE, ICON-D2,
# AROME) resolve Dutch coastal wind and showers better than global models.
DEFAULT_WEIGHTS = {
    "KNMI HARMONIE": 1.5,
    "DWD ICON": 1.2,
    "Meteo-France AROME": 1.0,
    "ECMWF IFS": 1.0,
    "MET Norway": 1.0,
    "NOAA GFS": 0.6,
}


@dataclass
class SourceResult:
    name: str
    frames: list[pd.DataFrame] | None = None   # one per sample point
    error: str | None = None
    latency_s: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.frames is not None and self.error is None


def _wmo_flags(code: pd.Series) -> pd.DataFrame:
    c = code.fillna(-1).astype(int)
    return pd.DataFrame({
        "thunder": c.isin([95, 96, 99]).astype(float),
        "freezing": c.isin([56, 57, 66, 67]).astype(float),
        "snow": c.isin([71, 73, 75, 77, 85, 86]).astype(float),
    }, index=code.index)


# --------------------------------------------------------------------------
# Open-Meteo (several models in a single request)
# --------------------------------------------------------------------------
def fetch_open_meteo(points: list[tuple[float, float]], days: int = 4) -> list[SourceResult]:
    t0 = time.time()
    hourly = ["temperature_2m", "apparent_temperature", "precipitation", "precipitation_probability",
              "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m", "weather_code",
              "relative_humidity_2m", "cloud_cover", "uv_index"]
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": ",".join(f"{p[0]:.4f}" for p in points),
            "longitude": ",".join(f"{p[1]:.4f}" for p in points),
            "hourly": ",".join(hourly), "models": ",".join(OPEN_METEO_MODELS),
            "forecast_days": days, "past_days": 1, "timezone": "GMT", "wind_speed_unit": "kmh",
        }, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        js = r.json()
    except Exception as e:  # noqa: BLE001
        return [SourceResult(n, error=f"{type(e).__name__}: {e}") for n in OPEN_METEO_MODELS.values()]
    js = js if isinstance(js, list) else [js]
    lat = time.time() - t0
    results = []
    for model, name in OPEN_METEO_MODELS.items():
        frames = []
        for loc in js:
            h = loc["hourly"]
            idx = pd.to_datetime(h["time"]).tz_localize("UTC")
            g = lambda v: pd.Series(h.get(f"{v}_{model}", [None] * len(idx)), index=idx, dtype="float64")  # noqa: E731
            df = pd.DataFrame({
                "temp": g("temperature_2m"), "feels": g("apparent_temperature"),
                # Open-Meteo precipitation/probability/code refer to the PRECEDING hour -> shift
                "precip": g("precipitation").shift(-1), "pop": g("precipitation_probability").shift(-1),
                "wind": g("wind_speed_10m"), "gust": g("wind_gusts_10m"),
                "wind_dir": g("wind_direction_10m"), "humidity": g("relative_humidity_2m"),
                "cloud": g("cloud_cover"), "uv": g("uv_index"),
            })
            df = df.join(_wmo_flags(g("weather_code").shift(-1)))
            df = df.dropna(subset=["temp"])
            frames.append(df[COLUMNS])
        if all(len(f) == 0 for f in frames):
            results.append(SourceResult(name, error="model has no data for this location", latency_s=lat))
        else:
            results.append(SourceResult(name, frames=frames, latency_s=lat,
                                        meta={"horizon": max(f.index.max() for f in frames if len(f))}))
    return results


# --------------------------------------------------------------------------
# MET Norway Locationforecast 2.0 "complete"
# --------------------------------------------------------------------------
def _met_no_point(lat: float, lon: float) -> pd.DataFrame:
    r = requests.get("https://api.met.no/weatherapi/locationforecast/2.0/complete",
                     params={"lat": round(lat, 4), "lon": round(lon, 4)},
                     headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    rows = []
    for ts in r.json()["properties"]["timeseries"]:
        inst = ts["data"]["instant"]["details"]
        nxt = ts["data"].get("next_1_hours") or ts["data"].get("next_6_hours") or {}
        hours = 1 if "next_1_hours" in ts["data"] else 6
        det = nxt.get("details", {})
        sym = nxt.get("summary", {}).get("symbol_code", "")
        pr = det.get("precipitation_amount")
        rows.append({
            "time": pd.Timestamp(ts["time"]),
            "temp": inst.get("air_temperature"), "feels": inst.get("apparent_air_temperature"),
            "precip": None if pr is None else pr / hours,
            "pop": det.get("probability_of_precipitation"),
            "wind": None if inst.get("wind_speed") is None else inst["wind_speed"] * 3.6,
            "gust": None if inst.get("wind_speed_of_gust") is None else inst["wind_speed_of_gust"] * 3.6,
            "wind_dir": inst.get("wind_from_direction"), "humidity": inst.get("relative_humidity"),
            "cloud": inst.get("cloud_area_fraction"), "uv": inst.get("ultraviolet_index_clear_sky"),
            "thunder": float("thunder" in sym), "freezing": 0.0,
            "snow": float(("snow" in sym) or ("sleet" in sym)),
        })
    df = pd.DataFrame(rows).set_index("time").astype("float64")
    full = pd.date_range(df.index.min(), df.index.max(), freq="1h")
    df = df.reindex(full)
    cont = ["temp", "feels", "wind", "gust", "humidity", "cloud", "uv"]
    df[cont] = df[cont].interpolate(limit=5)
    # wind direction: interpolate on unit vector
    u = np.sin(np.radians(df["wind_dir"])).interpolate(limit=5)
    v = np.cos(np.radians(df["wind_dir"])).interpolate(limit=5)
    df["wind_dir"] = (np.degrees(np.arctan2(u, v)) + 360) % 360
    df[["precip", "pop", "thunder", "freezing", "snow"]] = df[["precip", "pop", "thunder", "freezing", "snow"]].ffill(limit=5)
    return df[COLUMNS]


def fetch_met_no(points: list[tuple[float, float]]) -> SourceResult:
    t0 = time.time()
    try:
        frames = [_met_no_point(lat, lon) for lat, lon in points]
        return SourceResult(MET_NO, frames=frames, latency_s=time.time() - t0,
                            meta={"horizon": frames[0].index.max()})
    except Exception as e:  # noqa: BLE001
        return SourceResult(MET_NO, error=f"{type(e).__name__}: {e}", latency_s=time.time() - t0)


# --------------------------------------------------------------------------
# Buienradar rain nowcast (radar), 5-minute resolution, ~2 h ahead
# --------------------------------------------------------------------------
def _buienradar_nowcast_point(lat: float, lon: float) -> pd.Series:
    r = requests.get("https://gpsgadget.buienradar.nl/data/raintext",
                     params={"lat": f"{lat:.2f}", "lon": f"{lon:.2f}"},
                     headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    now = datetime.now(TZ_NL)
    vals, times = [], []
    for line in r.text.strip().splitlines():
        if "|" not in line:
            continue
        v, hhmm = line.split("|", 1)
        hh, mm = (int(x) for x in hhmm.strip().split(":"))
        t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if t < now - timedelta(hours=3):      # crossed midnight
            t += timedelta(days=1)
        v = int(v)
        vals.append(0.0 if v == 0 else 10 ** ((v - 109) / 32))   # mm/h
        times.append(t.astimezone(timezone.utc))
    return pd.Series(vals, index=pd.DatetimeIndex(times), name="rate_mmh").sort_index()


def fetch_buienradar_nowcast(points: list[tuple[float, float]]) -> SourceResult:
    t0 = time.time()
    try:
        series = [_buienradar_nowcast_point(lat, lon) for lat, lon in points]
        if not any(len(s) for s in series):
            raise RuntimeError("empty nowcast")
        frames = [s.to_frame() for s in series]
        return SourceResult("Buienradar radar nowcast", frames=frames, latency_s=time.time() - t0,
                            meta={"horizon": max(s.index.max() for s in series if len(s))})
    except Exception as e:  # noqa: BLE001
        return SourceResult("Buienradar radar nowcast", error=f"{type(e).__name__}: {e}",
                            latency_s=time.time() - t0)


# --------------------------------------------------------------------------
# Buienradar station observations (KNMI network) + Dutch weather report
# --------------------------------------------------------------------------
def fetch_buienradar_stations() -> SourceResult:
    t0 = time.time()
    try:
        r = requests.get("https://data.buienradar.nl/2.0/feed/json",
                         headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        st = pd.DataFrame(j["actual"]["stationmeasurements"])
        rep = j.get("forecast", {}).get("weatherreport", {})
        return SourceResult("Buienradar stations", frames=[st], latency_s=time.time() - t0,
                            meta={"report_title": rep.get("title", ""), "report_summary": rep.get("summary", ""),
                                  "published": rep.get("published", "")})
    except Exception as e:  # noqa: BLE001
        return SourceResult("Buienradar stations", error=f"{type(e).__name__}: {e}", latency_s=time.time() - t0)


def nearest_station(stations: pd.DataFrame, lat: float, lon: float, max_km: float = 35.0) -> dict | None:
    """Closest station with a valid temperature reading, in normalised units."""
    from routing import haversine
    best, best_d = None, 1e9
    for _, s in stations.iterrows():
        if pd.isna(s.get("temperature")) or pd.isna(s.get("lat")):
            continue
        d = haversine(lat, lon, s["lat"], s["lon"]) / 1000
        if d < best_d:
            best, best_d = s, d
    if best is None or best_d > max_km:
        return None

    def f(k, scale=1.0):
        v = best.get(k)
        return None if v is None or pd.isna(v) else float(v) * scale

    ts = pd.Timestamp(best["timestamp"]).tz_localize(TZ_NL).tz_convert("UTC")
    return {"name": best["stationname"], "distance_km": best_d, "time": ts,
            "temp": f("temperature"), "feels": f("feeltemperature"), "ground_temp": f("groundtemperature"),
            "wind": f("windspeed", 3.6), "gust": f("windgusts", 3.6), "wind_dir": f("winddirectiondegrees"),
            "humidity": f("humidity"), "precip_rate": f("precipitation"), "visibility_m": f("visibility"),
            "rain_24h": f("rainFallLast24Hour"), "description": best.get("weatherdescription", "")}


# --------------------------------------------------------------------------
# MeteoAlarm - official KNMI warnings
# --------------------------------------------------------------------------
LEVEL_COLOURS = {1: "green", 2: "yellow", 3: "orange", 4: "red"}


def fetch_meteoalarm() -> SourceResult:
    t0 = time.time()
    try:
        r = requests.get("https://feeds.meteoalarm.org/api/v1/warnings/feeds-netherlands",
                         headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        r.raise_for_status()
        warnings = []
        for w in r.json().get("warnings", []):
            al = w.get("alert", {})
            if al.get("msgType") == "Cancel":
                continue
            infos = al.get("info", [])
            info = next((i for i in infos if i.get("language", "").startswith("en")), infos[0] if infos else None)
            if not info:
                continue
            params = {p["valueName"]: p["value"] for p in info.get("parameter", [])}
            try:
                level = int(params.get("awareness_level", "1;").split(";")[0])
            except ValueError:
                level = 1
            polys = []
            for a in info.get("area", []):
                for poly in a.get("polygon", []):
                    pts = []
                    for pair in poly.split():
                        la, lo = pair.split(",")
                        pts.append((float(la), float(lo)))
                    polys.append(pts)
            warnings.append({
                "event": info.get("event", ""), "level": level, "colour": LEVEL_COLOURS.get(level, "?"),
                "headline": info.get("headline", ""), "description": info.get("description", ""),
                "instruction": info.get("instruction", ""),
                "onset": pd.Timestamp(info.get("onset")) if info.get("onset") else None,
                "expires": pd.Timestamp(info.get("expires")) if info.get("expires") else None,
                "areas": ", ".join(a.get("areaDesc", "") for a in info.get("area", [])),
                "polygons": polys, "type": params.get("awareness_type", ""),
            })
        return SourceResult("MeteoAlarm (KNMI warnings)", frames=[pd.DataFrame(warnings)],
                            latency_s=time.time() - t0, meta={"warnings": warnings})
    except Exception as e:  # noqa: BLE001
        return SourceResult("MeteoAlarm (KNMI warnings)", error=f"{type(e).__name__}: {e}",
                            latency_s=time.time() - t0)


def _point_in_poly(lat: float, lon: float, poly: list[tuple[float, float]]) -> bool:
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        yi, xi = poly[i]
        yj, xj = poly[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def warnings_for_ride(warnings: list[dict], points: list[tuple[float, float]],
                      start: pd.Timestamp, end: pd.Timestamp, min_level: int = 2) -> list[dict]:
    out = []
    for w in warnings:
        if w["level"] < min_level:
            continue
        if w["expires"] is not None and w["expires"] < start:
            continue
        if w["onset"] is not None and w["onset"] > end:
            continue
        if w["polygons"]:
            if not any(_point_in_poly(la, lo, p) for la, lo in points for p in w["polygons"]):
                continue
        out.append(w)
    return out


# --------------------------------------------------------------------------
# Convenience: fetch everything for a set of sample points
# --------------------------------------------------------------------------
def fetch_all(points: list[tuple[float, float]]) -> dict:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as ex:
        f_om = ex.submit(fetch_open_meteo, points)
        f_met = ex.submit(fetch_met_no, points)
        f_now = ex.submit(fetch_buienradar_nowcast, points)
        f_st = ex.submit(fetch_buienradar_stations)
        f_ma = ex.submit(fetch_meteoalarm)
        forecasts = f_om.result() + [f_met.result()]
        return {"forecasts": forecasts, "nowcast": f_now.result(), "stations": f_st.result(),
                "alarms": f_ma.result(), "fetched_at": pd.Timestamp.now(tz="UTC"), "points": points}
