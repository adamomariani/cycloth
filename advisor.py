"""Ensemble weather fusion, ride simulation, clothing selection and go/no-go verdict."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import routing
from wardrobe import SLOTS, Item
from weather_sources import DEFAULT_WEIGHTS, nearest_station, warnings_for_ride

# --------------------------------------------------------------------------
# Model constants (documented in the app's "Method" tab)
# --------------------------------------------------------------------------
V_REF_KMH = 20.0            # airspeed at which garment comfort ranges are assumed rated
RIDER_HEIGHT_FACTOR = 0.7   # 10 m wind -> ~1.5 m wind over open Dutch terrain (log profile)
INTENSITY = {"Relaxed (commute, upright)": -2.0, "Moderate (touring)": 0.0, "Hard (training)": 3.0}
DRESS_PERCENTILE = 30       # dress for the 30th percentile of felt temperature along the ride
NOWCAST_FULL_MIN, NOWCAST_FADE_MIN = 60, 120
STATION_BIAS_DECAY_H = 6.0
JACKET_BOOST = 6.0          # warmth a thermal jacket adds to the layer underneath


# --------------------------------------------------------------------------
# Small physics helpers
# --------------------------------------------------------------------------
def wind_chill(t_c, v_kmh):
    """JAG/TI wind-chill (Environment Canada / KNMI). Vectorised; T clipped to <=20 C."""
    t = np.minimum(np.asarray(t_c, dtype=float), 20.0)
    v = np.maximum(np.asarray(v_kmh, dtype=float), 4.8)
    vp = v ** 0.16
    return 13.12 + 0.6215 * t - 11.37 * vp + 0.3965 * t * vp


def solar_elevation(lat, lon, ts_seconds):
    """Approximate solar elevation (deg); accuracy ~0.5 deg - plenty for 'is it dark'."""
    d = np.asarray(ts_seconds, dtype=float) / 86400.0 + 2440587.5 - 2451545.0
    g = np.radians(357.529 + 0.98560028 * d)
    q = 280.459 + 0.98564736 * d
    L = np.radians(q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    e = np.radians(23.439 - 0.00000036 * d)
    ra = np.arctan2(np.cos(e) * np.sin(L), np.cos(L))
    dec = np.arcsin(np.sin(e) * np.sin(L))
    gmst = (18.697374558 + 24.06570982441908 * d) % 24
    ha = np.radians(gmst * 15 + lon) - ra
    la = np.radians(lat)
    return np.degrees(np.arcsin(np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)))


def beaufort(kmh: float) -> int:
    limits = [1, 6, 12, 20, 29, 39, 50, 62, 75, 89, 103, 118]
    return int(sum(kmh >= x for x in limits))


def compass(deg: float) -> str:
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((deg % 360) / 22.5 + 0.5) % 16]


def felt_temperature(temp, headwind, crosswind, speed_kmh, intensity_off, personal_off,
                     cloud, sun_el, precip_rate):
    """Felt riding temperature relative to the reference airspeed V_REF_KMH."""
    h = RIDER_HEIGHT_FACTOR * np.asarray(headwind, float)
    c = RIDER_HEIGHT_FACTOR * np.asarray(crosswind, float)
    v_rel = np.sqrt((speed_kmh + h) ** 2 + c ** 2)
    chill = wind_chill(temp, v_rel) - wind_chill(temp, V_REF_KMH)
    cloud = np.nan_to_num(np.asarray(cloud, float), nan=70.0)
    sun = np.where(sun_el > 10, 2.0, np.where(sun_el > 0, 1.0, 0.0)) * (1 - cloud / 100)
    p = np.nan_to_num(np.asarray(precip_rate, float))
    wet = np.where(p >= 0.3, -2.0, np.where(p >= 0.1, -1.0, 0.0))
    return np.asarray(temp, float) + chill + intensity_off + personal_off + sun + wet


# --------------------------------------------------------------------------
# Ensemble
# --------------------------------------------------------------------------
def _secs(idx) -> np.ndarray:
    return pd.DatetimeIndex(idx).tz_convert("UTC").as_unit("s").asi8.astype(float)


def _interp(tt, arr, ts):
    m = np.isfinite(arr)
    if m.sum() < 2:
        return np.full(len(ts), np.nan)
    x, y = tt[m], arr[m]
    out = np.interp(ts, x, y)
    out[(ts < x[0]) | (ts > x[-1])] = np.nan
    return out


def _block(tt, arr, ts):
    i = np.searchsorted(tt, ts, side="right") - 1
    ok = (i >= 0) & (ts < tt[-1] + 3600)
    return np.where(ok, arr[np.clip(i, 0, len(arr) - 1)], np.nan)


def _wmean(mat, w):
    """Weighted nan-mean over axis 0 (sources)."""
    fin = np.isfinite(mat)
    ww = w[:, None] * fin
    s = ww.sum(0)
    return np.where(s > 0, np.nansum(np.where(fin, mat, 0) * ww, 0) / np.where(s > 0, s, 1), np.nan)


class Ensemble:
    CONT = ["temp", "feels", "wind", "gust", "humidity", "cloud", "uv"]
    BLOCK = ["precip", "pop", "thunder", "freezing", "snow"]

    def __init__(self, data: dict, weights: dict | None = None):
        weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.points = data["points"]
        self.fetched = data["fetched_at"].value / 1e9
        self.sources = [s for s in data["forecasts"] if s.ok and weights.get(s.name, 1.0) > 0]
        self.names = [s.name for s in self.sources]
        self.w = np.array([weights.get(n, 1.0) for n in self.names], float)
        self.arr = []   # [source][point] -> dict
        for s in self.sources:
            per_pt = []
            for df in s.frames:
                d = {"tt": _secs(df.index)}
                for c in self.CONT + self.BLOCK:
                    d[c] = df[c].to_numpy(float)
                wd = np.radians(df["wind_dir"].to_numpy(float))
                d["u"], d["v"] = np.sin(wd), np.cos(wd)
                per_pt.append(d)
            self.arr.append(per_pt)
        # radar nowcast
        self.nowcast = []
        nc = data.get("nowcast")
        if nc is not None and nc.ok:
            for s in nc.frames:
                self.nowcast.append((_secs(s.index), s["rate_mmh"].to_numpy(float)) if len(s) else None)
        # station bias (temp, wind) per sample point
        self.bias = [None] * len(self.points)
        self.stations = [None] * len(self.points)
        st = data.get("stations")
        if st is not None and st.ok:
            for i, (la, lo) in enumerate(self.points):
                obs = nearest_station(st.frames[0], la, lo)
                self.stations[i] = obs
                if obs is None or abs(obs["time"].value / 1e9 - self.fetched) > 90 * 60:
                    continue
                raw = self._raw(i, np.array([obs["time"].value / 1e9]))
                bt = (obs["temp"] - raw["temp"][0]) if obs["temp"] is not None else 0.0
                bw = (obs["wind"] - raw["wind"][0]) if obs["wind"] is not None else 0.0
                bt = 0.0 if not np.isfinite(bt) else float(np.clip(bt, -4, 4))
                bw = 0.0 if not np.isfinite(bw) else float(np.clip(bw, -12, 12))
                self.bias[i] = {"t0": obs["time"].value / 1e9, "temp": bt, "wind": bw, "station": obs["name"]}

    # raw per-source values, shape (S, N)
    def _per_source(self, pi, ts):
        out = {c: [] for c in self.CONT + self.BLOCK + ["u", "v"]}
        for s in range(len(self.sources)):
            d = self.arr[s][pi]
            for c in self.CONT + ["u", "v"]:
                out[c].append(_interp(d["tt"], d[c], ts))
            for c in self.BLOCK:
                out[c].append(_block(d["tt"], d[c], ts))
        return {c: np.vstack(v) if v else np.empty((0, len(ts))) for c, v in out.items()}

    def _raw(self, pi, ts):
        ps = self._per_source(pi, ts)
        w = self.w
        res = {c: _wmean(ps[c], w) for c in ["temp", "feels", "wind", "gust", "humidity", "cloud", "uv", "precip"]}
        pop_src = np.where(np.isfinite(ps["pop"]), ps["pop"],
                           np.where(np.isfinite(ps["precip"]), 100.0 * (ps["precip"] >= 0.1), np.nan))
        res["pop"] = _wmean(pop_src, w)
        for c in ["thunder", "freezing", "snow"]:
            res[c] = _wmean(ps[c], w)
        spd = np.where(np.isfinite(ps["wind"]), ps["wind"], 0)
        u, v = _wmean(ps["u"] * spd, w), _wmean(ps["v"] * spd, w)
        res["wind_dir"] = (np.degrees(np.arctan2(u, v)) + 360) % 360
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            res["temp_spread"] = np.nanmax(ps["temp"], 0) - np.nanmin(ps["temp"], 0) if len(w) else np.nan
            res["wind_spread"] = np.nanmax(ps["wind"], 0) - np.nanmin(ps["wind"], 0) if len(w) else np.nan
            res["precip_max_src"] = np.nanmax(ps["precip"], 0) if len(w) else np.nan
        res["n_sources"] = np.isfinite(ps["temp"]).sum(0)
        res["_per_source"] = ps
        return res

    def at(self, pi: int, ts: np.ndarray) -> dict:
        ts = np.asarray(ts, float)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            res = self._raw(pi, ts)
        # 1) station bias correction, decaying over STATION_BIAS_DECAY_H
        b = self.bias[pi]
        res["temp_corr"] = np.zeros(len(ts))
        if b:
            decay = np.clip(1 - (ts - b["t0"]) / (STATION_BIAS_DECAY_H * 3600), 0, 1)
            decay[ts < b["t0"] - 3600] = 0
            res["temp"] = res["temp"] + b["temp"] * decay
            res["wind"] = np.maximum(0, res["wind"] + b["wind"] * decay)
            res["temp_corr"] = b["temp"] * decay
        # 2) radar nowcast for precipitation in the first ~2 h
        res["nowcast_w"] = np.zeros(len(ts))
        nc = self.nowcast[pi] if pi < len(self.nowcast) else None
        if nc is not None:
            tt, rate = nc
            r = np.interp(ts, tt, rate)
            lead = (ts - self.fetched) / 60.0
            wnc = np.where(lead <= NOWCAST_FULL_MIN, 1.0,
                           np.clip(1 - 0.5 * (lead - NOWCAST_FULL_MIN) / (NOWCAST_FADE_MIN - NOWCAST_FULL_MIN), 0.5, 1))
            wnc = np.where((ts >= tt[0] - 300) & (ts <= tt[-1]), wnc, 0.0)
            model_p = np.nan_to_num(res["precip"])
            res["precip"] = wnc * r + (1 - wnc) * model_p
            pop_nc = np.where(r >= 0.1, 95.0, 5.0)
            res["pop"] = wnc * pop_nc + (1 - wnc) * np.nan_to_num(res["pop"])
            res["nowcast_w"] = wnc
        return res


# --------------------------------------------------------------------------
# Ride description and simulation
# --------------------------------------------------------------------------
@dataclass
class Leg:
    name: str
    segs: list[dict]
    offset_s: float = 0.0               # departure offset from plan start
    sample_m: list[float] = field(default_factory=lambda: [0.0])  # distance of each sample point
    sample_idx: list[int] = field(default_factory=lambda: [0])    # index into Ensemble.points

    @property
    def distance_m(self) -> float:
        return sum(s["length_m"] for s in self.segs)


def simulate_leg(ens: Ensemble, leg: Leg, departure: pd.Timestamp, prefs: dict) -> pd.DataFrame:
    speed = float(prefs["speed_kmh"])
    v = speed / 3.6
    dep = departure.value / 1e9 + leg.offset_s
    seg = pd.DataFrame(leg.segs)
    seg["t"] = dep + seg["mid_m"] / v
    sm = np.asarray(leg.sample_m)
    seg["pi"] = [leg.sample_idx[int(np.argmin(np.abs(sm - m)))] for m in seg["mid_m"]]
    cols = ["temp", "precip", "pop", "wind", "gust", "wind_dir", "cloud", "uv", "humidity",
            "thunder", "freezing", "snow", "temp_spread", "wind_spread", "n_sources", "nowcast_w", "temp_corr"]
    for c in cols:
        seg[c] = np.nan
    for pi, grp in seg.groupby("pi"):
        r = ens.at(int(pi), grp["t"].to_numpy())
        for c in cols:
            seg.loc[grp.index, c] = r[c]
    rel = np.radians(seg["wind_dir"] - seg["bearing"])
    seg["headwind"] = seg["wind"] * np.cos(rel)       # + = into the wind
    seg["crosswind"] = seg["wind"] * np.sin(rel)
    seg["sun_el"] = solar_elevation(seg["lat"].to_numpy(), seg["lon"].to_numpy(), seg["t"].to_numpy())
    seg["felt"] = felt_temperature(seg["temp"], seg["headwind"], seg["crosswind"], speed,
                                   INTENSITY.get(prefs.get("intensity"), 0.0),
                                   float(prefs.get("personal_offset", 0.0)),
                                   seg["cloud"], seg["sun_el"], seg["precip"])
    seg["time"] = pd.to_datetime(seg["t"], unit="s", utc=True)
    seg["hours"] = seg["length_m"] / 1000 / speed
    seg["leg"] = leg.name
    return seg


def _wpct(x, w, q):
    o = np.argsort(x)
    x, w = np.asarray(x)[o], np.asarray(w)[o]
    cw = np.cumsum(w) / w.sum()
    return float(np.interp(q / 100, cw, x))


def summarize(seg: pd.DataFrame) -> dict:
    w = seg["length_m"].to_numpy()
    ok = np.isfinite(seg["temp"].to_numpy())
    if not ok.any():
        return {"valid": False}
    s = seg[ok]
    w = w[ok]
    wm = lambda c: float(np.average(s[c].fillna(0), weights=w))  # noqa: E731
    return {
        "valid": True,
        "start": s["time"].min() - pd.Timedelta(seconds=float(s["hours"].iloc[0] * 1800)),
        "end": s["time"].max() + pd.Timedelta(seconds=float(s["hours"].iloc[-1] * 1800)),
        "distance_km": float(seg["length_m"].sum() / 1000),
        "duration_min": float(seg["hours"].sum() * 60),
        "temp_min": float(s["temp"].min()), "temp_max": float(s["temp"].max()), "temp_mean": wm("temp"),
        "felt_min": float(s["felt"].min()), "felt_max": float(s["felt"].max()), "felt_mean": wm("felt"),
        "felt_dress": _wpct(s["felt"], w, DRESS_PERCENTILE),
        "rain_mm": float((s["precip"].fillna(0) * s["hours"]).sum()),
        "rain_max_rate": float(s["precip"].max()), "pop_max": float(s["pop"].max()),
        "wet_fraction": float(np.average(s["precip"].fillna(0) >= 0.1, weights=w)),
        "wind_mean": wm("wind"), "wind_max": float(s["wind"].max()),
        "gust_max": float(s["gust"].max()) if s["gust"].notna().any() else float("nan"),
        "wind_dir_mean": float((np.degrees(np.arctan2(np.average(np.sin(np.radians(s["wind_dir"])), weights=w),
                                                      np.average(np.cos(np.radians(s["wind_dir"])), weights=w))) + 360) % 360),
        "headwind_mean": wm("headwind"), "headwind_max": float(s["headwind"].max()),
        "headwind_fraction": float(np.average(s["headwind"] > 5, weights=w)),
        "dark_fraction": float(np.average(s["sun_el"] < -0.833, weights=w)),
        "sunny": bool(((s["sun_el"] > 10) & (s["cloud"].fillna(100) < 50)).any()),
        "uv_max": float(s["uv"].max()) if s["uv"].notna().any() else 0.0,
        "cloud_mean": wm("cloud"), "humidity_mean": wm("humidity"),
        "thunder": float(s["thunder"].max()), "freezing": float(s["freezing"].max()), "snow": float(s["snow"].max()),
        "temp_spread_max": float(s["temp_spread"].max()), "wind_spread_max": float(s["wind_spread"].max()),
        "n_sources_min": int(s["n_sources"].min()),
        "nowcast_used": bool((s["nowcast_w"] > 0).any()),
        "complete": bool(ok.all()),
    }


# --------------------------------------------------------------------------
# Clothing selection
# --------------------------------------------------------------------------
@dataclass
class Pick:
    slot: str
    item: Item
    action: str       # "wear" | "pack"
    reason: str


def _rng(i: Item) -> str:
    return f"{i.t_min:g} to {i.t_max:g} C"


def _best(items: list[Item], t: float, tol: float = 1.0, pred=None) -> tuple[Item | None, bool]:
    cand = [i for i in items if pred is None or pred(i)]
    if not cand:
        return None, False
    cov = [i for i in cand if i.covers(t, tol)]
    if cov:
        return min(cov, key=lambda i: abs((i.t_min + i.t_max) / 2 - t) + 0.05 * (i.t_max - i.t_min)), True
    return min(cand, key=lambda i: i.distance(t)), False


def recommend_outfit(st: dict, wardrobe: list[Item]) -> dict:
    by = {k: [i for i in wardrobe if i.slot == k] for k in SLOTS}
    T, lo, hi = st["felt_dress"], st["felt_min"], st["felt_max"]
    picks: list[Pick] = []
    warnings, tips = [], []

    def wear(slot, item, reason):
        picks.append(Pick(slot, item, "wear", reason))

    def pack(slot, item, reason):
        picks.append(Pick(slot, item, "pack", reason))

    if not wardrobe:
        return {"picks": [], "warnings": ["Your wardrobe is empty - add items on the 'My wardrobe' tab."],
                "tips": [], "dress_temp": T}

    rain_expected = st["rain_mm"] >= 0.5 or st["rain_max_rate"] >= 0.5 or st["pop_max"] >= 65
    rain_possible = not rain_expected and (st["pop_max"] >= 30 or st["rain_mm"] >= 0.1)
    heavy_rain = st["rain_mm"] >= 1.5 or st["rain_max_rate"] >= 2.0
    windy = st["wind_mean"] >= 25 or st["headwind_mean"] >= 12 or (st["gust_max"] or 0) >= 45
    dark = st["dark_fraction"] > 0
    swing = hi - lo

    # ---- temperature-driven slots --------------------------------------
    jacket, jacket_ok = _best(by["insulation"], T)
    jacket_on = jacket is not None and jacket_ok
    for slot in ["base", "insulation", "jersey", "legs", "hands", "feet", "overshoes", "head", "neck"]:
        meta = SLOTS[slot]
        if slot == "jersey" and jacket_on:
            # mid layer under a jacket: the jacket adds roughly JACKET_BOOST deg C
            item, _ = _best(by[slot], T + JACKET_BOOST, tol=3)
            if item:
                wear(slot, item, f"mid layer under '{jacket.name}'")
            continue
        item, ok = _best(by[slot], T)
        if item and ok:
            wear(slot, item, f"rated {_rng(item)}; you will feel ~{T:.0f} C")
        elif item and meta["required"]:
            wear(slot, item, f"closest match (rated {_rng(item)}) for a felt ~{T:.0f} C")
            warnings.append(f"No {meta['label'].lower()} in your wardrobe is rated for ~{T:.0f} C - "
                            f"'{item.name}' ({_rng(item)}) is the closest.")
        elif meta["required"]:
            warnings.append(f"Add at least one item to '{meta['label']}'.")
        elif meta["warn_below"] is not None and T < meta["warn_below"]:
            if item:
                warnings.append(f"{meta['label']}: nothing rated for ~{T:.0f} C; '{item.name}' only goes down "
                                f"to {item.t_min:g} C.")
                if item.t_max >= T:   # still better than nothing
                    wear(slot, item, f"warmest available (rated {_rng(item)})")
            else:
                warnings.append(f"{meta['label']}: none in your wardrobe - recommended below "
                                f"{meta['warn_below']} C.")

    worn = {p.slot: p.item for p in picks if p.action == "wear"}

    # ---- supplements: arm / knee warmers for short garments ------------
    arms_covered = any(worn.get(s) and worn[s].long_cover for s in ["base", "jersey", "insulation"])
    if not arms_covered:
        item, ok = _best(by["arms"], T)
        if item and ok:
            wear("arms", item, f"short-sleeve top at ~{T:.0f} C; roll down if it warms up")
        elif item and swing >= 4 and item.covers(lo):
            pack("arms", item, f"start of the ride may feel ~{lo:.0f} C")
    legs_long = worn.get("legs") is not None and worn["legs"].long_cover
    if not legs_long:
        item, ok = _best(by["knees"], T)
        if item and ok:
            wear("knees", item, f"shorts at ~{T:.0f} C")

    # ---- rain ------------------------------------------------------------
    worn = {p.slot: p.item for p in picks if p.action == "wear"}
    if rain_expected or rain_possible:
        item, _ = _best(by["rain_top"], T, tol=4)
        if item is None:
            item, _ = _best([i for i in wardrobe if i.waterproof and i.slot in ("insulation", "wind")], T, tol=4)
        txt = f"{st['rain_mm']:.1f} mm expected, chance up to {st['pop_max']:.0f}%"
        if item and rain_expected:
            wear(item.slot, item, txt)
            if "poncho" in item.name.lower() and (st["gust_max"] or 0) >= 40:
                tips.append("Gusty wind: a poncho catches the wind - a fitted rain jacket is safer today.")
        elif item:
            pack(item.slot, item, txt + " - take it along")
        elif rain_expected:
            warnings.append("Rain expected but there is no waterproof top in your wardrobe.")
        if heavy_rain and st["duration_min"] >= 15:
            it, _ = _best(by["rain_legs"], T, tol=5)
            if it:
                (wear if rain_expected else pack)("rain_legs", it, f"heavy spells up to {st['rain_max_rate']:.1f} mm/h")
        if rain_expected and T < 15 and "overshoes" not in worn:
            it, _ = _best(by["overshoes"], T, tol=6, pred=lambda i: i.waterproof)
            if it:
                wear("overshoes", it, "keeps feet dry")

    # ---- wind ------------------------------------------------------------
    worn = {p.slot: p.item for p in picks if p.action == "wear"}
    windproof_outer = any(i.windproof for s, i in worn.items() if s in ("insulation", "rain_top", "wind"))
    if by["wind"] and not windproof_outer:
        item, ok = _best(by["wind"], T)
        if item and ok:
            wear("wind", item, f"rated {_rng(item)}; blocks wind ({beaufort(st['wind_mean'])} Bft)")
        elif item and windy and T < 18:
            wear("wind", item, f"windy: {beaufort(st['wind_mean'])} Bft, headwind ~{max(0, st['headwind_mean']):.0f} km/h")
        elif item and swing >= 4 and item.covers(lo, tol=2):
            pack("wind", item, f"felt temperature varies {lo:.0f}-{hi:.0f} C along the ride")

    # ---- eyes / visibility ----------------------------------------------
    eyes = by["eyes"]
    if eyes:
        sun_l = [i for i in eyes if i.lens in ("sun", "photochromic")]
        clear_l = [i for i in eyes if i.lens in ("clear", "photochromic")]
        if st["sunny"] and sun_l:
            wear("eyes", sun_l[0], "sunny spells")
        elif (dark or rain_expected) and clear_l:
            wear("eyes", clear_l[0], "dark or rainy - keeps spray and grit out")
        elif clear_l or sun_l:
            wear("eyes", (clear_l or sun_l)[0], "eye protection against wind and insects")
    if dark:
        it = next((i for i in by["visibility"] if i.reflective), None)
        if it:
            wear("visibility", it, f"{st['dark_fraction']*100:.0f}% of the ride is after sunset / before sunrise")
        tips.append("Lights are legally required in the dark: white/yellow front light and red rear light (NL).")

    # ---- general tips -----------------------------------------------------
    if swing >= 4 and T < 20:
        tips.append(f"Felt temperature ranges {lo:.0f} to {hi:.0f} C along the ride - choose layers you can unzip or stow.")
    if worn.get("insulation") and hi > worn["insulation"].t_max + 3:
        tips.append(f"'{worn['insulation'].name}' is rated up to {worn['insulation'].t_max:g} C; you may overheat later.")
    if st["uv_max"] >= 3 and st["sunny"]:
        tips.append(f"UV index up to {st['uv_max']:.0f}: sunscreen on face, arms and legs.")
    if st["felt_max"] >= 25:
        tips.append("Warm ride: bring enough water.")
    if st["headwind_mean"] >= 8:
        tips.append(f"Average headwind ~{st['headwind_mean']:.0f} km/h - plan extra time.")
    elif st["headwind_mean"] <= -8:
        tips.append(f"Tailwind on average (~{-st['headwind_mean']:.0f} km/h).")
    order = list(SLOTS)
    picks.sort(key=lambda p: (p.action != "wear", order.index(p.slot)))
    # de-duplicate (a slot may have been picked twice by different rules)
    seen, uniq = set(), []
    for p in picks:
        if (p.slot, p.item.id) in seen:
            continue
        seen.add((p.slot, p.item.id))
        uniq.append(p)
    return {"picks": uniq, "warnings": warnings, "tips": tips, "dress_temp": T}


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------
LEVELS = {0: "GO", 1: "GO, WITH CAUTION", 2: "RECONSIDER / NO-GO"}


def assess(st: dict, warnings: list[dict], station: dict | None, outfit: dict, lead_h: float) -> dict:
    R = []

    def add(sev, msg):
        R.append((sev, msg))

    if not st.get("valid"):
        return {"level": 2, "label": "NO DATA", "reasons": [(2, "No forecast data covers this ride time.")],
                "confidence": "low", "score": 0}
    for w in warnings:
        sev = 2 if w["level"] >= 3 else 1
        add(sev, f"KNMI code {w['colour']}: {w['event']} ({w['areas'][:80]})")
    g, wm = st["gust_max"], st["wind_mean"]
    if np.isfinite(g):
        if g >= 75:
            add(2, f"Severe gusts up to {g:.0f} km/h - dangerous on a bike.")
        elif g >= 60:
            add(1, f"Strong gusts up to {g:.0f} km/h - watch crosswinds on open roads and bridges.")
    if wm >= 50:
        add(2, f"Storm-force mean wind {wm:.0f} km/h ({beaufort(wm)} Bft).")
    elif wm >= 39:
        add(1, f"Strong wind {wm:.0f} km/h ({beaufort(wm)} Bft).")
    elif st["headwind_mean"] >= 20:
        add(1, f"Hard headwind (~{st['headwind_mean']:.0f} km/h average).")
    if st["thunder"] >= 0.35:
        add(2, "Thunderstorms forecast by several sources - lightning risk in open country.")
    elif st["thunder"] >= 0.15:
        add(1, "Some sources forecast thunderstorms.")
    if st["freezing"] >= 0.2:
        add(2, "Freezing rain / drizzle forecast - black ice.")
    wet = st["rain_mm"] > 0.05 or st["wet_fraction"] > 0 or (station and (station.get("rain_24h") or 0) > 0.5)
    ground = station.get("ground_temp") if station else None
    if st["temp_min"] <= 1.0 and wet:
        add(1, f"Near-freezing ({st['temp_min']:.0f} C) on wet roads - risk of slippery patches (gladheid).")
    if ground is not None and ground <= 0 and wet and lead_h < 6:
        add(2 if st["rain_mm"] > 0.2 else 1, f"Ground temperature {ground:.1f} C with moisture - icy roads likely.")
    if st["snow"] >= 0.3:
        add(1, "Snow or sleet forecast.")
    if st["rain_max_rate"] >= 10 or st["rain_mm"] >= 8:
        add(2, f"Very heavy rain ({st['rain_mm']:.1f} mm, peaks {st['rain_max_rate']:.1f} mm/h).")
    elif st["rain_mm"] >= 2 or st["rain_max_rate"] >= 2.5:
        add(1, f"Wet ride: {st['rain_mm']:.1f} mm, peaks {st['rain_max_rate']:.1f} mm/h.")
    if st["felt_min"] <= -12:
        add(2, f"Extreme cold (feels {st['felt_min']:.0f} C).")
    elif st["felt_min"] <= -5:
        add(1, f"Very cold (feels {st['felt_min']:.0f} C).")
    if st["temp_max"] >= 35:
        add(2, f"Extreme heat ({st['temp_max']:.0f} C).")
    elif st["temp_max"] >= 30:
        add(1, f"Hot ({st['temp_max']:.0f} C).")
    if station and station.get("visibility_m") is not None and station["visibility_m"] < 300 and lead_h < 3:
        add(1, f"Fog at {station['name']}: visibility {station['visibility_m']:.0f} m.")
    if st["dark_fraction"] > 0:
        add(0, "Part of the ride is in the dark - lights on.")
    for wtxt in outfit.get("warnings", []):
        add(1 if "No " in wtxt or "Rain expected" in wtxt else 0, "Wardrobe: " + wtxt)
    if not st["complete"]:
        add(1, "Forecast horizon does not cover the whole ride.")

    spread = st["temp_spread_max"]
    conf = "high" if spread <= 2 and lead_h <= 24 else ("medium" if spread <= 4 and lead_h <= 72 else "low")
    if st["n_sources_min"] < 3:
        conf = "low"
    level = max([r[0] for r in R], default=0)
    return {"level": level, "label": LEVELS[level], "reasons": sorted(R, key=lambda r: -r[0]),
            "confidence": conf, "score": comfort_score(st, level)}


def comfort_score(st: dict, level: int = 0) -> float:
    if not st.get("valid"):
        return 0.0
    g = st["gust_max"] if np.isfinite(st["gust_max"]) else st["wind_max"] * 1.5
    pen = (12 * min(st["rain_mm"], 5) + 0.15 * max(0, st["pop_max"] - 20)
           + 1.0 * max(0, st["headwind_mean"]) + 0.5 * max(0, g - 40)
           + 1.5 * max(0, 5 - st["felt_dress"]) + 1.5 * max(0, st["felt_max"] - 28)
           + 8 * st["dark_fraction"] + 30 * st["thunder"] + {0: 0, 1: 10, 2: 40}[level])
    return float(max(0.0, 100 - pen))


# --------------------------------------------------------------------------
# Plan = list of legs, evaluated together
# --------------------------------------------------------------------------
def evaluate_plan(ens: Ensemble, legs: list[Leg], departure: pd.Timestamp, prefs: dict,
                  wardrobe: list[Item], alarms: list[dict]) -> dict:
    out = []
    for leg in legs:
        seg = simulate_leg(ens, leg, departure, prefs)
        st = summarize(seg)
        if not st.get("valid"):
            out.append({"leg": leg, "seg": seg, "stats": st, "outfit": {"picks": [], "warnings": [], "tips": []},
                        "verdict": assess(st, [], None, {}, 0), "warnings": []})
            continue
        pts = list(zip(seg["lat"], seg["lon"]))[:: max(1, len(seg) // 20)]
        wr = warnings_for_ride(alarms, pts, st["start"], st["end"])
        station = ens.stations[leg.sample_idx[0]] if leg.sample_idx else None
        outfit = recommend_outfit(st, wardrobe)
        lead_h = (st["start"].value / 1e9 - ens.fetched) / 3600
        out.append({"leg": leg, "seg": seg, "stats": st, "outfit": outfit, "warnings": wr,
                    "verdict": assess(st, wr, station, outfit, lead_h), "station": station})
    level = max(r["verdict"]["level"] for r in out)
    confs = [r["verdict"]["confidence"] for r in out]
    conf = "low" if "low" in confs else ("medium" if "medium" in confs else "high")
    return {"legs": out, "level": level, "label": LEVELS.get(level, "?"), "confidence": conf,
            "score": float(np.mean([r["verdict"]["score"] for r in out]))}


def scan_departures(ens: Ensemble, legs: list[Leg], centre: pd.Timestamp, prefs: dict,
                    wardrobe: list[Item], alarms: list[dict], flex_h: float = 3.0,
                    step_min: int = 30) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    start = max(now, centre - pd.Timedelta(hours=flex_h))
    rows = []
    t = start
    while t <= centre + pd.Timedelta(hours=flex_h):
        res = evaluate_plan(ens, legs, t, prefs, wardrobe, alarms)
        st0 = res["legs"][0]["stats"]
        if st0.get("valid"):
            rows.append({"departure": t, "score": res["score"], "level": res["level"],
                         "verdict": res["label"],
                         "rain_mm": sum(r["stats"]["rain_mm"] for r in res["legs"] if r["stats"].get("valid")),
                         "felt_dress": st0["felt_dress"], "headwind": st0["headwind_mean"],
                         "gust_max": st0["gust_max"]})
        t += pd.Timedelta(minutes=step_min)
    return pd.DataFrame(rows)


def build_legs(mode: str, route: dict | None, start: tuple[float, float], loop_km: float | None,
               return_after_h: float, ens_points: list[tuple[float, float]],
               sample_m: list[float]) -> list[Leg]:
    """Build legs; ens_points[i] were sampled at sample_m[i] metres along the outbound route."""
    idx = list(range(len(ens_points)))
    if mode == "loop":
        segs = routing.loop_segments(start[0], start[1], loop_km * 1000)
        return [Leg("Loop", segs, 0.0, [0.0], [0])]
    segs = routing.segments(route["coords"])
    legs = [Leg("Outbound", segs, 0.0, sample_m, idx)]
    if mode == "return":
        total = route["distance_m"]
        back = routing.segments(list(reversed(route["coords"])))
        legs.append(Leg("Return", back, return_after_h * 3600.0,
                        [total - m for m in sample_m], idx))
    return legs
