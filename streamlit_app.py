"""Cycloth - what to wear on the bike (Netherlands).

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydeck as pdk  # noqa: E402
import streamlit as st  # noqa: E402

import advisor as adv  # noqa: E402
import routing  # noqa: E402
import wardrobe as wd  # noqa: E402
import weather_sources as ws  # noqa: E402

TZ = ws.TZ_NL
st.set_page_config(page_title="Cycloth - cycling clothing advisor", page_icon="🚲", layout="wide")


# ==========================================================================
# Cached remote calls
# ==========================================================================
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_geocode(q: str):
    return [asdict(p) for p in routing.geocode(q)]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def cached_route(waypoints: tuple):
    return routing.bike_route(list(waypoints))


@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=20)
def cached_gpx(raw: bytes):
    return routing.parse_gpx(raw)


@st.cache_data(ttl=600, show_spinner=False)
def cached_weather(points: tuple):
    return ws.fetch_all(list(points))


# ==========================================================================
# State
# ==========================================================================
if "wardrobe" not in st.session_state:
    st.session_state.wardrobe = wd.load_wardrobe()
if "settings" not in st.session_state:
    st.session_state.settings = wd.load_settings()
S = st.session_state.settings


def persist_settings():
    wd.save_settings(S)


def local(ts: pd.Timestamp, fmt="%H:%M") -> str:
    return ts.tz_convert(TZ).strftime(fmt)


# ==========================================================================
# Sidebar - rider profile
# ==========================================================================
with st.sidebar:
    st.header("Rider profile")
    speed = st.slider("Average speed (km/h)", 8.0, 40.0, float(S["speed_kmh"]), 0.5)
    intensity = st.selectbox("Effort", list(adv.INTENSITY), index=list(adv.INTENSITY).index(S["intensity"])
                             if S["intensity"] in adv.INTENSITY else 1,
                             help="Harder effort produces more body heat, so you can dress lighter.")
    personal = st.slider("I usually feel ...", -4.0, 4.0, float(S["personal_offset"]), 0.5,
                         help="Negative = you get cold easily (advice dresses you warmer). "
                              "Positive = you run hot.")
    st.caption("cold-sensitive  ◀  0  ▶  runs hot")
    with st.expander("Weather source weights"):
        st.caption("Relative weight of each forecast in the ensemble (0 = ignore).")
        weights = {}
        for name, default in ws.DEFAULT_WEIGHTS.items():
            weights[name] = st.slider(name, 0.0, 3.0, float(S["weights"].get(name, default)), 0.1, key=f"w_{name}")
    favs_ = S.setdefault("places", {})
    with st.expander(f"Favorite places ({len(favs_)})"):
        if not favs_:
            st.caption("No favorites yet. Search an address on **Plan ride** and use *Save as favorite*.")
        for fav_name, fav in list(favs_.items()):
            fc1, fc2 = st.columns([5, 1], vertical_alignment="center")
            fc1.markdown(f"**{fav_name}**  \n{fav['label']}  \n`{fav['lat']:.5f}, {fav['lon']:.5f}`")
            if fc2.button(":material/delete:", key=f"delfav_{fav_name}", help=f"Remove '{fav_name}'"):
                favs_.pop(fav_name, None)
                persist_settings()
                for k in ("start_fav", "dest_fav"):           # drop a now-invalid selection
                    if st.session_state.get(k) == fav_name:
                        del st.session_state[k]
                st.toast(f"Removed favorite '{fav_name}'.")
                st.rerun()
    new = {"speed_kmh": speed, "intensity": intensity, "personal_offset": personal, "weights": weights}
    if any(S.get(k) != v for k, v in new.items()):
        S.update(new)
        persist_settings()
    if st.button("Refresh weather data"):
        cached_weather.clear()
        st.toast("Weather cache cleared - the next advice uses fresh data.")
    st.caption(f"Data stored in `{wd.DATA_DIR}`")

for _k in ("start", "dest"):
    _saved = st.session_state.pop(f"{_k}_fav_saved", None)
    if _saved:
        st.toast(f"Saved favorite '{_saved}'.")

prefs = {"speed_kmh": speed, "intensity": intensity, "personal_offset": personal}

st.title("Cycloth - what to wear on the bike")
tab_plan, tab_wardrobe, tab_method = st.tabs(["Plan ride", "My wardrobe", "Method & sources"])


# ==========================================================================
# Helpers for the plan tab
# ==========================================================================
def place_picker(key: str, label: str):
    favs = S.get("places", {})
    if favs:
        # dropdown only when at least one favorite exists; clear it to search an address instead
        choice = st.selectbox(label, list(favs), index=None, placeholder="Search favorites...", key=f"{key}_fav")
        if choice is not None and choice in favs:
            f = favs[choice]
            st.caption(f["label"])
            return routing.Place(f["label"], f["lat"], f["lon"])
        q_label = "... or search an address, street, town, postcode or 'lat, lon'"
    else:
        q_label = f"{label}: address, street, town, postcode or 'lat, lon'"
    q = st.text_input(q_label, key=f"{key}_q", placeholder="e.g. Stationsplein Wageningen")
    if not q:
        return None
    try:
        res = cached_geocode(q)
    except Exception as e:  # noqa: BLE001
        st.error(f"Geocoder unavailable: {e}")
        return None
    if not res:
        st.warning("No match found.")
        return None
    i = st.selectbox("Match", range(len(res)), format_func=lambda k: res[k]["label"], key=f"{key}_match")
    p = routing.Place(**res[i])
    if not routing.in_netherlands(p.lat, p.lon):
        st.warning("This location is outside the Netherlands; Buienradar and HARMONIE coverage may be missing.")
    with st.popover("Save as favorite"):
        nm = st.text_input("Name", key=f"{key}_favname", placeholder="Home / Work")
        if st.button("Save", key=f"{key}_favsave") and nm.strip():
            S.setdefault("places", {})[nm.strip()] = asdict(p)
            persist_settings()
            st.session_state[f"{key}_fav_saved"] = nm.strip()
            st.rerun()
    return p


def headwind_colour(h):
    # red = headwind, green = tailwind
    x = float(np.clip(h / 20.0, -1, 1))
    if x >= 0:
        return [220, int(200 * (1 - x)) + 30, 40, 220]
    return [int(200 * (1 + x)) + 30, 190, 60, 220]


def show_map(route_coords, segs: pd.DataFrame, pts):
    layers = []
    if route_coords:
        layers.append(pdk.Layer("PathLayer", data=[{"path": [[lo, la] for la, lo in route_coords]}],
                                get_path="path", get_width=6, width_min_pixels=3, get_color=[60, 60, 60, 120]))
    d = segs[segs["leg"] == segs["leg"].iloc[0]].copy()
    d["colour"] = d["headwind"].apply(headwind_colour)
    d["tip"] = d.apply(lambda r: f"{r['mid_m']/1000:.1f} km - {'head' if r['headwind']>0 else 'tail'}wind "
                                 f"{abs(r['headwind']):.0f} km/h, felt {r['felt']:.0f} C", axis=1)
    layers.append(pdk.Layer("ScatterplotLayer", data=d, get_position=["lon", "lat"], get_fill_color="colour",
                            get_radius=120, radius_min_pixels=4, pickable=True))
    layers.append(pdk.Layer("ScatterplotLayer", data=pd.DataFrame(pts, columns=["lat", "lon"]),
                            get_position=["lon", "lat"], get_fill_color=[30, 90, 200, 255],
                            get_radius=250, radius_min_pixels=6))
    lat0 = float(np.mean([p[0] for p in (route_coords or pts)]))
    lon0 = float(np.mean([p[1] for p in (route_coords or pts)]))
    span = max(np.ptp([p[0] for p in (route_coords or pts)]), np.ptp([p[1] for p in (route_coords or pts)]) * 0.6, 0.02)
    zoom = float(np.clip(9.5 - np.log2(span / 0.1), 6, 13))
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=lat0, longitude=lon0, zoom=zoom),
                             tooltip={"text": "{tip}"}, map_style="light"), height=380)
    st.caption("Dots: headwind (red) / tailwind (green) along the outbound route. Blue: weather sample points.")


def source_chart(ens: adv.Ensemble, start: pd.Timestamp, end: pd.Timestamp):
    t0 = (start - pd.Timedelta(hours=3)).floor("h")
    t1 = (end + pd.Timedelta(hours=3)).ceil("h")
    grid = pd.date_range(t0, t1, freq="30min")
    ts = grid.as_unit("s").asi8.astype(float)
    ps = ens._per_source(0, ts)
    rows = []
    for k, name in enumerate(ens.names):
        for j, t in enumerate(grid):
            rows.append({"time": t.tz_convert(TZ).tz_localize(None), "source": name, "temp": ps["temp"][k, j],
                         "wind": ps["wind"][k, j], "precip": ps["precip"][k, j]})
    ens_v = ens.at(0, ts)
    for j, t in enumerate(grid):
        rows.append({"time": t.tz_convert(TZ).tz_localize(None), "source": "ENSEMBLE (+ radar, station)",
                     "temp": ens_v["temp"][j], "wind": ens_v["wind"][j], "precip": ens_v["precip"][j]})
    df = pd.DataFrame(rows)
    band = pd.DataFrame({"a": [start.tz_convert(TZ).tz_localize(None)], "b": [end.tz_convert(TZ).tz_localize(None)]})
    shade = alt.Chart(band).mark_rect(opacity=0.12, color="#1f77b4").encode(x="a:T", x2="b:T")
    sel = alt.selection_point(name="src", fields=["source"], bind="legend")
    width = alt.condition(alt.datum.source == "ENSEMBLE (+ radar, station)", alt.value(3.5), alt.value(1.4))

    def panel(col, title, with_sel=False):
        line = alt.Chart(df).mark_line().encode(
            x=alt.X("time:T", title=None), y=alt.Y(f"{col}:Q", title=title),
            color=alt.Color("source:N", legend=alt.Legend(orient="bottom", columns=4)),
            strokeWidth=width, opacity=alt.condition(sel, alt.value(1), alt.value(0.15)),
            tooltip=["source", alt.Tooltip("time:T", format="%H:%M"), alt.Tooltip(f"{col}:Q", format=".1f")],
        )
        if with_sel:
            line = line.add_params(sel)
        return (shade + line).properties(height=170)
    chart = alt.vconcat(panel("temp", "Temperature (C)", True), panel("wind", "Wind 10 m (km/h)"),
                        panel("precip", "Rain (mm/h)")).resolve_scale(color="shared")
    st.altair_chart(chart, width="stretch")
    st.caption("Forecast at the start point from every source; shaded = your ride. Click a legend entry to highlight it.")


def outfit_table(outfit: dict):
    rows = [{"": "Wear" if p.action == "wear" else "Take along", "Body": wd.SLOTS[p.slot]["zone"],
             "Item": p.item.name, "Why": p.reason} for p in outfit["picks"]]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    for w in outfit["warnings"]:
        st.warning(w)
    for t in outfit["tips"]:
        st.info(t)


def leg_metrics(st_: dict):
    c = st.columns(6)
    c[0].metric("Temperature", f"{st_['temp_min']:.0f}–{st_['temp_max']:.0f} °C")
    c[1].metric("Feels like on the bike", f"{st_['felt_dress']:.0f} °C",
                help=f"Range {st_['felt_min']:.0f} to {st_['felt_max']:.0f} °C along the ride")
    c[2].metric("Wind", f"{adv.beaufort(st_['wind_mean'])} Bft {adv.compass(st_['wind_dir_mean'])}",
                f"{'headwind' if st_['headwind_mean'] > 0 else 'tailwind'} {abs(st_['headwind_mean']):.0f} km/h",
                delta_color="inverse" if st_["headwind_mean"] > 0 else "normal")
    c[3].metric("Gusts", "–" if not np.isfinite(st_["gust_max"]) else f"{st_['gust_max']:.0f} km/h")
    c[4].metric("Rain", f"{st_['rain_mm']:.1f} mm", f"max chance {st_['pop_max']:.0f}%", delta_color="off")
    c[5].metric("Daylight", "dark" if st_["dark_fraction"] > 0.5 else ("partly dark" if st_["dark_fraction"] > 0 else "light"))


# ==========================================================================
# PLAN TAB
# ==========================================================================
with tab_plan:
    if not st.session_state.wardrobe:
        st.info("Your wardrobe is empty - add your cycling clothes on the **My wardrobe** tab first.")
    route_src = st.radio("Route", ["Start → destination (auto-route by bike)", "Upload a GPX file"],
                         horizontal=True, key="route_src")
    use_gpx = route_src.startswith("Upload")
    start, dest, loop_km, gpx_route, gpx_rev, gpx_snap = None, None, None, None, False, False
    if use_gpx:
        g1, g2 = st.columns(2)
        with g1:
            up = st.file_uploader("GPX file - a track or route exported from e.g. Komoot, Strava, Garmin or RouteYou",
                                  type=["gpx"], key="gpx_file")
            if up is not None:
                try:
                    gpx_route = cached_gpx(up.getvalue())
                except ValueError as e:
                    st.error(f"Could not read the GPX file: {e}")
        with g2:
            if gpx_route:
                gain = gpx_route["elev_gain_m"]
                st.markdown(f"**{gpx_route['name']}**  \n"
                            f"{gpx_route['distance_m'] / 1000:.1f} km · {gpx_route['n_points']} {gpx_route['kind']} points"
                            + (f" · ↑ {gain:.0f} m" if gain is not None else "")
                            + (" · loop" if gpx_route["is_loop"] else ""))
                gpx_rev = st.checkbox("Ride it in the reverse direction", key="gpx_rev")
                if gpx_route["kind"] != "track" and gpx_route["n_points"] <= 25:
                    gpx_snap = st.checkbox("Follow bike roads between the GPX points", value=True, key="gpx_snap",
                                           help="Route-planner exports often contain only a few turn points; "
                                                "this routes between them over cycle paths (OSRM bike).")
                ride_type = st.radio("Ride type", ["Ride the GPX route", "There and back (return the same way)"],
                                     key="gpx_ride_type")
                mode = "return" if ride_type.startswith("There") else "oneway"
                la, lo = gpx_route["coords"][-1 if gpx_rev else 0]
                start = routing.Place(f"{gpx_route['name']} (start)", la, lo)
                if not routing.in_netherlands(la, lo):
                    st.warning("The route starts outside the Netherlands; Buienradar and HARMONIE coverage may be missing.")
            else:
                st.caption("Upload a .gpx file to plan with your own route.")
        if not gpx_route:
            mode = "oneway"
    else:
        mode_label = st.radio("Ride type", ["One way", "There and back", "Loop from start"], horizontal=True,
                              key="ride_type")
        mode = {"One way": "oneway", "There and back": "return", "Loop from start": "loop"}[mode_label]
        c1, c2 = st.columns(2)
        with c1:
            start = place_picker("start", "Start")
        with c2:
            if mode == "loop":
                loop_km = st.slider("Loop distance (km)", 5, 200, 40, 5)
                st.caption("Unknown route: the headwind is modelled for half of the loop.")
            else:
                dest = place_picker("dest", "Destination")
    now_l = datetime.now(TZ)
    d1, d2, d3, d4 = st.columns(4)
    dep_date = d1.date_input("Departure date", now_l.date(), min_value=now_l.date(),
                             max_value=now_l.date() + timedelta(days=3))
    default_t = (now_l + timedelta(minutes=15 - now_l.minute % 15)).time().replace(second=0, microsecond=0)
    dep_time = d2.time_input("Departure time", default_t, step=timedelta(minutes=15))
    ret_dt = None
    if mode == "return":
        rd = d3.date_input("Return date", dep_date, min_value=dep_date, max_value=now_l.date() + timedelta(days=3))
        rt_ = d4.time_input("Return departure", (datetime.combine(dep_date, dep_time) + timedelta(hours=8)).time(),
                            step=timedelta(minutes=15))
        ret_dt = datetime.combine(rd, rt_, tzinfo=TZ)
    flex = st.slider("Flexible departure (± hours) - suggest the best time", 0, 6, 2)

    ready = gpx_route is not None if use_gpx else (start is not None and (mode == "loop" or dest is not None))
    go = st.button("Get advice", type="primary", disabled=not ready)
    if go:
        dep = pd.Timestamp(datetime.combine(dep_date, dep_time, tzinfo=TZ)).tz_convert("UTC")
        try:
            with st.status("Fetching route and live weather ...", expanded=False) as status:
                if use_gpx:
                    route = gpx_route
                    if gpx_rev:
                        route = routing.reverse_route(route)
                    if gpx_snap:
                        route = cached_route(tuple(route["coords"]))
                    sp = routing.sample_points(route["coords"])
                elif mode == "loop":
                    route, sp = None, [(start.lat, start.lon, 0.0)]
                else:
                    route = cached_route(((start.lat, start.lon), (dest.lat, dest.lon)))
                    sp = routing.sample_points(route["coords"])
                pts = tuple((round(a, 4), round(b, 4)) for a, b, _ in sp)
                data = cached_weather(pts)
                if not any(src.ok for src in data["forecasts"]):
                    cached_weather.clear()
                    raise RuntimeError("none of the weather forecast services could be reached - "
                                       "check your internet connection and try again. Details: "
                                       + "; ".join(f"{src.name}: {src.error}" for src in data["forecasts"])[:400])
                ens = adv.Ensemble(data, weights)
                ret_h = 0.0 if ret_dt is None else (pd.Timestamp(ret_dt).tz_convert("UTC") - dep).total_seconds() / 3600
                legs = adv.build_legs(mode, route, (start.lat, start.lon), loop_km, ret_h, list(pts),
                                      [d for *_, d in sp])
                alarms = data["alarms"].meta.get("warnings", []) if data["alarms"].ok else []
                plan = adv.evaluate_plan(ens, legs, dep, prefs, st.session_state.wardrobe, alarms)
                scan = adv.scan_departures(ens, legs, dep, prefs, st.session_state.wardrobe, alarms, flex) \
                    if flex > 0 else pd.DataFrame()
                status.update(label="Done", state="complete")
            st.session_state.result = {"plan": plan, "scan": scan, "data": data, "ens": ens, "route": route,
                                       "pts": list(pts), "dep": dep, "mode": mode,
                                       "labels": (start.label, dest.label if dest else None)}
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not build the advice: {type(e).__name__}: {e}")

    R = st.session_state.get("result")
    if R:
        plan, data, ens = R["plan"], R["data"], R["ens"]
        st.divider()
        lvl = plan["level"]
        box = {0: st.success, 1: st.warning, 2: st.error}[lvl]
        box(f"### {plan['label']}\nComfort score **{plan['score']:.0f}/100** · forecast confidence "
            f"**{plan['confidence']}** · weather fetched {local(data['fetched_at'])}")
        reasons = [r for L in plan["legs"] for r in L["verdict"]["reasons"] if r[0] > 0]
        for sev, msg in dict.fromkeys(reasons):
            st.markdown(f"- {'⛔' if sev == 2 else '⚠️'} {msg}")

        scan = R["scan"]
        if len(scan):
            best = scan.sort_values(["level", "score"], ascending=[True, False]).iloc[0]
            if best["departure"] != R["dep"] and best["score"] > plan["score"] + 3:
                st.info(f"**Better departure: {local(best['departure'])}** - score {best['score']:.0f} vs "
                        f"{plan['score']:.0f} ({best['verdict']}, {best['rain_mm']:.1f} mm rain, "
                        f"feels {best['felt_dress']:.0f} °C).")
            else:
                st.caption("Your chosen departure is already (close to) the best in the window.")

        first_picks = None
        for L in plan["legs"]:
            leg, s_ = L["leg"], L["stats"]
            if not s_.get("valid"):
                st.error(f"{leg.name}: no forecast data for this time.")
                continue
            st.subheader(f"{leg.name}: {local(s_['start'], '%a %H:%M')} → {local(s_['end'])}  ·  "
                         f"{s_['distance_km']:.1f} km, {s_['duration_min']:.0f} min")
            leg_metrics(s_)
            if first_picks is not None:
                prev = {p.item.id for p in first_picks if p.action == "wear"}
                now_ = {p.item.id for p in L["outfit"]["picks"] if p.action == "wear"}
                names = {p.item.id: p.item.name for p in first_picks + L["outfit"]["picks"]}
                add, rem = now_ - prev, prev - now_
                if add or rem:
                    st.markdown("**Changes for this leg:** " +
                                ("; ".join([f"add *{names[i]}*" for i in add] + [f"stow *{names[i]}*" for i in rem])))
                    st.caption("Bring the extra items along on the way out.")
                with st.expander("Full outfit for this leg"):
                    outfit_table(L["outfit"])
            else:
                st.markdown("#### What to wear")
                outfit_table(L["outfit"])
                first_picks = L["outfit"]["picks"]
            for w in L["warnings"]:
                st.error(f"KNMI code {w['colour']}: **{w['event']}** - {w['headline']} "
                         f"({local(w['onset'], '%a %H:%M') if w['onset'] is not None else ''} – "
                         f"{local(w['expires'], '%a %H:%M') if w['expires'] is not None else ''})")

        valid_segs = [L["seg"] for L in plan["legs"] if L["stats"].get("valid")]
        allseg = pd.concat(valid_segs) if valid_segs else None
        st.divider()
        m1, m2 = st.columns([1, 1]) if allseg is not None else (None, None)
    if R and allseg is not None:
        with m1:
            st.markdown("#### Route and wind")
            show_map(R["route"]["coords"] if R["route"] else None, allseg, R["pts"])
        with m2:
            st.markdown("#### Sources compared")
            s0 = plan["legs"][0]["stats"]
            if s0.get("valid"):
                source_chart(ens, s0["start"], plan["legs"][0]["stats"]["end"])

        if len(scan):
            st.markdown("#### Best departure time")
            sc = scan.copy()
            sc["time"] = sc["departure"].dt.tz_convert(TZ).dt.tz_localize(None)
            base = alt.Chart(sc).encode(x=alt.X("time:T", title="Departure"))
            ch = base.mark_line(point=True).encode(
                y=alt.Y("score:Q", title="Comfort score", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("verdict:N", scale=alt.Scale(domain=list(adv.LEVELS.values()),
                                                            range=["#2ca02c", "#ff9f0e", "#d62728"])),
                tooltip=[alt.Tooltip("time:T", format="%H:%M"), alt.Tooltip("score:Q", format=".0f"), "verdict",
                         alt.Tooltip("rain_mm:Q", format=".1f"), alt.Tooltip("felt_dress:Q", format=".1f"),
                         alt.Tooltip("headwind:Q", format=".0f")])
            st.altair_chart(ch.properties(height=220), width="stretch")

        with st.expander("Live observations, warnings and source status"):
            stn = ens.stations[0]
            if stn:
                st.markdown(f"**Nearest station: {stn['name']}** ({stn['distance_km']:.0f} km, "
                            f"{local(stn['time'])}) - {stn['description']}")
                st.write(pd.DataFrame([{k: stn[k] for k in ["temp", "feels", "ground_temp", "wind", "gust",
                                                            "humidity", "precip_rate", "visibility_m", "rain_24h"]}]))
                b = ens.bias[0]
                if b:
                    st.caption(f"Station-based correction applied to the first hours: temperature "
                               f"{b['temp']:+.1f} °C, wind {b['wind']:+.1f} km/h (fades out over "
                               f"{adv.STATION_BIAS_DECAY_H:.0f} h).")
            if data["stations"].ok and data["stations"].meta.get("report_title"):
                st.markdown(f"**Buienradar weerbericht:** *{data['stations'].meta['report_title']}* - "
                            f"{data['stations'].meta['report_summary']}")
            all_w = [w for w in (data["alarms"].meta.get("warnings", []) if data["alarms"].ok else [])
                     if w["level"] >= 2 and (w["expires"] is None or w["expires"] > pd.Timestamp.now(tz="UTC"))]
            st.markdown(f"**Active KNMI warnings in NL:** {len(all_w)}")
            for w in all_w:
                st.markdown(f"- code {w['colour']}: {w['event']} - {w['areas'][:120]}")
            rows = []
            for s in data["forecasts"] + [data["nowcast"], data["stations"], data["alarms"]]:
                rows.append({"source": s.name, "status": "ok" if s.ok else f"error: {s.error}",
                             "latency (s)": round(s.latency_s, 2),
                             "weight": str(weights.get(s.name, "–")),
                             "data until": local(s.meta["horizon"], "%a %H:%M") if s.meta.get("horizon") is not None else ""})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# ==========================================================================
# WARDROBE TAB
# ==========================================================================
COLS = ["name", "slot", "t_min", "t_max", "waterproof", "windproof", "long_cover", "reflective", "lens", "source", "id"]


def items_to_df(items):
    df = pd.DataFrame([asdict(i) for i in items], columns=COLS)
    df["slot"] = df["slot"].map(wd.SLOT_LABELS)
    return df


def df_to_items(df) -> tuple[list[wd.Item], list[str]]:
    items, errs = [], []
    for _, r in df.iterrows():
        if not isinstance(r["name"], str) or not r["name"].strip():
            continue
        slot = wd.LABEL_TO_SLOT.get(r["slot"])
        if slot is None:
            errs.append(f"'{r['name']}': choose a body slot.")
            continue
        try:
            lo, hi = float(r["t_min"]), float(r["t_max"])
        except (TypeError, ValueError):
            errs.append(f"'{r['name']}': temperature range missing.")
            continue
        if lo > hi:
            lo, hi = hi, lo
        kw = dict(name=r["name"].strip(), slot=slot, t_min=lo, t_max=hi,
                  waterproof=bool(r["waterproof"]), windproof=bool(r["windproof"]),
                  long_cover=bool(r["long_cover"]), reflective=bool(r["reflective"]),
                  lens=r["lens"] if isinstance(r["lens"], str) else "",
                  source=r["source"] if isinstance(r["source"], str) else "")
        if isinstance(r["id"], str) and r["id"]:
            kw["id"] = r["id"]
        items.append(wd.Item(**kw))
    return items, errs


with tab_wardrobe:
    W = st.session_state.wardrobe
    st.markdown(f"**{len(W)} items** saved in `{wd.WARDROBE_FILE}`")
    a1, a2 = st.columns([2, 1])
    with a1:
        have = {i.name for i in W}
        cat = [n for n in wd.catalogue_by_name() if n not in have]
        add = st.multiselect("Quick add from the starter catalogue (typical comfort ranges - edit them afterwards)",
                             cat)
        if st.button("Add selected", disabled=not add):
            cb = wd.catalogue_by_name()
            for n in add:
                d = asdict(cb[n])
                d.pop("id")
                W.append(wd.Item(**d))
            wd.save_wardrobe(W)
            st.rerun()
    with a2:
        st.download_button("Export wardrobe (JSON)", json.dumps({"version": 1, "items": [asdict(i) for i in W]},
                                                                 indent=2), "wardrobe.json", "application/json")
        up = st.file_uploader("Import wardrobe (JSON)", type=["json"], label_visibility="collapsed")
        if up is not None and st.button("Replace wardrobe with upload"):
            raw = json.loads(up.read().decode("utf-8"))
            known = set(wd.Item.__dataclass_fields__)
            st.session_state.wardrobe = [wd.Item(**{k: v for k, v in d.items() if k in known})
                                         for d in raw.get("items", []) if d.get("slot") in wd.SLOTS]
            wd.save_wardrobe(st.session_state.wardrobe)
            st.rerun()

    st.markdown("#### Edit items")
    st.caption("Comfort range = the *felt* riding temperature at which you would wear the item. "
               "Add rows at the bottom; select rows and press Delete to remove. Press **Save wardrobe** afterwards.")
    edited = st.data_editor(
        items_to_df(W), num_rows="dynamic", hide_index=True, width="stretch", key="wardrobe_editor",
        column_config={
            "name": st.column_config.TextColumn("Item", required=True, width="medium"),
            "slot": st.column_config.SelectboxColumn("Body slot", options=list(wd.SLOT_LABELS.values()), required=True),
            "t_min": st.column_config.NumberColumn("Min °C", min_value=-30, max_value=45, step=1, required=True),
            "t_max": st.column_config.NumberColumn("Max °C", min_value=-30, max_value=45, step=1, required=True),
            "waterproof": st.column_config.CheckboxColumn("Waterproof", default=False),
            "windproof": st.column_config.CheckboxColumn("Windproof", default=False),
            "long_cover": st.column_config.CheckboxColumn("Long sleeves / legs", default=False),
            "reflective": st.column_config.CheckboxColumn("Reflective", default=False),
            "lens": st.column_config.SelectboxColumn("Lens", options=wd.LENS_OPTIONS),
            "source": st.column_config.TextColumn("Source / URL"),
            "id": None,
        })
    if st.button("Save wardrobe", type="primary"):
        items, errs = df_to_items(edited)
        for e in errs:
            st.error(e)
        if not errs:
            st.session_state.wardrobe = items
            wd.save_wardrobe(items)
            st.success(f"Saved {len(items)} items.")
            st.rerun()

    with st.expander("Import an item from Decathlon NL (reads its comfort temperature)"):
        st.caption("Uses a browser (Playwright) to read the product page, as in the original cycloth.py. "
                   "Needs `pip install playwright beautifulsoup4` and `playwright install chromium`.")
        pid = st.text_input("Product id or URL", placeholder="8804566 or https://www.decathlon.nl/p/...")
        headless = st.checkbox("Run browser hidden (headless)", value=False,
                               help="Some shops block hidden browsers; leave off if import fails.")
        if st.button("Read rating", disabled=not pid):
            import decathlon
            with st.spinner("Opening product page ..."):
                st.session_state.dec = decathlon.fetch_rating(pid, headless=headless)
        d = st.session_state.get("dec")
        if d:
            if "error" in d:
                st.error(d["error"])
            else:
                with st.form("dec_add"):
                    nm = st.text_input("Name", d["name"])
                    sl = st.selectbox("Body slot", list(wd.SLOT_LABELS.values()),
                                      index=list(wd.SLOTS).index(wd.guess_slot(d["name"])))
                    lo, hi = st.slider("Comfort range (°C)", -30.0, 45.0, (float(d["t_min"]), float(d["t_max"])), 0.5)
                    f1, f2, f3 = st.columns(3)
                    wp = f1.checkbox("Waterproof")
                    wdp = f2.checkbox("Windproof")
                    lc = f3.checkbox("Long sleeves / legs")
                    if st.form_submit_button("Add to wardrobe"):
                        st.session_state.wardrobe.append(wd.Item(nm, wd.LABEL_TO_SLOT[sl], lo, hi, wp, wdp, lc,
                                                                 source=d["url"]))
                        wd.save_wardrobe(st.session_state.wardrobe)
                        st.session_state.dec = None
                        st.rerun()

    if W:
        st.markdown("#### Temperature coverage")
        cov = pd.DataFrame([{"slot": wd.SLOT_LABELS[i.slot], "item": i.name, "lo": i.t_min, "hi": i.t_max,
                             "order": list(wd.SLOTS).index(i.slot)} for i in W])
        ch = alt.Chart(cov).mark_bar(height=10, cornerRadius=3).encode(
            x=alt.X("lo:Q", title="Felt temperature (°C)", scale=alt.Scale(domain=[-20, 40])), x2="hi:Q",
            y=alt.Y("item:N", sort=alt.EncodingSortField("order"), title=None),
            color=alt.Color("slot:N", legend=None), tooltip=["item", "slot", "lo", "hi"])
        rules = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4], color="#888").encode(x="x:Q")
        st.altair_chart((ch + rules).properties(height=max(150, 22 * len(W))), width="stretch")
        st.caption("Gaps in a slot mean the advisor will warn or fall back to the closest item.")


# ==========================================================================
# METHOD TAB
# ==========================================================================
with tab_method:
    st.markdown(f"""
### Data sources (all live, keyless)
| Source | What it adds |
|---|---|
| **KNMI HARMONIE-AROME** (via Open-Meteo) | 2.5 km Dutch model; best for local wind and showers (≈60 h) |
| **DWD ICON-D2 / EU**, **Météo-France AROME**, **ECMWF IFS**, **NOAA GFS** (via Open-Meteo) | independent models; their spread gives forecast confidence |
| **MET Norway** Locationforecast | separately post-processed ECMWF-based forecast |
| **Buienradar radar nowcast** | 5-min rain forecast for the next 2 h - replaces model rain for the first hour, then fades to 50 % by 2 h |
| **Buienradar / KNMI stations** | current measurements; the temperature/wind bias of the ensemble vs the nearest station is corrected, fading over {adv.STATION_BIAS_DECAY_H:.0f} h; ground temperature is used for ice risk |
| **MeteoAlarm** | official KNMI warnings (code yellow / orange / red) matched to your route polygon and time |
| **PDOK Locatieserver** | Dutch address geocoding |
| **OSRM bicycle profile** (routing.openstreetmap.de) | bike route, distance and headings |

### How the advice is computed
1. The route is cut into ~400 m segments; each gets its passage time (from your average speed), heading and the ensemble
   weather of the nearest of up to 5 sample points.
2. **Ensemble** = weighted mean over models (weights in the sidebar); rain probability = mean of each source's probability
   (or 0/100 % from its rain amount when a source has none).
3. **Headwind** = wind × cos(wind direction − heading). Wind at rider height ≈ {adv.RIDER_HEIGHT_FACTOR:.1f} × 10 m wind.
4. **Felt temperature** = air temperature + wind-chill difference between your actual airspeed and the
   {adv.V_REF_KMH:.0f} km/h at which garment ranges are assumed to be rated + effort offset
   ({', '.join(f'{k.split(" (")[0]} {v:+.0f}' for k, v in adv.INTENSITY.items())} °C) + your personal offset
   + sun (up to +2 °C) − rain (up to −2 °C).
5. You dress for the **{adv.DRESS_PERCENTILE}th percentile** of felt temperature along the ride (a little on the cool side,
   because you warm up). Per body slot the item whose comfort range contains that temperature and is centred closest to it
   is chosen; under a jacket the mid layer is chosen for ~{adv.JACKET_BOOST:.0f} °C warmer. Rain gear is worn when rain
   is expected (≥ 0.5 mm, ≥ 0.5 mm/h or ≥ 65 % chance) and packed when it is possible (≥ 30 %).
6. **Verdict** - *No-go*: KNMI code orange/red, gusts ≥ 75 km/h, mean wind ≥ 50 km/h, thunderstorms in ≥ 35 % of the
   ensemble, freezing rain, very heavy rain, felt ≤ −12 °C or ≥ 35 °C. *Caution*: code yellow, gusts ≥ 60 km/h,
   6 Bft, strong headwind, near-freezing wet roads, snow, ≥ 2 mm rain, missing wardrobe items.
   Confidence comes from the model spread and lead time.

These are heuristics for guidance; always use your own judgement on the road.
""")
