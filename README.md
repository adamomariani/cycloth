# Cycloth – cycling clothing advisor (Netherlands)

A local Streamlit web-app that tells you **whether to ride** and **what to wear from your own cycling wardrobe**,
based on your route and a live multi-source weather ensemble.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```
The browser opens at http://localhost:8501. Wardrobe and settings are stored in `%USERPROFILE%\.cycloth\`
(`wardrobe.json`, `settings.json`); set the environment variable `CYCLOTH_DATA_DIR` to store them elsewhere.

Optional Decathlon import (reads an item's comfort temperature, adapted from `cycloth.py`):
```powershell
pip install playwright beautifulsoup4
playwright install chromium
```

## Files
| File | Role |
|---|---|
| `app.py` | Streamlit interface: *Plan ride*, *My wardrobe*, *Method & sources* |
| `wardrobe.py` | Item model, body slots, starter catalogue with typical comfort ranges, JSON storage |
| `routing.py` | PDOK geocoding, OSRM bicycle routing, GPX import, route segmentation and headings |
| `weather_sources.py` | Live fetchers: Open-Meteo (KNMI HARMONIE, DWD ICON, Météo-France AROME, ECMWF, GFS), MET Norway, Buienradar radar nowcast + stations, MeteoAlarm (KNMI warnings) |
| `advisor.py` | Weighted ensemble + radar/station fusion, ride simulation, felt temperature, outfit selection, go/no-go verdict, best-departure scan |
| `decathlon.py` | Optional Decathlon NL comfort-temperature reader |

## Using it
1. **My wardrobe** – quick-add items from the starter catalogue, then edit names and comfort ranges (min/max °C) and
   flags (waterproof, windproof, long sleeves/legs, reflective, lens type). Press **Save wardrobe**.
2. **Plan ride** – pick the route source:
   * *Start → destination*: choose *One way*, *There and back* (with return time) or *Loop*; type an address, street,
     town or postcode; save frequent places as favourites.
   * *Upload a GPX file*: a track or route exported from Komoot, Strava, Garmin, RouteYou, etc. (GPX 1.0/1.1; track
     points are used first, then route points, then waypoints). Optionally ride it in reverse or there-and-back. For
     sparse route-planner files (≤ 25 route points) the app can follow bike roads between the points.
   Then choose the departure time and how flexible you are.
3. Read the verdict (GO / GO WITH CAUTION / RECONSIDER), the outfit per leg (what to wear, what to take along, what to
   add or stow for the return), the headwind map, the source comparison, and the best departure time in your window.

All services used are free and need no API key. The advice is heuristic guidance; use your own judgement on the road.
