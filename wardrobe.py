"""Wardrobe model, starter catalogue and JSON persistence.

Every item carries a comfort-temperature range (t_min..t_max, deg C) in the same
spirit as the ranges Decathlon publishes ("comforttemperatuur"). The ranges are
interpreted as *felt riding temperature* (see advisor.felt_temperature).
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Storage location (override with the CYCLOTH_DATA_DIR environment variable)
# --------------------------------------------------------------------------
DATA_DIR = Path(os.environ.get("CYCLOTH_DATA_DIR", Path.home() / ".cycloth"))
WARDROBE_FILE = DATA_DIR / "wardrobe.json"
SETTINGS_FILE = DATA_DIR / "settings.json"

# --------------------------------------------------------------------------
# Slots: where on the body an item goes and how the advisor treats the slot
#   required      : always choose something (closest match if nothing fits)
#   warn_below    : if felt temp is below this and nothing fits -> warning
#   kind          : "temp"  -> chosen purely on comfort range
#                   "rain"  -> chosen when rain is expected
#                   "wind"  -> chosen on range or when windy
#                   "supp"  -> supplement (arm/knee warmers) for short garments
#                   "eyes" / "vis" -> condition based
# --------------------------------------------------------------------------
SLOTS: dict[str, dict] = {
    "base":       {"label": "Base layer",                  "zone": "Torso", "kind": "temp", "required": False, "warn_below": None},
    "jersey":     {"label": "Jersey / top",                "zone": "Torso", "kind": "temp", "required": True,  "warn_below": None},
    "insulation": {"label": "Thermal / winter jacket",     "zone": "Torso", "kind": "temp", "required": False, "warn_below": 5},
    "wind":       {"label": "Wind vest / wind jacket",     "zone": "Torso", "kind": "wind", "required": False, "warn_below": None},
    "rain_top":   {"label": "Rain jacket / poncho",        "zone": "Torso", "kind": "rain", "required": False, "warn_below": None},
    "arms":       {"label": "Arm warmers",                 "zone": "Arms",  "kind": "supp", "required": False, "warn_below": None},
    "legs":       {"label": "Shorts / tights / trousers",  "zone": "Legs",  "kind": "temp", "required": True,  "warn_below": None},
    "knees":      {"label": "Knee / leg warmers",          "zone": "Legs",  "kind": "supp", "required": False, "warn_below": None},
    "rain_legs":  {"label": "Rain trousers",               "zone": "Legs",  "kind": "rain", "required": False, "warn_below": None},
    "hands":      {"label": "Gloves",                      "zone": "Hands", "kind": "temp", "required": False, "warn_below": 10},
    "feet":       {"label": "Socks",                       "zone": "Feet",  "kind": "temp", "required": False, "warn_below": None},
    "overshoes":  {"label": "Overshoes / toe covers",      "zone": "Feet",  "kind": "temp", "required": False, "warn_below": 3},
    "head":       {"label": "Cap / skull cap / headband",  "zone": "Head",  "kind": "temp", "required": False, "warn_below": 5},
    "neck":       {"label": "Neck warmer / buff",          "zone": "Head",  "kind": "temp", "required": False, "warn_below": 3},
    "eyes":       {"label": "Glasses",                     "zone": "Head",  "kind": "eyes", "required": False, "warn_below": None},
    "visibility": {"label": "Reflective / hi-vis",         "zone": "Extras","kind": "vis",  "required": False, "warn_below": None},
}
SLOT_LABELS = {k: v["label"] for k, v in SLOTS.items()}
LABEL_TO_SLOT = {v: k for k, v in SLOT_LABELS.items()}
LENS_OPTIONS = ["", "sun", "clear", "photochromic"]


@dataclass
class Item:
    name: str
    slot: str
    t_min: float
    t_max: float
    waterproof: bool = False
    windproof: bool = False
    long_cover: bool = False      # long sleeves (tops) or full-length (legs)
    reflective: bool = False
    lens: str = ""                # eyes only: sun / clear / photochromic
    source: str = ""              # e.g. Decathlon URL
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def covers(self, t: float, tol: float = 1.0) -> bool:
        return self.t_min - tol <= t <= self.t_max + tol

    def distance(self, t: float) -> float:
        if t < self.t_min:
            return self.t_min - t
        if t > self.t_max:
            return t - self.t_max
        return 0.0


def _i(name, slot, lo, hi, **kw) -> Item:
    return Item(name=name, slot=slot, t_min=lo, t_max=hi, **kw)


# Starter catalogue with typical comfort ranges (felt riding temperature, deg C).
# These are sensible defaults compiled from common cycling layering guidance;
# edit them to match your own garments / the manufacturer's rating.
STARTER_CATALOGUE: list[Item] = [
    _i("Sleeveless mesh base layer", "base", 12, 30),
    _i("Long-sleeve thermal base layer", "base", -10, 12, long_cover=True),
    _i("Merino long-sleeve base layer", "base", -5, 15, long_cover=True),
    _i("Short-sleeve jersey", "jersey", 15, 35),
    _i("Long-sleeve jersey", "jersey", 9, 18, long_cover=True),
    _i("Thermal long-sleeve jersey", "jersey", 0, 12, long_cover=True),
    _i("Casual T-shirt / shirt", "jersey", 14, 32),
    _i("Sweater / hoodie", "jersey", 4, 15, long_cover=True),
    _i("Softshell winter jacket", "insulation", -10, 8, windproof=True, long_cover=True),
    _i("Everyday winter coat", "insulation", -8, 8, windproof=True, long_cover=True),
    _i("Wind vest (gilet)", "wind", 8, 18, windproof=True),
    _i("Packable wind jacket", "wind", 5, 15, windproof=True, long_cover=True),
    _i("Waterproof rain jacket", "rain_top", -10, 20, waterproof=True, windproof=True, long_cover=True),
    _i("Rain poncho / cape", "rain_top", 0, 25, waterproof=True),
    _i("Arm warmers", "arms", 10, 16, long_cover=True),
    _i("Bib shorts", "legs", 15, 35),
    _i("Thermal bib tights", "legs", -10, 10, long_cover=True),
    _i("3/4 knickers", "legs", 10, 17),
    _i("Casual trousers / jeans", "legs", 2, 24, long_cover=True),
    _i("Knee warmers", "knees", 12, 17),
    _i("Leg warmers", "knees", 6, 14, long_cover=True),
    _i("Rain trousers", "rain_legs", -10, 20, waterproof=True, long_cover=True),
    _i("Fingerless summer mitts", "hands", 15, 35),
    _i("Light full-finger gloves", "hands", 8, 16),
    _i("Winter gloves", "hands", -5, 8, windproof=True),
    _i("Lobster / deep-winter gloves", "hands", -20, 0, windproof=True, waterproof=True),
    _i("Summer cycling socks", "feet", 12, 35),
    _i("Wool winter socks", "feet", -10, 13),
    _i("Toe covers", "overshoes", 4, 11),
    _i("Neoprene overshoes", "overshoes", -10, 6, waterproof=True, windproof=True),
    _i("Cycling cap", "head", 8, 22),
    _i("Thermal skull cap", "head", -10, 8),
    _i("Ear-warmer headband", "head", 0, 10),
    _i("Neck warmer / buff", "neck", -10, 12),
    _i("Sunglasses", "eyes", -20, 45, lens="sun"),
    _i("Clear-lens glasses", "eyes", -20, 45, lens="clear"),
    _i("Reflective vest / sash", "visibility", -20, 45, reflective=True),
]


def catalogue_by_name() -> dict[str, Item]:
    return {i.name: i for i in STARTER_CATALOGUE}


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def _atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_wardrobe() -> list[Item]:
    if not WARDROBE_FILE.exists():
        return []
    raw = json.loads(WARDROBE_FILE.read_text(encoding="utf-8"))
    known = set(Item.__dataclass_fields__)
    items = []
    for d in raw.get("items", []):
        d = {k: v for k, v in d.items() if k in known}
        if d.get("slot") in SLOTS:
            items.append(Item(**d))
    return items


def save_wardrobe(items: list[Item]) -> None:
    _atomic_write(WARDROBE_FILE, {"version": 1, "items": [asdict(i) for i in items]})


DEFAULT_SETTINGS = {
    "speed_kmh": 18.0,
    "intensity": "Moderate (touring)",
    "personal_offset": 0.0,
    "places": {},                      # name -> {"label", "lat", "lon"}
    "weights": {},                     # source name -> weight override
}


def load_settings() -> dict:
    s = json.loads(json.dumps(DEFAULT_SETTINGS))
    if SETTINGS_FILE.exists():
        try:
            s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return s


def save_settings(settings: dict) -> None:
    _atomic_write(SETTINGS_FILE, settings)


# --------------------------------------------------------------------------
# Guessing a slot from a (Dutch or English) product name, used by the
# Decathlon importer.
# --------------------------------------------------------------------------
_SLOT_KEYWORDS = [
    ("rain_legs", ["regenbroek", "rain trouser", "rain pant", "overbroek"]),
    ("rain_top", ["regenjas", "regenjack", "poncho", "regencape", "rain jacket", "waterdicht jack"]),
    ("overshoes", ["overschoen", "overshoe", "toe cover", "teenkap"]),
    ("knees", ["beenstuk", "kniestuk", "knee warmer", "leg warmer", "beenwarmer"]),
    ("arms", ["armstuk", "arm warmer", "armwarmer"]),
    ("wind", ["bodywarmer", "windvest", "gilet", "windjack", "windstopper", "wind jacket"]),
    ("base", ["onderhemd", "ondershirt", "baselayer", "base layer", "onderlaag", "thermoshirt"]),
    ("hands", ["handschoen", "glove", "want"]),
    ("feet", ["sok", "sock"]),
    ("neck", ["nekwarmer", "col ", "buff", "neck"]),
    ("head", ["muts", "pet ", "cap", "hoofdband", "headband", "onderhelm"]),
    ("eyes", ["bril", "glasses", "sunglass"]),
    ("visibility", ["reflect", "hesje", "hi-vis", "fluo"]),
    ("legs", ["broek", "short", "tight", "bib", "knicker"]),
    ("insulation", ["winterjas", "winterjack", "jas", "jack", "jacket", "softshell"]),
    ("jersey", ["shirt", "jersey", "trui", "maillot", "top"]),
]


def guess_slot(name: str) -> str:
    n = f" {name.lower()} "
    for slot, words in _SLOT_KEYWORDS:
        if any(w in n for w in words):
            return slot
    return "jersey"
