"""Optional: read a product's comfort-temperature range from decathlon.nl.

Adapted from the user's cycloth.py prototype (Playwright + BeautifulSoup).
Requires:  pip install playwright beautifulsoup4  &&  playwright install chromium
Decathlon's page markup can change at any time; failures return an error string.
"""
from __future__ import annotations

import re

TEMP_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*°?\s*C", re.IGNORECASE)


def _page_html(url: str, headless: bool) -> str:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")
        tab = page.locator("#product-info-specifications")
        if tab.count() > 0:
            tab.first.click()
            page.wait_for_timeout(500)
        html = page.content()
        browser.close()
        return html


def fetch_rating(product: str, headless: bool = False) -> dict:
    """product: Decathlon NL product id (e.g. '8804566') or full product URL.

    Returns {"name", "t_min", "t_max", "url"} or {"error": "..."}.
    """
    product = product.strip()
    url = product if product.startswith("http") else f"https://www.decathlon.nl/p/{product}"
    try:
        from bs4 import BeautifulSoup
        html = _page_html(url, headless)
    except ImportError:
        return {"error": "Playwright/BeautifulSoup not installed: "
                         "pip install playwright beautifulsoup4 && playwright install chromium"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Could not load page: {type(e).__name__}: {e}"}

    soup = BeautifulSoup(html, "html.parser")
    name_el = soup.find("div", {"class": "product-info__name", "data-cs-override-id": "product_title_main"}) \
        or soup.find("h1")
    name = name_el.get_text(strip=True) if name_el else product
    temps: list[float] = []

    # 1) benefit cards ("Warmte", "Optimale temperatuur", ...)
    for card in soup.find_all("article", class_="benefit-card"):
        title = card.find("h4")
        if title is None or title.get_text(strip=True).lower() not in {"warmte", "optimale temperatuur", "temperatuur"}:
            continue
        val = card.find("p")
        if val is not None:
            temps += [float(v.replace(",", ".")) for v in TEMP_RE.findall(val.get_text(strip=True))]

    # 2) specification table ("Comforttemperatuur", "Temperatuur")
    for row in soup.find_all("div", class_="specifications__characteristics-row"):
        dt, dd = row.find("dt"), row.find("dd")
        if dt is None or dd is None:
            continue
        if dt.get_text(strip=True).lower() in {"comforttemperatuur", "temperatuur", "gebruikstemperatuur"}:
            temps += [float(v.replace(",", ".")) for v in TEMP_RE.findall(dd.get_text(strip=True))]

    if not temps:
        return {"error": f"No temperature rating found on the page for '{name}'.", "name": name, "url": url}
    return {"name": name, "t_min": min(temps), "t_max": max(temps), "url": url}
