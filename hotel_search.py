"""
Hotel search via Google Maps.
DeepSeek generates the URL, browser-use scrapes results.

Usage:
    cd ~/.config/opencode/browser-use-project
    uv run python hotel_search.py "横滨车站" --date 2026-07-04 --nights 2 --count 10 --sort price
"""

import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timedelta

from openai import OpenAI
from browser_use import Agent, Browser, ChatBrowserUse

DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY", "")
BROWSER_USE_KEY = os.getenv("BROWSER_USE_KEY", "")

URL_PROMPT = """You are a URL generator. Given hotel search parameters, output ONLY a Google Maps search URL.

Format: https://www.google.com/maps/search/hotels+near+{english_location}/@{lat},{lng},15z/data=!3m1!4b1!4m7!2m6!5m4!5m3!1s{date}!4m1!1i{adults}!6e3

Rules:
- IMPORTANT: Translate the location to English (e.g. "横滨车站" → "Yokohama Station", "东京站" → "Tokyo Station").
- The query MUST be "hotels near {english_location}" in English so Google Maps shows distance info.
- URL-encode the location (use + for spaces).
- Use accurate coordinates for the location.
- Output ONLY the URL, no explanation or markdown.
- Example: https://www.google.com/maps/search/hotels+near+Shinjuku+Station+Tokyo/@35.6963,139.6997,15z/data=!3m1!4b1!4m7!2m6!5m4!5m3!1s2026-07-01!4m1!1i2!6e3"""


def extract_json(text: str):
    """Extract JSON array from text that may be wrapped in markdown or escaped."""
    text = text.replace('\\"', '"')
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\[[\s\S]*\]', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def parse_distance(d) -> float:
    """Convert distance string like '0.3 km' or '150 m' to meters."""
    s = str(d)
    m = re.search(r'([\d.]+)\s*km', s)
    if m:
        return float(m.group(1)) * 1000
    m = re.search(r'(\d+)\s*m', s)
    if m:
        return float(m.group(1))
    return 999999


def parse_price(p) -> float:
    """Extract numeric price from string like '$120' or '¥15,000'."""
    s = str(p).replace(',', '')
    m = re.search(r'([\d.]+)', s)
    return float(m.group(1)) if m else 0


async def main():
    parser = argparse.ArgumentParser(description="Search hotels near a location via Google Maps")
    parser.add_argument("location", help="Address or landmark (e.g. '横滨车站')")
    parser.add_argument("--date", default="2026-07-04", help="Check-in date YYYY-MM-DD (default: 2026-07-04)")
    parser.add_argument("--nights", type=int, default=2, help="Number of nights (default: 2)")
    parser.add_argument("--count", type=int, default=10, help="Number of hotels to return (default: 10)")
    parser.add_argument("--sort", choices=["distance", "price", "star", "rating"], default="distance",
                        help="Sort by: distance (near-far), price (high-low), star (high-low), rating (high-low)")
    args = parser.parse_args()

    check_in = args.date
    check_out = (datetime.strptime(check_in, "%Y-%m-%d") + timedelta(days=args.nights)).strftime("%Y-%m-%d")
    location = args.location

    print(f"Location:  {location}")
    print(f"Check-in:  {check_in}")
    print(f"Check-out: {check_out} ({args.nights} nights)")
    print(f"Hotels:    {args.count}")
    print(f"Sort:      {args.sort}")
    print()

    # Step 1: Generate Google Maps URL with DeepSeek
    print("Generating Google Maps URL...")
    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": URL_PROMPT},
            {"role": "user", "content": f"Location: {location}, Date: {check_in}, Adults: 2"},
        ],
        temperature=0,
    )
    maps_url = response.choices[0].message.content.strip()

    if not maps_url.startswith("https://www.google.com/maps"):
        print(f"Invalid URL: {maps_url}")
        return

    print(f"URL: {maps_url}\n")

    # Step 2: Browser-use scrapes Google Maps
    llm = ChatBrowserUse(model="bu-2-0", api_key=BROWSER_USE_KEY)
    browser = Browser()

    task = f"""
    1. Go to {maps_url}
    2. Wait for hotel results to load in the sidebar.
    3. If there is a date picker or "Check-in" option on the page, set:
       - Check-in: {check_in}
       - Check-out: {check_out}
       If no date picker is visible, skip this step.
    4. Scroll the sidebar to load more hotels.
    5. Use the EXTRACT tool ONCE to get all visible hotel data. Extract for each hotel:
       - name, price (number), currency, rating (number), star_class, distance, features (array)
       - If a field is not visible, use null. Do NOT retry for missing fields.
    6. Return the extracted data as a raw JSON array immediately. Do NOT use evaluate scripts.
       Start with [, end with ].
    """

    agent = Agent(
        task=task,
        llm=llm,
        browser=browser,
        use_vision=True,
        llm_screenshot_size=(800, 600),
        max_clickable_elements_length=3000,
        message_compaction=True,
    )

    print("Opening browser and scraping hotels...")
    result = await agent.run()
    await browser.close()

    # Step 3: Parse and sort results
    raw = result.final_result()
    hotels = extract_json(raw) if raw else None

    if not hotels:
        print("\nFailed to extract hotel data.")
        print(f"Raw output: {raw[:500] if raw else 'None'}")
        return

    # Deduplicate by name
    seen = set()
    unique = []
    for h in hotels:
        name = h.get("name", "")
        if name and name not in seen:
            seen.add(name)
            unique.append({
                "name": name,
                "price": parse_price(h.get("price", 0)),
                "currency": h.get("currency", ""),
                "rating": float(h.get("rating", 0)),
                "star_class": h.get("star_class", h.get("class", "")),
                "distance": h.get("distance", "?"),
                "features": h.get("features", []),
            })

    # Sort
    sort_key = args.sort
    if sort_key == "distance":
        unique.sort(key=lambda h: parse_distance(h["distance"]))
    elif sort_key == "price":
        unique.sort(key=lambda h: h["price"], reverse=True)
    elif sort_key == "star":
        unique.sort(key=lambda h: h.get("star_class", ""), reverse=True)
    elif sort_key == "rating":
        unique.sort(key=lambda h: h["rating"], reverse=True)

    # Trim to requested count
    unique = unique[: args.count]

    # Display
    print(f"\n{'=' * 90}")
    print(f"  Hotels near '{location}' | {check_in} ~ {check_out} | Sorted by {sort_key}")
    print(f"{'=' * 90}")
    print(f"{'#':<4}{'Hotel':<32}{'Price':>10}{'Rating':>8}{'Star':>14}{'Distance':>12}  Features")
    print(f"{'-' * 90}")
    for i, h in enumerate(unique, 1):
        price_str = f"{h['currency']} {h['price']:,.0f}" if h['price'] else "?"
        features_str = ", ".join(h["features"][:3]) if h["features"] else "-"
        print(f"{i:<4}{h['name'][:30]:<32}{price_str:>10}{h['rating']:>7.1f}"
              f"{h['star_class']:>14}{h['distance']:>12}  {features_str}")
    print(f"{'=' * 90}")

    # Also output raw JSON
    print(f"\n--- JSON ---")
    print(json.dumps(unique, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
