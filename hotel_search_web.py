"""
Hotel Search Web Application.
FastAPI backend with real-time progress via SSE.

Usage:
    cd ~/.config/opencode/browser-use-project
    uv run uvicorn hotel_search_web:app --reload --port 8000
"""

import asyncio
import base64
import json
import os
import re
import uuid
from datetime import datetime, timedelta

import httpx
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse
from openai import OpenAI
from browser_use import Agent, Browser, ChatBrowserUse

DEEPSEEK_KEY = os.getenv("DEEPSEEK_KEY", "")
BROWSER_USE_KEY = os.getenv("BROWSER_USE_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

URL_PROMPT = """You are a URL generator. Given hotel search parameters, output ONLY a Google Maps search URL.

Format: https://www.google.com/maps/search/hotels+near+{english_location}/@{lat},{lng},15z/data=!3m1!4b1!4m7!2m6!5m4!5m3!1s{date}!4m1!1i{adults}!6e3

Rules:
- IMPORTANT: Translate the location to English (e.g. "横滨车站" → "Yokohama Station").
- The query MUST be "hotels near {english_location}" in English so Google Maps shows distance info.
- URL-encode the location (use + for spaces).
- Use accurate coordinates for the location.
- Output ONLY the URL, no explanation or markdown.
- Example: https://www.google.com/maps/search/hotels+near+Shinjuku+Station+Tokyo/@35.6963,139.6997,15z/data=!3m1!4b1!4m7!2m6!5m4!5m3!1s2026-07-01!4m1!1i2!6e3"""

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# In-memory task store: { task_id: { status, progress, result, error } }
tasks: dict[str, dict] = {}
task_events: dict[str, asyncio.Event] = {}


def extract_json(text: str):
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
    s = str(d)
    m = re.search(r'([\d.]+)\s*km', s)
    if m:
        return float(m.group(1)) * 1000
    m = re.search(r'(\d+)\s*m', s)
    if m:
        return float(m.group(1))
    return 999999


def parse_price(p) -> float:
    s = str(p).replace(',', '')
    m = re.search(r'([\d.]+)', s)
    return float(m.group(1)) if m else 0


async def run_search(task_id: str, location: str, check_in: str, nights: int, count: int, sort: str):
    """Run the hotel search as a background task. Returns whatever hotels are found."""
    check_out = (datetime.strptime(check_in, "%Y-%m-%d") + timedelta(days=nights)).strftime("%Y-%m-%d")
    maps_url = ""
    found_hotels: list[dict] = []

    def update(status: str, progress: str):
        tasks[task_id] = {"status": status, "progress": progress, "result": None, "error": None}
        if task_id in task_events:
            task_events[task_id].set()

    def finish_with_hotels(hotels: list[dict], partial: bool = False):
        """Return results even if partial."""
        if not hotels:
            update("error", "No hotels found.")
            return

        if sort == "distance":
            hotels.sort(key=lambda h: parse_distance(h["distance"]))
        elif sort == "price":
            hotels.sort(key=lambda h: h["price"], reverse=True)
        elif sort == "star":
            hotels.sort(key=lambda h: h.get("star_class", ""), reverse=True)
        elif sort == "rating":
            hotels.sort(key=lambda h: h["rating"], reverse=True)

        hotels = hotels[:count]
        msg = f"Found {len(hotels)} hotels." + (" (partial results)" if partial else "")
        tasks[task_id] = {
            "status": "done",
            "progress": msg,
            "result": {
                "location": location,
                "check_in": check_in,
                "check_out": check_out,
                "nights": nights,
                "count": count,
                "sort": sort,
                "maps_url": maps_url,
                "hotels": hotels,
                "partial": partial,
            },
            "error": None,
        }
        if task_id in task_events:
            task_events[task_id].set()

    def normalize_hotels(raw_list: list[dict]) -> list[dict]:
        seen = set()
        result = []
        for h in raw_list:
            name = h.get("name", "")
            if name and name not in seen:
                seen.add(name)
                result.append({
                    "name": name,
                    "price": parse_price(h.get("price", 0)),
                    "currency": h.get("currency", ""),
                    "rating": float(h.get("rating", 0)),
                    "star_class": h.get("star_class", h.get("class", "")) or "Unrated",
                    "distance": h.get("distance", "?"),
                    "features": h.get("features", []),
                })
        return result

    try:
        # Step 1: Generate URL
        update("running", "Generating Google Maps URL with DeepSeek...")
        client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": URL_PROMPT},
                {"role": "user", "content": f"Location: {location}, Date: {check_in}, Adults: 2"},
            ],
            temperature=0,
        )
        maps_url = response.choices[0].message.content.strip()

        if not maps_url.startswith("https://www.google.com/maps"):
            update("error", f"Invalid URL generated: {maps_url}")
            return

        update("running", f"URL generated. Opening browser to search hotels...")

        # Step 2: Browser-use scraping
        llm = ChatBrowserUse(model="bu-2-0", api_key=BROWSER_USE_KEY)
        browser = Browser(headless=True)

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

        update("running", "Browser is searching Google Maps... This may take 1-2 minutes.")
        try:
            result = await agent.run()
            raw = result.final_result()
        except Exception as agent_err:
            raw = None
            update("running", f"Browser error ({agent_err}). Returning any results found so far...")
        finally:
            try:
                await browser.close()
            except Exception:
                pass

        # Step 3: Parse results
        update("running", "Parsing results...")
        if raw:
            hotels = extract_json(raw)
            if hotels:
                found_hotels = normalize_hotels(hotels)

        if found_hotels:
            finish_with_hotels(found_hotels)
        else:
            update("error", "No hotel data could be extracted from the search results.")

    except Exception as e:
        # If we already have some hotels, return them instead of failing
        if found_hotels:
            finish_with_hotels(found_hotels, partial=True)
        else:
            update("error", str(e))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/search")
async def search(
    location: str = Form(...),
    date: str = Form("2026-07-04"),
    nights: int = Form(2),
    count: int = Form(10),
    sort: str = Form("distance"),
):
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {"status": "starting", "progress": "Starting search...", "result": None, "error": None}
    task_events[task_id] = asyncio.Event()

    asyncio.create_task(run_search(task_id, location, date, nights, count, sort))
    return RedirectResponse(url=f"/result/{task_id}", status_code=303)


@app.get("/result/{task_id}", response_class=HTMLResponse)
async def result_page(request: Request, task_id: str):
    if task_id not in tasks:
        return HTMLResponse("Task not found", status_code=404)
    return templates.TemplateResponse(request, "result.html", {"task_id": task_id})


@app.get("/stream/{task_id}")
async def stream(request: Request, task_id: str):
    """SSE endpoint for real-time progress updates."""
    if task_id not in tasks:
        return HTMLResponse("Task not found", status_code=404)

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break

            task = tasks.get(task_id)
            if not task:
                break

            yield {
                "event": "progress",
                "data": json.dumps({
                    "status": task["status"],
                    "progress": task["progress"],
                }),
            }

            if task["status"] in ("done", "error"):
                if task["status"] == "done":
                    yield {
                        "event": "done",
                        "data": json.dumps(task["result"]),
                    }
                else:
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": task["error"]}),
                    }
                break

            # Wait for update or timeout
            event = task_events.get(task_id)
            if event:
                event.clear()
                try:
                    await asyncio.wait_for(event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(1)

    return EventSourceResponse(event_generator())


# ── Report Generation ──────────────────────────────────────────────

async def scrape_hotel_detail(hotel_name: str, llm) -> dict:
    """Scrape a single hotel's details from Google Maps."""
    browser = Browser(headless=True)
    try:
        task = f"""
        1. Go to https://www.google.com/maps/search/{hotel_name.replace(' ', '+')}
        2. Wait for results. Click on the first hotel result to open its detail page.
        3. On the hotel detail page, extract:
           - address: full street address
           - website: official hotel website URL (not Google/Booking/Agoda)
           - image: URL of the main photo shown on the page (look for the large hero image)
        4. Return a JSON object with these fields. Use null for any field not found.
           Return ONLY the JSON object, no markdown.
           Example: {{"address":"123 Main St","website":"https://hotel.com","image":"https://..."}}
        """

        agent = Agent(
            task=task, llm=llm, browser=browser,
            use_vision=True, llm_screenshot_size=(800, 600),
            max_clickable_elements_length=3000, message_compaction=True,
        )

        result = await agent.run()
        raw = result.final_result() or ""

        # Parse JSON
        raw = raw.replace('\\"', '"')
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return {}
    except Exception:
        return {}
    finally:
        try:
            await browser.close()
        except Exception:
            pass


async def run_report(task_id: str, hotels: list[dict]):
    """Generate a detailed report for selected hotels."""
    total = len(hotels)
    results = []

    def update(status: str, progress: str, current: int = 0):
        tasks[task_id] = {
            "status": status,
            "progress": progress,
            "result": None,
            "error": None,
            "current": current,
            "total": total,
        }
        if task_id in task_events:
            task_events[task_id].set()

    try:
        llm = ChatBrowserUse(model="bu-2-0", api_key=BROWSER_USE_KEY)

        for i, hotel in enumerate(hotels):
            name = hotel.get("name", "Unknown")
            update("running", f"[{i+1}/{total}] Scraping: {name}...", i)

            detail = await scrape_hotel_detail(name, llm)

            # Merge search data with scraped detail
            hotel_info = {
                **hotel,
                "address": detail.get("address"),
                "website": detail.get("website"),
                "image": detail.get("image"),
            }
            results.append(hotel_info)

        # Done
        tasks[task_id] = {
            "status": "done",
            "progress": f"Report complete! {total} hotels processed.",
            "result": {
                "hotels": results,
                "location": hotels[0].get("location", ""),
                "check_in": hotels[0].get("check_in", ""),
                "check_out": hotels[0].get("check_out", ""),
            },
            "error": None,
        }
        if task_id in task_events:
            task_events[task_id].set()

    except Exception as e:
        # Return whatever we have
        if results:
            tasks[task_id] = {
                "status": "done",
                "progress": f"Partial report: {len(results)}/{total} hotels.",
                "result": {
                    "hotels": results,
                    "location": "",
                    "check_in": "",
                    "check_out": "",
                },
                "error": None,
            }
        else:
            tasks[task_id] = {
                "status": "error",
                "progress": str(e),
                "result": None,
                "error": str(e),
            }
        if task_id in task_events:
            task_events[task_id].set()


@app.post("/report")
async def create_report(hotels: str = Form(...)):
    task_id = str(uuid.uuid4())[:8]
    hotel_list = json.loads(hotels)
    tasks[task_id] = {"status": "starting", "progress": "Starting report...", "result": None, "error": None}
    task_events[task_id] = asyncio.Event()

    asyncio.create_task(run_report(task_id, hotel_list))
    return RedirectResponse(url=f"/report/{task_id}", status_code=303)


@app.get("/report/{task_id}", response_class=HTMLResponse)
async def report_page(request: Request, task_id: str):
    if task_id not in tasks:
        return HTMLResponse("Task not found", status_code=404)
    return templates.TemplateResponse(request, "report.html", {"task_id": task_id})


@app.get("/report-status/{task_id}")
async def report_status(task_id: str):
    """Polling endpoint for report progress. Returns JSON."""
    if task_id not in tasks:
        return {"status": "not_found"}
    task = tasks[task_id]
    return {
        "status": task["status"],
        "progress": task["progress"],
        "current": task.get("current", 0),
        "total": task.get("total", 0),
        "result": task.get("result"),
        "error": task.get("error"),
    }


@app.get("/report-stream/{task_id}")
async def report_stream(request: Request, task_id: str):
    """SSE endpoint for report generation progress."""
    if task_id not in tasks:
        return HTMLResponse("Task not found", status_code=404)

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break

            task = tasks.get(task_id)
            if not task:
                await asyncio.sleep(1)
                continue

            yield {
                "event": "progress",
                "data": json.dumps({
                    "status": task["status"],
                    "progress": task["progress"],
                    "current": task.get("current", 0),
                    "total": task.get("total", 0),
                }),
            }

            if task["status"] == "done":
                yield {
                    "event": "done",
                    "data": json.dumps(task["result"]),
                }
                break
            elif task["status"] == "error":
                yield {
                    "event": "error",
                    "data": json.dumps({"error": task["error"]}),
                }
                break

            await asyncio.sleep(2)

    return EventSourceResponse(event_generator())


# ── GitHub Save ───────────────────────────────────────────────────

def build_static_report(hotels: list[dict], location: str, check_in: str, check_out: str) -> str:
    """Generate a self-contained HTML report."""
    cards = []
    for i, h in enumerate(hotels):
        img = h.get("image") or ""
        img_html = (
            f'<img src="{img}" style="width:100%;height:250px;object-fit:cover;border-radius:8px 8px 0 0" alt="{h["name"]}">'
            if img
            else '<div style="width:100%;height:250px;background:#e9ecef;display:flex;align-items:center;justify-content:center;color:#6c757d;border-radius:8px 8px 0 0">No image</div>'
        )
        features = " ".join(
            f'<span style="display:inline-block;padding:2px 8px;margin:2px;background:#0d6efd;color:#fff;border-radius:4px;font-size:12px">{f}</span>'
            for f in (h.get("features") or [])
        )
        website = h.get("website")
        website_btn = (
            f'<a href="{website}" target="_blank" style="display:inline-block;padding:6px 12px;margin-top:8px;border:1px solid #0d6efd;color:#0d6efd;border-radius:4px;text-decoration:none;font-size:14px">Visit Website</a>'
            if website and website != "N/A"
            else ""
        )
        cards.append(f"""<div style="border:none;box-shadow:0 2px 12px rgba(0,0,0,0.08);margin-bottom:2rem;border-radius:8px;overflow:hidden;background:#fff">
            {img_html}
            <div style="padding:20px">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
                    <h3 style="margin:0">{i + 1}. {h["name"]}</h3>
                    <span style="background:#ffc107;color:#333;padding:4px 12px;border-radius:4px;font-weight:bold">{h.get("rating", "?")} / 5</span>
                </div>
                <p style="color:#666;margin-bottom:8px">{h.get("star_class", "")} | {h.get("distance", "?")} from center</p>
                <h4 style="color:#0d6efd;margin-bottom:12px">{h.get("currency", "$")} {round(h.get("price", 0))} / night</h4>
                {f'<p style="margin-bottom:4px"><strong>Address:</strong> {h.get("address", "")}</p>' if h.get("address") else ""}
                {website_btn}
                {f'<div style="margin-top:12px">{features}</div>' if features else ""}
            </div>
        </div>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>Hotel Report</title>
<style>
body{{background:#f8f9fa;font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;padding:20px}}
.container{{max-width:960px;margin:0 auto}}
h1{{margin-bottom:4px}}h1+p{{color:#666;margin-top:0}}
</style></head>
<body><div class="container">
<h1>Hotel Report</h1>
<p>{location} | {check_in} ~ {check_out} | {len(hotels)} hotels</p>
{"".join(cards)}
<p style="text-align:center;color:#999;margin-top:2rem">Generated by Hotel Search</p>
</div></body></html>"""


async def save_to_github(repo: str, hotels: list[dict], location: str, check_in: str, check_out: str) -> str:
    """Save report to GitHub repo, return the file URL."""
    html = build_static_report(hotels, location, check_in, check_out)
    content_b64 = base64.b64encode(html.encode()).decode()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = f"reports/{location.replace(' ', '-')}-{timestamp}.html"

    url = f"https://api.github.com/repos/{repo}/contents/{filepath}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    body = {
        "message": f"Add hotel report: {location} ({check_in} ~ {check_out})",
        "content": content_b64,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.put(url, json=body, headers=headers)
        if resp.status_code in (201, 200):
            data = resp.json()
            html_url = data.get("content", {}).get("html_url", "")
            return html_url.replace("https://github.com/", "https://htmlpreview.github.io/?")
        else:
            raise Exception(f"GitHub API error ({resp.status_code}): {resp.text[:200]}")


@app.post("/save-to-github")
async def save_to_github_endpoint(
    repo: str = Form(...),
    hotels: str = Form(...),
    location: str = Form(""),
    check_in: str = Form(""),
    check_out: str = Form(""),
):
    try:
        hotel_list = json.loads(hotels)
        preview_url = await save_to_github(repo, hotel_list, location, check_in, check_out)
        return {"ok": True, "url": preview_url}
    except Exception as e:
        return {"ok": False, "error": str(e)}
