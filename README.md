# Hotel Search

A web application that searches Google Maps for hotels, scrapes details, and generates shareable reports via GitHub.

## Architecture

```
User Input → DeepSeek (URL generation) → Browser-Use (Google Maps scraping) → Report
                                                                                  ↓
                                                                          GitHub Pages
```

- **DeepSeek** converts natural language queries into Google Maps search URLs
- **Browser-Use** (headless Playwright) scrapes hotel listings and details
- **FastAPI** serves the web UI with real-time progress polling
- **GitHub API** saves reports as static HTML (viewable via htmlpreview.github.io)

## Features

- Natural language location input (any language)
- Filters: check-in date, number of nights, hotel count
- Sorting: distance, price, rating, star class
- Multi-select hotels → detailed report with images, address, website
- Save reports to GitHub as self-contained HTML files

## Setup

```bash
cd ~/.config/opencode/browser-use-project
uv sync
```

## Usage

### Web App

```bash
uv run uvicorn hotel_search_web:app --reload --port 8000
```

Open `http://localhost:8000` in a browser.

### CLI

```bash
uv run python hotel_search.py "Yokohama Station" --date 2026-07-04 --nights 2 --count 10 --sort distance
```

## How It Works

1. **Search**: DeepSeek generates a `google.com/maps/search/hotels+near+{location}` URL with date parameters
2. **Scrape**: Browser-Use opens Google Maps, scrolls the results sidebar, extracts hotel data (name, price, rating, star class, distance, features)
3. **Report**: Selected hotels are re-scraped for detailed info (address, website, hero image) via individual Google Maps detail pages
4. **Save**: The report is rendered as a static HTML file and pushed to GitHub, accessible via htmlpreview.github.io

## Files

| File | Description |
|------|-------------|
| `hotel_search_web.py` | FastAPI web app with search, report, and GitHub save endpoints |
| `hotel_search.py` | CLI version with argparse |
| `templates/index.html` | Search form page |
| `templates/result.html` | Results page with checkboxes and "Generate Report" button |
| `templates/report.html` | Report page with polling, hotel cards, and GitHub save |
| `pyproject.toml` | Dependencies (browser-use, fastapi, jinja2, etc.) |

## Dependencies

- `browser-use` — headless browser automation
- `fastapi` / `uvicorn` — web framework
- `jinja2` — HTML templates
- `openai` — DeepSeek API client
- `httpx` — GitHub API client
