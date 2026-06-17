"""FastAPI backend for React drone viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
GEOJSON_PATH = ROOT / "data" / "buildings.geojson"
EVAL_PATH = ROOT / "logs" / "eval" / "eval_results.json"
FRONTEND_DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="Drone Navigation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve built React app (production)
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/")
def serve_frontend():
    """Serve React app at root (run `npm run build` in frontend/ first)."""
    index = FRONTEND_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse(
        "<h1>Frontend not built</h1><p>Run: cd frontend && npm.cmd install && npm.cmd run build</p>"
        "<p>Or dev mode: npm.cmd run dev (port 5173)</p>",
        status_code=200,
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/buildings")
def get_buildings(
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    min_lon: float | None = None,
    min_lat: float | None = None,
    max_lon: float | None = None,
    max_lat: float | None = None,
) -> dict[str, Any]:
    """Return building footprints as GeoJSON (paginated / bbox filtered)."""
    with GEOJSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    if min_lon is not None and min_lat is not None and max_lon is not None and max_lat is not None:
        filtered = []
        for feat in features:
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [])
            if not coords:
                continue
            ring = coords[0] if geom.get("type") == "Polygon" else coords[0][0]
            lons = [c[0] for c in ring]
            lats = [c[1] for c in ring]
            if max(lons) >= min_lon and min(lons) <= max_lon and max(lats) >= min_lat and min(lats) <= max_lat:
                filtered.append(feat)
        features = filtered

    total = len(features)
    page = features[offset : offset + limit]
    return {
        "type": "FeatureCollection",
        "features": page,
        "meta": {"total": total, "limit": limit, "offset": offset},
    }


@app.get("/api/buildings/bounds")
def get_bounds() -> dict[str, float]:
    """Approximate bounding box of all buildings."""
    with GEOJSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    min_lon = min_lat = float("inf")
    max_lon = max_lat = float("-inf")
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if not coords:
            continue
        rings: list = []
        if gtype == "Polygon":
            rings = [coords[0]]
        elif gtype == "MultiPolygon":
            rings = [poly[0] for poly in coords]
        for ring in rings:
            for lon, lat, *_ in ring:
                min_lon, max_lon = min(min_lon, lon), max(max_lon, lon)
                min_lat, max_lat = min(min_lat, lat), max(max_lat, lat)
    return {"min_lon": min_lon, "min_lat": min_lat, "max_lon": max_lon, "max_lat": max_lat}


@app.get("/api/trajectories")
def get_trajectories() -> dict[str, Any]:
    """Return saved evaluation trajectories if available."""
    if not EVAL_PATH.exists():
        return {"episodes": [], "summary": {}}
    with EVAL_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/demo-flight")
def demo_flight() -> dict[str, Any]:
    """Generate a demo flight path weaving through the building area."""
    bounds = get_bounds()
    cx = (bounds["min_lon"] + bounds["max_lon"]) / 2
    cy = (bounds["min_lat"] + bounds["max_lat"]) / 2
    alt = 85.0
    d = 0.005
    trajectory = [
        [cx - d, cy - d, alt],
        [cx - d * 0.5, cy - d * 0.25, alt + 8],
        [cx, cy - d * 0.1, alt + 15],
        [cx + d * 0.4, cy + d * 0.3, alt + 10],
        [cx + d * 0.8, cy + d * 0.6, alt + 5],
        [cx + d, cy + d, alt],
    ]
    return {
        "start": trajectory[0],
        "goal": trajectory[-1],
        "planned_path": trajectory,
        "trajectory": trajectory,
        "name": "GeoJSON demo flight",
    }
