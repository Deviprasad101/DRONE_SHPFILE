# Drone Navigation — React Frontend

3D map viewer for building footprints (GeoJSON) and animated drone flight paths.

## Stack

- **React 18** + **TypeScript** + **Vite**
- **MapLibre GL** (base map, no API key)
- **deck.gl** (3D extruded buildings, paths, drone marker)

## Prerequisites

- Node.js 18+
- Python backend running on port 8000 (GeoJSON + path planning API)

## Run

**Terminal 1 — API backend:**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Terminal 2 — React app:**

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

## Features

- 3D extruded buildings from `data/buildings.geojson` (loaded by viewport)
- Set **Start** (blue) and **Goal** (green) by clicking the map
- **Start Demo** — A* path via `/api/plan-path` that avoids building footprints
- **Reset** — clear flight and return drone to start
- Start (blue), goal (green), drone (cyan), flight path (yellow), 3D buildings (gray)

## Build for production

```bash
npm run build
```

The backend serves `frontend/dist/` at `http://localhost:8000` when built.
