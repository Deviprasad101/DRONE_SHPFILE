# Drone Navigation — React Frontend

3D map viewer for building footprints (GeoJSON) and animated drone flight paths.

## Stack

- **React 18** + **TypeScript** + **Vite**
- **MapLibre GL** (base map, no API key)
- **deck.gl** (3D extruded buildings, paths, drone marker)

## Prerequisites

- Node.js 18+
- Python backend running on port 8000 (serves GeoJSON API)

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
- Demo flight animation (Play / Pause / Reset)
- Load evaluation trajectories from `logs/eval/eval_results.json` (after running `evaluate.py`)
- Start (green), goal (red), drone (cyan), planned path (yellow), trajectory (blue)

## Build for production

```bash
npm run build
npm run preview
```

Set `VITE_API_URL` if the API is not proxied (update `src/api/client.ts`).
