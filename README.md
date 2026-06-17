# Autonomous Drone Navigation with DreamerV3

Production-grade system that converts **GeoJSON building footprints** into a **3D navigable city**, trains a **DreamerV3** model-based RL agent, and flies a drone from any start to any goal while avoiding collisions.

## Architecture

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `data_loader/geojson_loader.py` | GeoPandas + PyProj WGS84 → local ENU |
| 2 | `occupancy/voxel_map.py` | 3D voxel grid + distance transform |
| 3 | `planner/astar.py` | A* with cost map + path smoothing |
| 4 | `simulation/simulator.py` | PyBullet / kinematic sim (footprint collision) |
| 5 | `env/drone_navigation_env.py` | Gymnasium RL environment |
| 6 | `rl/` | DreamerV3 (RSSM, actor, critic, replay) |
| 7–8 | `train.py` | Training + TensorBoard + checkpoints |
| 9 | `visualization/visualizer.py` | Interactive PyVista 3D viewer |

## Requirements

- Python 3.11+ (3.11 recommended on Windows for PyBullet wheels)
- `data/buildings.geojson` (included)

## Installation

```bash
cd "d:\TIH PROJECTS\DRONE_SHPFILE"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> Use `configs/dev.yaml` (`max_buildings: 200`) for fast iteration. Full dataset load can take several minutes.

## Training

```bash
# Development
python train.py --config configs/dev.yaml

# Full training
python train.py --config configs/default.yaml

# Resume
python train.py --config configs/dev.yaml --resume --checkpoint checkpoints/dreamer_step_100.pt
```

TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/dreamer_final.pt --config configs/dev.yaml --visualize

# Custom start/goal (local meters)
python evaluate.py --checkpoint checkpoints/dreamer_final.pt --config configs/dev.yaml \
  --start 100 200 40 --goal 500 600 40 --visualize
```

## Project Structure

```
├── data/buildings.geojson
├── data_loader/geojson_loader.py
├── simulation/simulator.py
├── occupancy/voxel_map.py
├── planner/astar.py
├── env/drone_navigation_env.py
├── rl/  (dreamer, rssm, world_model, actor, critic, replay_buffer)
├── configs/default.yaml, dev.yaml
├── visualization/visualizer.py
├── train.py
├── evaluate.py
└── tests/
```

## Observation & Action

**Observations:** position, velocity, goal, distance, obstacle distance, next waypoint, local occupancy (16×16), top-down image.

**Actions:** `[vx, vy, vz, yaw_rate]` (continuous velocity control).

## Metrics

Success rate, collision rate, average reward, path length, energy, flight time — logged to TensorBoard and `logs/metrics.json`.

## Tests

```bash
pytest tests/ -v
```

## React 3D Viewer (Frontend)

Interactive map to view buildings and drone flights:

```bash
# Terminal 1 — API
cd backend && pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2 — React
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 — see `frontend/README.md` for details.

## License

MIT
