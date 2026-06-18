import { useCallback, useEffect, useState } from "react";
import DroneMap from "./components/DroneMap";
import {
  fetchAllBuildingsInArea,
  fetchBounds,
  fetchPlannedPath,
  flightCenter,
} from "./api/client";
import { useDroneAnimation } from "./hooks/useDroneAnimation";
import { defaultRouteFromCenter, planPathBetween } from "./utils/flightPath";
import type { BuildingCollection, FlightPath } from "./types/geo";
import "./App.css";

function dist3(a: number[], b: number[]): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const dz = (b[2] ?? 0) - (a[2] ?? 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function fmtCoord(v: number) {
  return v.toFixed(5);
}

export default function App() {
  const [buildings, setBuildings] = useState<BuildingCollection | null>(null);
  const [totalBuildings, setTotalBuildings] = useState(0);
  const [startPoint, setStartPoint] = useState<number[] | null>(null);
  const [goalPoint, setGoalPoint] = useState<number[] | null>(null);
  const [flight, setFlight] = useState<FlightPath | null>(null);
  const [placementMode, setPlacementMode] = useState<"start" | "goal" | null>(null);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("Idle");
  const [steps, setSteps] = useState(0);
  const [distance, setDistance] = useState<number | null>(null);
  const [reward, setReward] = useState(0);
  const [stepReward, setStepReward] = useState(0);
  const [prevDistance, setPrevDistance] = useState<number | null>(null);
  const [playId, setPlayId] = useState(0);
  const [followDrone, setFollowDrone] = useState(true);
  const [viewState, setViewState] = useState({
    longitude: 80.2292,
    latitude: 12.9982,
    zoom: 15.5,
    pitch: 60,
    bearing: -25,
  });

  const trajectory = flight?.trajectory ?? null;
  const plannedPath = flight?.planned_path ?? null;

  const handleFlightComplete = useCallback(() => {
    setPlaying(false);
    setStatus("Goal reached ✓");
    if (goalPoint) {
      // +100 goal bonus on arrival
      const goalBonus = 100.0;
      setReward((r) => r + goalBonus);
      setStepReward(goalBonus);
      setDistance(0);
    }
  }, [goalPoint]);

  const { position: dronePosition, stepIndex, finished, reset: resetDrone } =
    useDroneAnimation(trajectory, 1, playing, playId, handleFlightComplete);

  const loadBuildingsForArea = useCallback(
    async (lons: number[], lats: number[]) => {
      const pad = 0.012;
      const bounds = {
        min_lon: Math.min(...lons) - pad,
        max_lon: Math.max(...lons) + pad,
        min_lat: Math.min(...lats) - pad,
        max_lat: Math.max(...lats) + pad,
      };
      const data = await fetchAllBuildingsInArea(bounds, 15000);
      setBuildings(data);
      setTotalBuildings(data.meta?.total ?? data.features.length);
    },
    []
  );

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const bounds = await fetchBounds();
        const cx = (bounds.min_lon + bounds.max_lon) / 2;
        const cy = (bounds.min_lat + bounds.max_lat) / 2;
        const { start, goal } = defaultRouteFromCenter(cx, cy);

        setStartPoint(start);
        setGoalPoint(goal);
        setViewState((vs) => ({
          ...vs,
          longitude: cx,
          latitude: cy,
          zoom: 16,
          pitch: 60,
        }));

        await loadBuildingsForArea(
          [start[0], goal[0], cx],
          [start[1], goal[1], cy]
        );
        setStatus("Set start & goal on map, then press Start Demo");
      } catch {
        setStatus("Could not load buildings — start backend on :8000");
      } finally {
        setLoading(false);
      }
    })();
  }, [loadBuildingsForArea]);

  const handleMapClick = useCallback(
    async (lon: number, lat: number) => {
      const mode = placementMode;
      if (!mode) return;

      const point = [lon, lat, startPoint?.[2] ?? goalPoint?.[2] ?? 85];

      if (mode === "start") {
        setStartPoint(point);
      } else {
        setGoalPoint(point);
      }

      setPlacementMode(null);
      setFlight(null);
      setPlaying(false);
      setSteps(0);
      setDistance(null);
      setReward(0);
      setStepReward(0);
      setPrevDistance(null);
      setStatus("Route updated — press Start Demo");

      const s = mode === "start" ? point : startPoint;
      const g = mode === "goal" ? point : goalPoint;
      if (s && g) {
        try {
          await loadBuildingsForArea(
            [s[0], g[0]],
            [s[1], g[1]]
          );
        } catch {
          /* keep existing buildings */
        }
      }
    },
    [placementMode, startPoint, goalPoint, loadBuildingsForArea]
  );

  useEffect(() => {
    const goal = goalPoint ?? flight?.goal;
    const pos = dronePosition ?? startPoint;
    if (!pos || !goal) return;
    setSteps(stepIndex);
    const newDist = dist3(pos, goal);
    setDistance(newDist);

    // Compute step reward when actively flying
    if (playing && !finished && stepIndex > 0) {
      setPrevDistance((prev) => {
        const progress = prev !== null ? prev - newDist : 0;
        // Mirror Python env reward function:
        const progressReward  = progress * 2.0;   // progress_scale = 2.0
        const timePenalty     = -0.05;              // time_penalty
        const energyPenalty   = -0.01;              // small energy penalty
        const sr = progressReward + timePenalty + energyPenalty;
        setStepReward(sr);
        setReward((r) => r + sr);
        return newDist;
      });
    }

    if (playing && !finished) setStatus("Flying");
  }, [dronePosition, goalPoint, flight?.goal, stepIndex, playing, finished, startPoint]);

  const startDemo = useCallback(async () => {
    if (!startPoint || !goalPoint) {
      setStatus("Set start and goal on the map first");
      return;
    }
    if (
      startPoint[0] === goalPoint[0] &&
      startPoint[1] === goalPoint[1]
    ) {
      setStatus("Start and goal must be different");
      return;
    }

    setStatus("Planning collision-free route…");

    let planned: FlightPath;
    try {
      planned = await fetchPlannedPath(startPoint, goalPoint);
    } catch {
      setStatus("Planner unavailable — using straight-line fallback");
      planned = planPathBetween(startPoint, goalPoint);
    }

    setFlight(planned);
    setPlaying(false);

    try {
      await loadBuildingsForArea(
        planned.trajectory.map((p) => p[0]),
        planned.trajectory.map((p) => p[1])
      );
      const center = flightCenter(planned.trajectory);
      setViewState((vs) => ({
        ...vs,
        longitude: center.longitude,
        latitude: center.latitude,
        zoom: Math.max(vs.zoom, 16),
      }));
    } catch {
      setStatus("Could not reload buildings for route");
      return;
    }

    setSteps(0);
    setReward(0);
    setStepReward(0);
    setPrevDistance(dist3(planned.start, planned.goal));
    setDistance(dist3(planned.start, planned.goal));
    setFollowDrone(true);
    setStatus("Flying");
    setPlayId((id) => id + 1);
    setPlaying(true);
  }, [startPoint, goalPoint, loadBuildingsForArea]);

  const reset = useCallback(() => {
    setPlaying(false);
    setFlight(null);
    resetDrone();
    setSteps(0);
    setDistance(
      startPoint && goalPoint ? dist3(startPoint, goalPoint) : null
    );
    setStatus("Idle");
    setReward(0);
    setStepReward(0);
    setPrevDistance(null);
  }, [startPoint, goalPoint, resetDrone]);

  useEffect(() => {
    if (!playing || !dronePosition || !followDrone) return;
    setViewState((vs) => ({
      ...vs,
      longitude: dronePosition[0],
      latitude: dronePosition[1],
    }));
  }, [playing, dronePosition, followDrone]);

  const displayDrone = playing || flight ? dronePosition : startPoint;
  const visibleCount = buildings?.features.length ?? 0;

  return (
    <div className="indoor-app">
      <header className="indoor-header">
        <h1>RL BASED INDOOR DRONE NAVIGATION</h1>
        <p>Autonomous Navigation with 3D Buildings from GeoJSON (Chennai)</p>
        <div className="header-actions">
          <button
            type="button"
            className="btn-primary"
            onClick={startDemo}
            disabled={playing || loading || !startPoint || !goalPoint}
          >
            Start Demo
          </button>
          <button type="button" onClick={reset} disabled={loading}>
            Reset
          </button>
          {playing && (
            <button
              type="button"
              className={followDrone ? "btn-follow active" : "btn-follow"}
              onClick={() => setFollowDrone((f) => !f)}
            >
              {followDrone ? "Following Drone ✓" : "Follow Drone"}
            </button>
          )}
        </div>
      </header>

      <section className="sim-section">
        <h2>3D ANIMATION (SIMULATION)</h2>
        <p className="hint">
          {loading
            ? "Loading buildings.geojson into 3D view…"
            : placementMode
              ? `Click the map to place ${placementMode === "start" ? "START (blue)" : "GOAL (green)"}`
              : `Showing ${visibleCount.toLocaleString()} of ${totalBuildings.toLocaleString()} buildings. Use Set Start / Set Goal, click the map, then Start Demo.`}
        </p>
        <div className="sim-canvas">
          <DroneMap
            buildings={buildings}
            plannedPath={plannedPath}
            trajectory={trajectory}
            dronePosition={displayDrone}
            start={startPoint}
            goal={goalPoint}
            viewState={viewState}
            onMove={(vs) => {
              setViewState(vs);
              // If user drags the map while flying, disable auto-follow
              if (playing) setFollowDrone(false);
            }}
            placementMode={placementMode}
            onMapClick={handleMapClick}
          />
        </div>
        <div className="stats-row">
          <span>Steps: <strong>{steps}</strong></span>
          <span>
            Distance to Goal:{" "}
            <strong>{distance !== null ? distance.toFixed(5) : "—"}</strong>
          </span>
          <span>Status: <strong>{status}</strong></span>
          <span>
            Step Reward:{" "}
            <strong style={{ color: stepReward >= 0 ? "#16a34a" : "#dc2626" }}>
              {stepReward >= 0 ? "+" : ""}{stepReward.toFixed(3)}
            </strong>
          </span>
          <span>
            Total Reward:{" "}
            <strong style={{ color: reward >= 0 ? "#16a34a" : "#dc2626" }}>
              {reward >= 0 ? "+" : ""}{reward.toFixed(2)}
            </strong>
          </span>
        </div>
      </section>

      <section className="route-section">
        <h2>Select Route</h2>
        <p>
          Click a button below, then click on the open area of the base map.
          The drone will fly an A* path that avoids building footprints between start and goal.
        </p>
        <div className="btn-row">
          <button
            type="button"
            className={placementMode === "start" ? "active" : ""}
            onClick={() => setPlacementMode("start")}
            disabled={loading || playing}
          >
            Set Start (Blue)
          </button>
          <button
            type="button"
            className={placementMode === "goal" ? "active" : ""}
            onClick={() => setPlacementMode("goal")}
            disabled={loading || playing}
          >
            Set Goal (Green)
          </button>
        </div>
        {startPoint && goalPoint && (
          <p className="coords">
            Start{" "}
            <strong>
              {fmtCoord(startPoint[1])}, {fmtCoord(startPoint[0])}
            </strong>{" "}
            &nbsp;|&nbsp; Goal{" "}
            <strong>
              {fmtCoord(goalPoint[1])}, {fmtCoord(goalPoint[0])}
            </strong>{" "}
            &nbsp;|&nbsp; Altitude{" "}
            <strong>{startPoint[2]?.toFixed(0) ?? 85} m</strong>
          </p>
        )}
        <p className="hint">Select start and goal on the map, then press Start Demo.</p>
      </section>

      <section className="legend indoor-legend">
        <span><i className="swatch blue" /> Start Position</span>
        <span><i className="swatch green" /> Goal Position</span>
        <span><i className="swatch yellow" /> Flight Path</span>
        <span><i className="swatch cyan" /> RL Agent (Drone)</span>
        <span><i className="swatch gray" /> 3D Buildings (GeoJSON)</span>
      </section>
    </div>
  );
}
