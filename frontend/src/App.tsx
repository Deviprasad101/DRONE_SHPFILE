import { useCallback, useEffect, useState } from "react";
import DroneMap from "./components/DroneMap";
import {
  fetchAllBuildingsInArea,
  fetchBounds,
  fetchDemoFlight,
  flightCenter,
} from "./api/client";
import { useDroneAnimation } from "./hooks/useDroneAnimation";
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
  const [flight, setFlight] = useState<FlightPath | null>(null);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("Idle");
  const [steps, setSteps] = useState(0);
  const [distance, setDistance] = useState<number | null>(null);
  const [reward, setReward] = useState(0);
  const [playId, setPlayId] = useState(0);
  const [viewState, setViewState] = useState({
    longitude: 80.2292,
    latitude: 12.9982,
    zoom: 15.5,
    pitch: 60,
    bearing: -25,
  });

  const trajectory = flight?.trajectory ?? null;
  const plannedPath = flight?.planned_path ?? null;
  const start = flight?.start ?? null;
  const goal = flight?.goal ?? null;

  const handleFlightComplete = useCallback(() => {
    setPlaying(false);
    setStatus("Goal reached");
    if (flight) {
      setReward(100 - flight.trajectory.length);
      setDistance(0);
    }
  }, [flight]);

  const { position: dronePosition, stepIndex, finished, reset: resetDrone } =
    useDroneAnimation(trajectory, 1, playing, playId, handleFlightComplete);

  const loadBuildingsForFlight = useCallback(async (f: FlightPath) => {
    const lons = f.trajectory.map((p) => p[0]);
    const lats = f.trajectory.map((p) => p[1]);
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
  }, []);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const bounds = await fetchBounds();
        const cx = (bounds.min_lon + bounds.max_lon) / 2;
        const cy = (bounds.min_lat + bounds.max_lat) / 2;
        setViewState((vs) => ({
          ...vs,
          longitude: cx,
          latitude: cy,
        }));

        const demo = await fetchDemoFlight();
        setFlight(demo);
        const center = flightCenter(demo.trajectory);
        setViewState((vs) => ({
          ...vs,
          longitude: center.longitude,
          latitude: center.latitude,
          zoom: 16,
          pitch: 60,
        }));

        await loadBuildingsForFlight(demo);
        setStatus("Ready — press Start Demo");
      } catch {
        setStatus("Could not load buildings — start backend on :8000");
      } finally {
        setLoading(false);
      }
    })();
  }, [loadBuildingsForFlight]);

  useEffect(() => {
    if (!dronePosition || !goal) return;
    setSteps(stepIndex);
    setDistance(dist3(dronePosition, goal));
    if (playing && !finished) setStatus("Flying");
  }, [dronePosition, goal, stepIndex, playing, finished]);

  const startDemo = useCallback(async () => {
    let activeFlight = flight;
    if (!activeFlight) {
      try {
        activeFlight = await fetchDemoFlight();
        setFlight(activeFlight);
        await loadBuildingsForFlight(activeFlight);
      } catch {
        setStatus("Failed to load demo flight");
        return;
      }
    }
    setSteps(0);
    setReward(0);
    setDistance(dist3(activeFlight.start, activeFlight.goal));
    setStatus("Flying");
    setPlayId((id) => id + 1);
    setPlaying(true);
  }, [flight, loadBuildingsForFlight]);

  const reset = useCallback(() => {
    setPlaying(false);
    resetDrone();
    setSteps(0);
    setDistance(flight ? dist3(flight.start, flight.goal) : null);
    setStatus("Idle");
    setReward(0);
  }, [flight, resetDrone]);

  // Keep map centered on drone while flying
  useEffect(() => {
    if (!playing || !dronePosition) return;
    setViewState((vs) => ({
      ...vs,
      longitude: dronePosition[0],
      latitude: dronePosition[1],
    }));
  }, [playing, dronePosition]);

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
            disabled={playing || loading}
          >
            Start Demo
          </button>
          <button type="button" onClick={reset} disabled={loading}>
            Reset
          </button>
        </div>
      </header>

      <section className="sim-section">
        <h2>3D ANIMATION (SIMULATION)</h2>
        <p className="hint">
          {loading
            ? "Loading buildings.geojson into 3D view…"
            : `Showing ${visibleCount.toLocaleString()} of ${totalBuildings.toLocaleString()} buildings in 3D over the base map. Press Start Demo to fly the drone.`}
        </p>
        <div className="sim-canvas">
          <DroneMap
            buildings={buildings}
            plannedPath={plannedPath}
            trajectory={trajectory}
            dronePosition={dronePosition}
            start={start}
            goal={goal}
            viewState={viewState}
            onMove={setViewState}
          />
        </div>
        <div className="stats-row">
          <span>Steps: <strong>{steps}</strong></span>
          <span>
            Distance to Goal:{" "}
            <strong>{distance !== null ? distance.toFixed(5) : "—"}</strong>
          </span>
          <span>Status: <strong>{status}</strong></span>
          <span>Reward: <strong>{reward.toFixed(1)}</strong></span>
        </div>
      </section>

      <section className="route-section">
        <h2>Flight Route</h2>
        <p>
          Demo path is planned over real building footprints from{" "}
          <code>data/buildings.geojson</code>. Buildings are extruded to their{" "}
          <code>height_m</code> in 3D on top of a street base map.
        </p>
        {start && goal && (
          <p className="coords">
            Start{" "}
            <strong>
              {fmtCoord(start[1])}, {fmtCoord(start[0])}
            </strong>{" "}
            &nbsp;|&nbsp; Goal{" "}
            <strong>
              {fmtCoord(goal[1])}, {fmtCoord(goal[0])}
            </strong>{" "}
            &nbsp;|&nbsp; Altitude{" "}
            <strong>{start[2]?.toFixed(0) ?? 80} m</strong>
          </p>
        )}
        <p className="hint">Press Start Demo to launch the RL agent drone through the city.</p>
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
