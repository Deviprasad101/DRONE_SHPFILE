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
import type { BuildingCollection, FlightPath, PathResponse } from "./types/geo";
import ProgressUI from "./components/ProgressUI";
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
  const [flights, setFlights] = useState<FlightPath[] | null>(null);
  const [selectedFlightIndex, setSelectedFlightIndex] = useState(0);
  const [placementMode, setPlacementMode] = useState<"start" | "goal" | null>(null);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("Idle");
  const [steps, setSteps] = useState(0);
  const [distance, setDistance] = useState<number | null>(null);
  const [reward, setReward] = useState(0);
  const [stepReward, setStepReward] = useState(0);
  const [totalPenalty, setTotalPenalty] = useState(0);
  const [, setPrevDistance] = useState<number | null>(null);
  const [playId, setPlayId] = useState(0);
  const [followDrone, setFollowDrone] = useState(true);
  const [planningProgress, setPlanningProgress] = useState<number | null>(null);
  const [viewState, setViewState] = useState({
    longitude: 80.2292,
    latitude: 12.9982,
    zoom: 15.5,
    pitch: 60,
    bearing: -25,
  });

  const flight = flights ? flights[selectedFlightIndex] : null;
  const trajectory = flight?.trajectory ?? null;

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

        // Load the ENTIRE dataset bounds in a single request.
        // Using the full bounds guarantees every building in the GeoJSON
        // is pre-loaded regardless of where the user places start/goal.
        const data = await fetchAllBuildingsInArea(bounds);
        setBuildings(data);
        setTotalBuildings(data.meta?.total ?? data.features.length);
        setStatus("Set start & goal on map, then press Start Demo");
      } catch {
        setStatus("Could not load buildings — start backend on :8000");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

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
      setFlights(null);
      setSelectedFlightIndex(0);
      setPlaying(false);
      setSteps(0);
      setDistance(null);
      setReward(0);
      setStepReward(0);
      setPrevDistance(null);
      setStatus("Route updated — press Start Demo");
    },
    [placementMode, startPoint, goalPoint]
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
        const progressReward = progress * 2.0;   // progress_scale = 2.0
        const timePenalty    = -0.05;              // time_penalty
        const energyPenalty  = -0.01;              // small energy penalty
        const stepPenalty    = timePenalty + energyPenalty;
        const sr = progressReward + stepPenalty;
        setStepReward(sr);
        setReward((r) => r + sr);
        setTotalPenalty((p) => p + stepPenalty); // accumulate penalties separately
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

    setStatus("Planning collision-free routes…");
    setPlanningProgress(0);

    const startTime = Date.now();
    const interval = setInterval(() => {
      setPlanningProgress(() => {
        const elapsed = Date.now() - startTime;
        // 60000ms time constant makes it extremely slow from the start:
        // ~3% at 2s, ~8% at 5s, ~15% at 10s, ~28% at 20s
        return 99 * (1 - Math.exp(-elapsed / 60000));
      });
    }, 50);

    let planned: PathResponse;
    try {
      planned = await fetchPlannedPath(startPoint, goalPoint);
    } catch {
      setStatus("Planner unavailable — using straight-line fallback");
      const fallback = planPathBetween(startPoint, goalPoint);
      planned = { paths: [fallback] };
    }

    clearInterval(interval);
    setPlanningProgress(100);

    setTimeout(() => {
      setPlanningProgress(null);
      setFlights(planned.paths);
      setSelectedFlightIndex(0);
      setPlaying(false);

      const center = flightCenter(planned.paths[0].trajectory);
      setViewState((vs) => ({
        ...vs,
        longitude: center.longitude,
        latitude: center.latitude,
        zoom: Math.max(vs.zoom, 16),
      }));

      setSteps(0);
      setDistance(null);
      setStatus("Select a route and press Confirm & Fly");
    }, 600);
  }, [startPoint, goalPoint]);

  const confirmAndFly = useCallback(() => {
    if (!flight) return;
    setSteps(0);
    setReward(0);
    setStepReward(0);
    setTotalPenalty(0);
    setPrevDistance(dist3(flight.start, flight.goal));
    setDistance(dist3(flight.start, flight.goal));
    setFollowDrone(true);
    setStatus("Flying");
    setPlayId((id) => id + 1);
    setPlaying(true);
  }, [flight]);

  const reset = useCallback(() => {
    setPlaying(false);
    setFlights(null);
    setSelectedFlightIndex(0);
    resetDrone();
    setSteps(0);
    setDistance(
      startPoint && goalPoint ? dist3(startPoint, goalPoint) : null
    );
    setStatus("Idle");
    setReward(0);
    setStepReward(0);
    setTotalPenalty(0);
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

  const displayDrone = playing || flights ? dronePosition : startPoint;
  const visibleCount = buildings?.features.length ?? 0;
  
  const totalSteps = trajectory ? trajectory.length - 1 : 0;
  const flightPercentage = playing || finished
    ? (finished ? 100 : (totalSteps > 0 ? (stepIndex / totalSteps) * 100 : 0))
    : 0;

  const isPlanning = planningProgress !== null;
  const displayPercentage = isPlanning ? planningProgress : flightPercentage;
  const displayTitle = isPlanning ? "PLANNING PROGRESS" : "FLIGHT PROGRESS";
  const displayStatus = isPlanning 
    ? (planningProgress >= 100 ? "ROUTES FOUND" : "CALCULATING...") 
    : (flightPercentage >= 100 ? "FLIGHT COMPLETED" : "IN PROGRESS");

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
        <div className="sim-layout">
          <div className="sim-canvas">
            <DroneMap
              buildings={buildings}
              flights={flights}
              selectedFlightIndex={selectedFlightIndex}
              dronePosition={displayDrone}
              start={startPoint}
              goal={goalPoint}
              viewState={viewState}
              onMove={(vs) => {
                setViewState(vs);
                if (playing) setFollowDrone(false);
              }}
              placementMode={placementMode}
              onMapClick={handleMapClick}
            />
          </div>
            <ProgressUI
              percentage={displayPercentage}
              title={displayTitle}
              statusLabel={displayStatus}
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
          <span>
            Penalties:{" "}
            <strong style={{ color: "#dc2626" }}>
              {totalPenalty.toFixed(2)}
            </strong>
            <small style={{ color: "#9ca3af", marginLeft: 4 }}>
              (−0.06/step)
            </small>
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
        {flights && flights.length > 0 && !playing && (
          <div className="route-options">
            <p className="hint">Select from available paths:</p>
            <div className="btn-row" style={{ marginTop: "1rem" }}>
              {flights.map((f, i) => (
                <button
                  key={i}
                  type="button"
                  className={selectedFlightIndex === i ? "active" : ""}
                  onClick={() => setSelectedFlightIndex(i)}
                >
                  {f.name || `Path ${i + 1}`}
                </button>
              ))}
            </div>
            <div style={{ marginTop: "1rem" }}>
              <button type="button" className="btn-primary" onClick={confirmAndFly}>
                Confirm & Fly
              </button>
            </div>
          </div>
        )}
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
