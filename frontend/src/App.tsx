import { useCallback, useEffect, useMemo, useState } from "react";
import { Play, RotateCcw, MapPin, Eye, Target, Settings, Activity, Info, Trophy, AlertTriangle, Footprints } from "lucide-react";
import DroneMap from "./components/DroneMap";
import HeightFilter from "./components/HeightFilter";
import {
  fetchAllBuildingsInArea,
  fetchBounds,
  fetchPlannedPath,
  flightCenter,
} from "./api/client";
import { useDroneAnimation } from "./hooks/useDroneAnimation";
import { defaultRouteFromCenter, planPathBetween } from "./utils/flightPath";
import type { BuildingCollection, FlightPath, PathResponse } from "./types/geo";
import {
  defaultHeightVisibility,
  isBuildingVisible,
  type FilterHeightCategoryId,
  type HeightVisibility,
} from "./utils/buildingHeightColor";
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
  const [loadingMessage, setLoadingMessage] = useState<string | null>(
    "Loading map bounds…"
  );
  const [status, setStatus] = useState("Idle");
  const [heightVisibility, setHeightVisibility] = useState<HeightVisibility>(
    defaultHeightVisibility
  );
  const [plainBuildings, setPlainBuildings] = useState(false);
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
      setLoadingMessage("Loading map bounds…");
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

        setLoadingMessage("Loading buildings (GeoJSON)…");
        const data = await fetchAllBuildingsInArea(bounds);
        setBuildings(data);
        setTotalBuildings(data.meta?.total ?? data.features.length);
        setLoadingMessage(null);
        setStatus("Set start & goal on map, then press Start Demo");
      } catch {
        setLoadingMessage(null);
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

    if (playing && !finished && stepIndex > 0) {
      setPrevDistance((prev) => {
        const progress = prev !== null ? prev - newDist : 0;
        const progressReward = progress * 2.0;   
        const timePenalty    = -0.05;              
        const energyPenalty  = -0.01;              
        const stepPenalty    = timePenalty + energyPenalty;
        const sr = progressReward + stepPenalty;
        setStepReward(sr);
        setReward((r) => r + sr);
        setTotalPenalty((p) => p + stepPenalty); 
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
    if (startPoint[0] === goalPoint[0] && startPoint[1] === goalPoint[1]) {
      setStatus("Start and goal must be different");
      return;
    }

    setStatus("Planning collision-free routes…");
    setLoadingMessage("Planning flight paths (A*)…");
    setPlanningProgress(0);

    const startTime = Date.now();
    const interval = setInterval(() => {
      setPlanningProgress(() => {
        const elapsed = Date.now() - startTime;
        return 99 * (1 - Math.exp(-elapsed / 60000));
      });
    }, 50);

    let planned: PathResponse;
    try {
      planned = await fetchPlannedPath(startPoint, goalPoint);
    } catch {
      setLoadingMessage(null);
      setStatus("Planner unavailable — using straight-line fallback");
      const fallback = planPathBetween(startPoint, goalPoint);
      planned = { paths: [fallback] };
    }

    clearInterval(interval);
    setPlanningProgress(100);
    setLoadingMessage(null);

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

  const filteredBuildings = useMemo((): BuildingCollection | null => {
    if (!buildings) return null;
    const features = buildings.features.filter((f) =>
      isBuildingVisible(f.properties, heightVisibility)
    );
    return {
      type: "FeatureCollection",
      features,
      meta: {
        total: buildings.meta?.total ?? buildings.features.length,
        limit: features.length,
        offset: 0,
      },
    };
  }, [buildings, heightVisibility]);

  const toggleHeightCategory = useCallback((id: FilterHeightCategoryId) => {
    setHeightVisibility((prev) => ({ ...prev, [id]: !prev[id] }));
  }, []);

  const visibleCount = filteredBuildings?.features.length ?? 0;
  const mapBuildingCount = buildings?.features.length ?? 0;

  const statusBadge = loading
    ? "Loading"
    : planningProgress !== null
      ? "Planning"
      : playing
        ? "Flying"
        : placementMode
          ? "Placing"
          : "Ready";

  const statusDetail =
    loading && loadingMessage
      ? loadingMessage
      : planningProgress !== null
        ? planningProgress >= 100
          ? "Finalizing routes…"
          : "Planning collision-free routes (A*)…"
        : status;
  
  const totalSteps = trajectory ? trajectory.length - 1 : 0;
  const flightPercentage = playing || finished
    ? (finished ? 100 : (totalSteps > 0 ? (stepIndex / totalSteps) * 100 : 0))
    : 0;

  const isPlanning = planningProgress !== null;
  const displayPercentage = isPlanning ? planningProgress : flightPercentage;



  return (
    <div className="dashboard-app">
      <header className="top-nav">
        <div className="nav-brand">
          <div className="nav-icon">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
               <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
               <circle cx="12" cy="12" r="2" />
               <path d="M12 10v-2" />
               <path d="M12 14v2" />
               <path d="M10 12H8" />
               <path d="M14 12h2" />
            </svg>
          </div>
          <div className="nav-titles">
            <h1>RL INDOOR DRONE NAVIGATION</h1>
            <p>Autonomous Navigation with 3D Buildings from GeoJSON (Chennai)</p>
          </div>
        </div>

      </header>

      <main className="dashboard-content">
        {/* LEFT SIDEBAR */}
        <aside className="sidebar-left">
          <div className="card">
            <h3 className="sidebar-title"><Settings size={16} /> CONTROLS</h3>
            <div className="action-row">
              <button 
                className="btn btn-primary" 
                onClick={startDemo}
                disabled={playing || loading || !startPoint || !goalPoint}
              >
                <Play size={16} fill="currentColor" /> Start Demo
              </button>
              <button className="btn btn-outline" onClick={reset} disabled={loading}>
                <RotateCcw size={16} /> Reset
              </button>
            </div>
          </div>

          <div className="card">
            <h3 className="sidebar-title"><MapPin size={16} /> ROUTE SELECTION</h3>
            <div className="action-row">
              <span className="stat-label">Set Start Position</span>
              <button 
                className={`btn ${placementMode === "start" ? "btn-light-blue" : "btn-outline"}`}
                onClick={() => setPlacementMode("start")}
                disabled={loading || playing}
              >
                <MapPin size={16} className="stat-icon" /> Set Start (Blue)
              </button>
              
              <span className="stat-label" style={{marginTop: "0.5rem"}}>Set Goal Position</span>
              <button 
                className={`btn ${placementMode === "goal" ? "btn-light-green" : "btn-outline"}`}
                onClick={() => setPlacementMode("goal")}
                disabled={loading || playing}
              >
                <MapPin size={16} className="stat-icon green" /> Set Goal (Green)
              </button>
            </div>
          </div>



          <div className="card">
            <h3 className="sidebar-title"><Eye size={16} /> VIEW OPTIONS</h3>
            <div className="view-toggle">
              <button className={viewState.pitch === 0 ? "active" : ""} onClick={()=>setViewState({...viewState, pitch: 0})}>2D View</button>
              <button className={viewState.pitch > 0 ? "active" : ""} onClick={()=>setViewState({...viewState, pitch: 60})}>3D View</button>
            </div>
            
            <div style={{ marginTop: "0.75rem", marginBottom: "0.75rem" }}>
              <button 
                className={`btn ${followDrone ? "btn-light-blue" : "btn-outline"}`}
                onClick={() => setFollowDrone(!followDrone)}
                style={{ width: "100%", display: "flex", justifyContent: "center", alignItems: "center" }}
              >
                <Target size={16} style={{ marginRight: "0.5rem" }}/> {followDrone ? "Following Drone" : "Follow Drone"}
              </button>
            </div>

            <div className="slider-container">
              <span>Tilt</span>
              <input 
                type="range" 
                min="0" max="60" 
                value={viewState.pitch} 
                onChange={(e) => setViewState({...viewState, pitch: parseInt(e.target.value)})}
              />
              <span style={{minWidth: '24px', textAlign: 'right'}}>{Math.round(viewState.pitch)}°</span>
            </div>
          </div>

          <HeightFilter
            visibility={heightVisibility}
            onToggle={toggleHeightCategory}
            plainBuildings={plainBuildings}
            onPlainBuildingsChange={setPlainBuildings}
            visibleCount={visibleCount}
            totalCount={mapBuildingCount}
          />
        </aside>

        {/* MIDDLE COLUMN */}
        <div className="main-center">
          <div className="sim-card">
            <div className="sim-header">
              <div className="sim-title">
                <h2><Activity size={18} /> 3D ANIMATION (SIMULATION)</h2>
                <p>
                  {loading && loadingMessage
                    ? loadingMessage
                    : placementMode
                      ? `Click the map to place ${placementMode === "start" ? "START (blue)" : "GOAL (green)"}`
                      : planningProgress !== null
                        ? "Computing collision-free routes…"
                        : `Showing ${visibleCount.toLocaleString()} of ${totalBuildings.toLocaleString()} buildings. Use Set Start / Set Goal, click the map, then Start Demo.`}
                </p>
              </div>
              <span
                className={`sim-badge ${
                  loading
                    ? "sim-badge-loading"
                    : planningProgress !== null
                      ? "sim-badge-planning"
                      : playing
                        ? "sim-badge-flying"
                        : "sim-badge-ready"
                }`}
              >
                {statusBadge.toUpperCase()}
              </span>
            </div>
            
            <div className="map-container">
              <DroneMap
                buildings={filteredBuildings}
                useHeightColors={!plainBuildings}
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

            <div className="stats-bar">
              <div className="stat-item">
                <Footprints className="stat-icon" />
                <div className="stat-details">
                  <span className="stat-label">Steps</span>
                  <span className="stat-value">{steps}</span>
                </div>
              </div>
              
              <div className="stat-divider"></div>
              
              <div className="stat-item">
                <Target className="stat-icon red" />
                <div className="stat-details">
                  <span className="stat-label">Distance to Goal</span>
                  <span className="stat-value">{distance !== null ? distance.toFixed(5) : "0.00000"} km</span>
                </div>
              </div>
              
              <div className="stat-divider"></div>
              
              <div className="stat-item">
                <Info className="stat-icon" />
                <div className="stat-details">
                  <span className="stat-label">Status</span>
                  <span className="stat-value" style={{fontSize: "0.85rem"}}>{statusDetail}</span>
                </div>
              </div>

              <div className="stat-divider"></div>

              <div className="stat-item">
                <Activity className="stat-icon green" />
                <div className="stat-details">
                  <span className="stat-label">Step Reward</span>
                  <span className={`stat-value ${stepReward >= 0 ? "green" : "red"}`}>
                    {stepReward >= 0 ? "+" : ""}{stepReward.toFixed(3)}
                  </span>
                </div>
              </div>

              <div className="stat-divider"></div>

              <div className="stat-item">
                <Trophy className="stat-icon yellow" />
                <div className="stat-details">
                  <span className="stat-label">Total Reward</span>
                  <span className={`stat-value ${reward >= 0 ? "green" : "red"}`}>
                    {reward >= 0 ? "+" : ""}{reward.toFixed(2)}
                  </span>
                </div>
              </div>

              <div className="stat-divider"></div>

              <div className="stat-item">
                <AlertTriangle className="stat-icon red" />
                <div className="stat-details">
                  <span className="stat-label">Penalties</span>
                  <span className="stat-value red">{totalPenalty.toFixed(2)}</span>
                  <span className="stat-sub">(−0.06/step)</span>
                </div>
              </div>
            </div>

            {flights && flights.length > 0 && !playing && (
               <div style={{marginTop: "1rem", display: "flex", gap: "1rem", alignItems: "center"}}>
                  <span className="stat-label">Select from available paths:</span>
                  {flights.map((f, i) => (
                    <button
                      key={i}
                      type="button"
                      className={`btn ${selectedFlightIndex === i ? "btn-primary" : "btn-outline"}`}
                      style={{width: "auto"}}
                      onClick={() => setSelectedFlightIndex(i)}
                    >
                      {f.name || `Path ${i + 1}`}
                    </button>
                  ))}
                  <button type="button" className="btn btn-primary" style={{width: "auto", marginLeft: "auto"}} onClick={confirmAndFly}>
                    Confirm & Fly
                  </button>
               </div>
            )}
          </div>
        </div>

        {/* RIGHT SIDEBAR */}
        <aside className="sidebar-right">
          <ProgressUI
            percentage={displayPercentage}
            title={isPlanning ? "PLANNING PROGRESS" : "FLIGHT PROGRESS"}
            statusLabel={isPlanning ? (planningProgress >= 100 ? "ROUTES FOUND" : "CALCULATING...") : (flightPercentage >= 100 ? "COMPLETED" : "IN PROGRESS")}
            timeElapsed={playing ? (stepIndex * 0.5).toFixed(2) : "00:00:00"}
            estRemaining={playing && totalSteps > 0 ? ((totalSteps - stepIndex) * 0.5).toFixed(2) : "--:--:--"}
            avgSpeed={playing ? "12.5 m/s" : "0.00 m/s"}
          />

          <div className="card">
            <h3 className="sidebar-title" style={{color: "#3b82f6"}}><MapPin size={16} /> ROUTE DETAILS</h3>
            <div className="route-details">
              <div className="detail-row">
                <div className="dot blue"></div>
                <div className="detail-text">
                  <span className="label">Start</span>
                  <span className="val">{startPoint ? `${fmtCoord(startPoint[1])}, ${fmtCoord(startPoint[0])}` : "Not Set"}</span>
                </div>
              </div>
              <div className="detail-row">
                <div className="dot green"></div>
                <div className="detail-text">
                  <span className="label">Goal</span>
                  <span className="val">{goalPoint ? `${fmtCoord(goalPoint[1])}, ${fmtCoord(goalPoint[0])}` : "Not Set"}</span>
                </div>
              </div>
              <div className="detail-row">
                <Activity size={12} />
                <div className="detail-text" style={{gap: '0.5rem'}}>
                  <span className="label" style={{width:'auto'}}>Altitude</span>
                  <span className="val" style={{fontWeight: 600}}>{startPoint?.[2]?.toFixed(0) ?? 85} m</span>
                </div>
              </div>
              <div className="route-distance">
                <span className="label">Route Distance</span>
                <span className="val">~ {flight ? (flight.trajectory.length * 0.01).toFixed(2) : "0.00"} km (est.)</span>
              </div>
            </div>
          </div>

          <div className="card">
            <h3 className="sidebar-title" style={{color: "#3b82f6"}}><Info size={16} /> STATUS</h3>
            <div>
              <div className={`status-badge status-badge-${statusBadge.toLowerCase()}`}>
                {statusBadge}
              </div>
              <p className="status-msg">{statusDetail}</p>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
