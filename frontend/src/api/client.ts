import type { Bounds, BuildingCollection, FlightPath } from "../types/geo";

const API = "/api";

/** Fallback demo path over Chennai building area (WGS84). */
export function localDemoFlight(): FlightPath {
  const cx = 80.2292;
  const cy = 12.9982;
  const d = 0.005;
  const alt = 85;
  const trajectory: number[][] = [
    [cx - d, cy - d, alt],
    [cx - d * 0.5, cy - d * 0.25, alt + 8],
    [cx, cy - d * 0.1, alt + 15],
    [cx + d * 0.4, cy + d * 0.3, alt + 10],
    [cx + d * 0.8, cy + d * 0.6, alt + 5],
    [cx + d, cy + d, alt],
  ];
  return {
    start: trajectory[0],
    goal: trajectory[trajectory.length - 1],
    planned_path: trajectory,
    trajectory,
    name: "Local demo flight",
  };
}

export async function fetchBounds(): Promise<Bounds> {
  try {
    const res = await fetch(`${API}/buildings/bounds`);
    if (!res.ok) throw new Error("bounds failed");
    return res.json();
  } catch {
    return { min_lon: 80.12, min_lat: 12.93, max_lon: 80.26, max_lat: 13.04 };
  }
}

export async function fetchBuildingsInView(
  bounds: Bounds,
  limit = 1500,
  offset = 0
): Promise<BuildingCollection> {
  const q = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
    min_lon: String(bounds.min_lon),
    min_lat: String(bounds.min_lat),
    max_lon: String(bounds.max_lon),
    max_lat: String(bounds.max_lat),
  });
  const res = await fetch(`${API}/buildings?${q}`);
  if (!res.ok) throw new Error("Failed to load buildings");
  return res.json();
}

/** Load all buildings in a bbox (paginated, capped for browser performance). */
export async function fetchAllBuildingsInArea(
  bounds: Bounds,
  maxFeatures = 15000
): Promise<BuildingCollection> {
  const pageSize = 5000;
  const features: BuildingCollection["features"] = [];
  let offset = 0;
  let total = 0;

  while (features.length < maxFeatures) {
    const batch = await fetchBuildingsInView(bounds, pageSize, offset);
    total = batch.meta?.total ?? batch.features.length;
    features.push(...batch.features);
    if (features.length >= total || batch.features.length < pageSize) break;
    offset += pageSize;
  }

  return {
    type: "FeatureCollection",
    features: features.slice(0, maxFeatures),
    meta: { total, limit: features.length, offset: 0 },
  };
}

export async function fetchDemoFlight(): Promise<FlightPath> {
  try {
    const res = await fetch(`${API}/demo-flight`);
    if (!res.ok) throw new Error("demo failed");
    return res.json();
  } catch {
    return localDemoFlight();
  }
}

/** Center point of a flight path for map fly-to. */
export function flightCenter(path: number[][]): { longitude: number; latitude: number } {
  const mid = path[Math.floor(path.length / 2)];
  return { longitude: mid[0], latitude: mid[1] };
}
