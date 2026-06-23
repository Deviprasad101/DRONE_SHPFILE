export interface BuildingFeature {
  type: "Feature";
  properties: { building_id?: number; height_m?: number };
  geometry: GeoJSON.Polygon | GeoJSON.MultiPolygon;
}

export interface BuildingCollection {
  type: "FeatureCollection";
  features: BuildingFeature[];
  meta?: { total: number; limit: number; offset: number };
}

export interface FlightPath {
  start: number[];
  goal: number[];
  planned_path: number[][];
  trajectory: number[][];
  name?: string;
}

export interface PathResponse {
  paths: FlightPath[];
}

export interface Bounds {
  min_lon: number;
  min_lat: number;
  max_lon: number;
  max_lat: number;
}
