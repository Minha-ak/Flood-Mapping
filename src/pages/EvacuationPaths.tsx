import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, Marker, Popup } from 'react-leaflet';
import MapClickHandler from '../components/MapClickHandler';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { Feature, FeatureCollection, GeoJsonObject, LineString } from 'geojson';

interface FloodData extends FeatureCollection {
  type: 'FeatureCollection';
  features: Array<{
    type: 'Feature';
    properties: Record<string, any>;
    geometry: {
      type: 'Polygon';
      coordinates: number[][][];
    };
  }>;
}

interface BBox {
  min: [number, number];
  max: [number, number];
}

interface RouteData extends Feature {
  type: 'Feature';
  properties: Record<string, any>;
  geometry: LineString;
}

const EvacuationPaths: React.FC = () => {
  const [floodData, setFloodData] = useState<FloodData | null>(null);
  const [bbox, setBbox] = useState<BBox | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [route, setRoute] = useState<RouteData | null>(null);
  const [startPoint, setStartPoint] = useState<[number, number] | null>(null);
  const [endPoint, setEndPoint] = useState<[number, number] | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch flood area data
        const floodResponse = await fetch('http://localhost:5000/evacuation/new/flood_area');
        if (!floodResponse.ok) {
          throw new Error('Failed to load flood data');
        }
        const floodJson = await floodResponse.json();
        setFloodData(floodJson);

        // Fetch bounding box
        const bboxResponse = await fetch('http://localhost:5000/evacuation/new/bbox');
        if (!bboxResponse.ok) {
          throw new Error('Failed to load bounding box');
        }
        const bboxJson = await bboxResponse.json();
        setBbox(bboxJson);

        setLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const calculateRoute = async (start: [number, number], end: [number, number]) => {
    try {
      setError(null);
      const response = await fetch('http://localhost:5000/evacuation/new/route', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          start: { lat: start[0], lng: start[1] },
          end: { lat: end[0], lng: end[1] },
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to calculate route');
      }

      const routeData = await response.json();
      setRoute(routeData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to calculate route');
      setRoute(null);
    }
  };

  const handleMapClick = (latlng: L.LatLng) => {
    if (!startPoint) {
      setStartPoint([latlng.lat, latlng.lng]);
    } else if (!endPoint) {
      setEndPoint([latlng.lat, latlng.lng]);
      calculateRoute(startPoint, [latlng.lat, latlng.lng]);
    }
  };

  const resetPoints = () => {
    setStartPoint(null);
    setEndPoint(null);
    setRoute(null);
    setError(null);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading flood data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <p className="text-red-500 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  if (!bbox) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <p className="text-red-500">No flood data available</p>
        </div>
      </div>
    );
  }

  const center: [number, number] = [
    (bbox.min[1] + bbox.max[1]) / 2,
    (bbox.min[0] + bbox.max[0]) / 2,
  ];

  return (
    <div className="h-screen flex flex-col">
      <div className="p-4 bg-white shadow">
        <h1 className="text-2xl font-bold mb-2">Evacuation Path Planning</h1>
        <p className="text-gray-600 mb-4">
          {!startPoint
            ? "Click on the map to set the starting point"
            : !endPoint
            ? "Click on the map to set the destination point"
            : "Route calculated. Click 'Reset' to plan a new route"}
        </p>
        <button
          onClick={resetPoints}
          className="bg-red-500 text-white px-4 py-2 rounded hover:bg-red-600"
          disabled={!startPoint && !endPoint}
        >
          Reset Points
        </button>
      </div>

      <div className="flex-1 relative">
        <MapContainer
          center={center}
          zoom={13}
          style={{ height: '100%', width: '100%' }}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          <MapClickHandler onMapClick={handleMapClick} />
          
          {floodData && (
            <GeoJSON
              data={floodData}
              style={() => ({
                color: '#3182ce',
                weight: 2,
                opacity: 0.6,
                fillOpacity: 0.2,
              })}
            />
          )}

          {startPoint && (
            <Marker position={startPoint}>
              <Popup>Start Point</Popup>
            </Marker>
          )}

          {endPoint && (
            <Marker position={endPoint}>
              <Popup>End Point</Popup>
            </Marker>
          )}

          {route && (
            <GeoJSON
              data={route}
              style={() => ({
                color: '#e53e3e',
                weight: 4,
                opacity: 0.8,
              })}
            />
          )}
        </MapContainer>
      </div>
    </div>
  );
};

export default EvacuationPaths;