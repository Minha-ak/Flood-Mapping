from flask import Flask, request, jsonify, render_template
import geopandas as gpd
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString, Point, box
import json
import traceback
import os
import pickle
from math import sqrt, sin, cos, asin

app = Flask(__name__)

# Cache file paths
CACHE_DIR = "cache"
NETWORK_CACHE = os.path.join(CACHE_DIR, "road_network.pkl")
FLOOD_CACHE = os.path.join(CACHE_DIR, "flood_data.pkl")

# Global variables
G = None
safe_nodes = None
safe_nodes_proj = None
flood = None
flood_union = None
area_bounds = None
center_lat = None
center_lon = None
error_message = None
map_initialized = False

def geographic_distance(node1, node2):
    """Calculate the geographic distance between two nodes in meters"""
    # Get node coordinates
    lat1, lon1 = G.nodes[node1]['y'], G.nodes[node1]['x']
    lat2, lon2 = G.nodes[node2]['y'], G.nodes[node2]['x']
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(lambda x: x * 3.14159 / 180, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Earth's radius in meters
    return c * r

def load_cached_data(force_refresh=False):
    global G, safe_nodes, flood, flood_union, area_bounds, center_lat, center_lon, safe_nodes_proj, error_message, map_initialized
    
    # Create cache directory if it doesn't exist
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    # Try to load from cache first if not forcing refresh
    if not force_refresh and os.path.exists(NETWORK_CACHE) and os.path.exists(FLOOD_CACHE):
        print("Loading data from cache...")
        try:
            with open(NETWORK_CACHE, 'rb') as f:
                cache_data = pickle.load(f)
                G = cache_data['graph']
                safe_nodes = cache_data['safe_nodes']
                safe_nodes_proj = cache_data['safe_nodes_proj']
            with open(FLOOD_CACHE, 'rb') as f:
                cache_data = pickle.load(f)
                flood = cache_data['flood']
                flood_union = cache_data['flood_union']
                area_bounds = cache_data['area_bounds']
                center_lat = cache_data['center_lat']
                center_lon = cache_data['center_lon']
            print("✅ Successfully loaded data from cache")
            map_initialized = True
            return True
        except Exception as e:
            print(f"❌ Error loading from cache: {str(e)}")
            error_message = f"Error loading cached data: {str(e)}"
            print("Will try to process data again...")
    
    return False

def process_and_cache_data():
    global G, safe_nodes, flood, flood_union, area_bounds, center_lat, center_lon, safe_nodes_proj, error_message, map_initialized
    
    try:
        print("Loading flood polygons...")
        flood_file = "uploads/2/flood_polygons_final.geojson"
        
        # Check if flood file exists
        if not os.path.exists(flood_file):
            print(f"❌ ERROR: Flood file not found at {os.path.abspath(flood_file)}")
            return False
        
        # Load and process flood data
        flood = gpd.read_file(flood_file).to_crs(epsg=4326)
        flood_union = flood.geometry.union_all()
        
        # Get the bounds of the flood area
        bounds = flood_union.bounds
        buffer_size = 0.03
        area_bounds = {
            'west': bounds[0] - buffer_size,
            'south': bounds[1] - buffer_size,
            'east': bounds[2] + buffer_size,
            'north': bounds[3] + buffer_size
        }
        
        # Create a study area polygon
        study_area = box(area_bounds['west'], 
                        area_bounds['south'], 
                        area_bounds['east'], 
                        area_bounds['north'])
        
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        print(f"✅ Flood polygons processed, center coordinates: {center_lat}, {center_lon}")
        
        # Get road network
        print("📡 Downloading road network (this may take a few minutes)...")
        try:
            G = ox.graph_from_polygon(
                study_area, 
                network_type="walk",
                simplify=True,
                retain_all=False,
                truncate_by_edge=True
            )
        except Exception as e:
            print(f"❌ Error downloading road network: {str(e)}")
            error_message = f"Error downloading road network: {str(e)}"
            return False
        
        # Process road network
        nodes, edges = ox.graph_to_gdfs(G)
        edges_clipped = edges.clip(study_area)
        flooded_edges = edges_clipped[edges_clipped.intersects(flood_union)]
        G.remove_edges_from(flooded_edges.index)
        print(f"🚫 Removed {len(flooded_edges)} flooded road segments")
        
        # Get largest connected component
        largest_components = list(nx.strongly_connected_components(G))
        if not largest_components:
            print("❌ No connected components found in the road network")
            error_message = "No connected components found in the road network"
            return False
            
        largest_component = max(largest_components, key=len)
        G = G.subgraph(largest_component).copy()
        print(f"🧩 Using largest connected component with {len(G.nodes)} nodes")
        
        # Get safe nodes and project them
        nodes = ox.graph_to_gdfs(G, edges=False)
        safe_nodes = nodes[~nodes.geometry.intersects(flood_union)]
        
        if safe_nodes.empty:
            print("❌ No safe nodes found in the road network")
            error_message = "No safe nodes found in the road network"
            return False
            
        safe_nodes_proj = safe_nodes.to_crs("EPSG:3857")
        
        print("✅ Road network processed")
        
        # Cache the processed data
        print("Caching processed data...")
        with open(NETWORK_CACHE, 'wb') as f:
            pickle.dump({
                'graph': G,
                'safe_nodes': safe_nodes,
                'safe_nodes_proj': safe_nodes_proj
            }, f)
        with open(FLOOD_CACHE, 'wb') as f:
            pickle.dump({
                'flood': flood,
                'flood_union': flood_union,
                'area_bounds': area_bounds,
                'center_lat': center_lat,
                'center_lon': center_lon
            }, f)
        print("✅ Data cached successfully")
        
        map_initialized = True
        return True
    except Exception as e:
        print(f"❌ Error processing data: {str(e)}")
        error_message = f"Error processing data: {str(e)}"
        traceback.print_exc()
        return False

# Initialize data
if not load_cached_data():
    if not process_and_cache_data():
        print("❌ Failed to initialize data")
        error_message = "Failed to initialize data. See server logs for details."

@app.route("/")
def index():
    return render_template('map.html')

@app.route("/bbox")
def get_bbox():
    """Return the bounding box of the flood area"""
    if area_bounds is None:
        return jsonify({"error": "Bounding box not available"}), 500
    return jsonify({
        "min": [area_bounds['west'], area_bounds['south']],
        "max": [area_bounds['east'], area_bounds['north']]
    })

@app.route("/flood_area")
def flood_area():
    """Return the flood polygons as GeoJSON"""
    try:
        if flood is None:
            print("❌ Error: flood data is None")
            return jsonify({"error": "Flood data not available"}), 500
            
        flood_json = json.loads(flood.to_json())
        print(f"✅ Serving flood area data with {len(flood_json.get('features', []))} features")
        return jsonify(flood_json)
    except Exception as e:
        print(f"❌ Error serving flood area: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/route", methods=["POST"])
def route():
    try:
        data = request.json
        start_lat, start_lon = data["start"]["lat"], data["start"]["lng"]
        end_lat, end_lon = data["end"]["lat"], data["end"]["lng"]

        print(f"Calculating route from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")

        if G is None or safe_nodes is None:
            print("❌ Road network not available")
            return jsonify({"error": "Road network not available. Please try again later."})

        # Convert input points to projected CRS for accurate distance calculations
        start_point = gpd.GeoSeries([Point(start_lon, start_lat)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        end_point = gpd.GeoSeries([Point(end_lon, end_lat)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

        # Check if points are in flooded area
        if flood_union is not None:
            if start_point.intersects(flood_union):
                print("❌ Start point is in flooded area")
                return jsonify({"error": "Start point is in a flooded area. Please choose a safe location."})
            if end_point.intersects(flood_union):
                print("❌ End point is in flooded area")
                return jsonify({"error": "End point is in a flooded area. Please choose a safe location."})

        # Find nearest safe nodes using projected coordinates
        safe_nodes_proj["dist_start"] = safe_nodes_proj.distance(start_point)
        safe_nodes_proj["dist_end"] = safe_nodes_proj.distance(end_point)
        
        if len(safe_nodes_proj) == 0:
            print("❌ No safe nodes found")
            return jsonify({"error": "No safe nodes found in the area. Please try different locations."})
            
        start_node = safe_nodes_proj["dist_start"].idxmin()
        end_node = safe_nodes_proj["dist_end"].idxmin()

        print(f"Found nearest nodes: {start_node} and {end_node}")

        if start_node == end_node:
            print("❌ Start and end points are too close")
            return jsonify({"error": "Start and end points are too close. Select locations farther apart."})

        try:
            # Check if both nodes are in the graph
            if start_node not in G or end_node not in G:
                print(f"❌ One or both nodes not in graph: start={start_node in G}, end={end_node in G}")
                return jsonify({"error": "One or both selected points cannot be connected to the road network."})
                
            # Use A* algorithm with custom heuristic
            print("Calculating path...")
            try:
                path = nx.astar_path(G, start_node, end_node, weight="length", heuristic=geographic_distance)
                print(f"Path found with {len(path)} nodes")
            except nx.NetworkXNoPath:
                # Try Dijkstra's algorithm as fallback
                print("A* failed, trying Dijkstra...")
                path = nx.dijkstra_path(G, start_node, end_node, weight="length")
                print(f"Dijkstra path found with {len(path)} nodes")
            
            path_nodes = [G.nodes[n] for n in path]
            coords = [(node['x'], node['y']) for node in path_nodes]
                
            if len(coords) < 2:
                print("❌ Path is too short")
                return jsonify({"error": "Evacuation path is too short."})
                
            line = LineString(coords)
            path_gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326")
            print("✅ Route calculated successfully")
            return jsonify(json.loads(path_gdf.to_json()))
        except nx.NetworkXNoPath:
            print("❌ No path found between points")
            return jsonify({"error": "No safe path found between the selected points. Please try different locations."})
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/refresh_data", methods=["POST"])
def refresh_data():
    """Force refresh the road network and flood data"""
    global G, safe_nodes, flood, flood_union, area_bounds, center_lat, center_lon, safe_nodes_proj, error_message, map_initialized
    
    try:
        # Clear existing cache
        if os.path.exists(NETWORK_CACHE):
            os.remove(NETWORK_CACHE)
        if os.path.exists(FLOOD_CACHE):
            os.remove(FLOOD_CACHE)
        
        # Reset error message
        error_message = None
        map_initialized = False
        
        # Process data again
        if process_and_cache_data():
            return jsonify({"status": "success", "message": "Data refreshed successfully"})
        else:
            return jsonify({"status": "error", "message": error_message or "Failed to refresh data"})
    except Exception as e:
        print(f"❌ Error refreshing data: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)