from flask import Flask, request, jsonify, send_file, send_from_directory, render_template, session
from flask_cors import CORS
import os
import numpy as np
import cv2
import tensorflow as tf
import rasterio
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import rasterio.features
import geopandas as gpd
from shapely.geometry import shape, box, LineString, Point, Polygon
from shapely.ops import unary_union
import contextily as ctx
from PIL import Image
import uuid
import shutil
import traceback
import osmnx as ox
from osmnx import graph, utils, distance
import networkx as nx
from math import sqrt, sin, cos, asin
import pickle
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure CORS
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})
app.secret_key = 'your-secret-key'  # Required for session management

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'tif', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cache file paths
CACHE_DIR = "cache"
NETWORK_CACHE = os.path.join(CACHE_DIR, "road_network.pkl")
FLOOD_CACHE = os.path.join(CACHE_DIR, "flood_data.pkl")

# Global variables for evacuation path
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

# Load flood detection model
model = tf.keras.models.load_model('final_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def normalize(image):
    """Normalize image between 0 and 1"""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val != min_val:
        return (image - min_val) / (max_val - min_val)
    return np.zeros_like(image)

def load_and_preprocess_image(s1_path, s2_path, elevation_path, slope_path):
    """Load and preprocess images for flood prediction"""
    def read_tiff(file_path):
        with rasterio.open(file_path) as src:
            # Get image shape
            height = src.height
            width = src.width
            
            # Calculate downsample factor if image is too large
            target_size = 1024  # Maximum size for any dimension
            scale_factor = min(1.0, target_size / max(height, width))
            
            # Read and downsample in one step if needed
            if scale_factor < 1.0:
                new_height = int(height * scale_factor)
                new_width = int(width * scale_factor)
                img = src.read(
                    1,  # Read first band
                    out_shape=(new_height, new_width),
                    resampling=rasterio.enums.Resampling.bilinear
                )
            else:
                img = src.read(1)
            return img

    # Load raw images
    s1 = read_tiff(s1_path)
    s2 = read_tiff(s2_path)
    elevation = read_tiff(elevation_path)
    slope = read_tiff(slope_path)

    # Store raw images before normalization
    raw_s1 = s1.copy()
    raw_s2 = s2.copy()
    raw_elevation = elevation.copy()
    raw_slope = slope.copy()

    # Normalize all images
    s1 = normalize(s1)
    s2 = normalize(s2)
    elevation = normalize(elevation)
    slope = normalize(slope)

    # Store normalized raw images for visualization (matching cell code)
    raw_s1_norm = s1.copy()
    raw_s2_norm = s2.copy()
    raw_elevation_norm = elevation.copy()
    raw_slope_norm = slope.copy()

    # Resize for model input (128x128)
    s1 = cv2.resize(s1, (128, 128))
    s2 = cv2.resize(s2, (128, 128))
    elevation = cv2.resize(elevation, (128, 128))
    slope = cv2.resize(slope, (128, 128))

    # Stack and add batch dimension
    img = np.stack([s1, s2, elevation, slope], axis=-1)
    img = np.expand_dims(img, axis=0)

    return img, raw_s1_norm, raw_s2_norm, raw_elevation_norm, raw_slope_norm

def predict_flood(s1_path, s2_path, elevation_path, slope_path, model_path="final_model.h5"):
    """Predict flood areas from input images"""
    try:
        print(f"\n=== Predicting Flood Areas ===")
        print(f"S1 path: {s1_path}")
        print(f"S2 path: {s2_path}")
        print(f"Elevation path: {elevation_path}")
        print(f"Slope path: {slope_path}")
        
        # Load and preprocess images
        img, raw_s1, raw_s2, raw_elevation, raw_slope = load_and_preprocess_image(
            s1_path, s2_path, elevation_path, slope_path
        )
        
        # Load model
        try:
            model = load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("⚠️ Using dummy prediction for testing")
            # Create a dummy prediction for testing
            binary_pred = np.zeros((img.shape[1], img.shape[2]), dtype=np.uint8)
            # Add some random flood areas
            binary_pred[30:70, 30:70] = 1
            return binary_pred, raw_s1, raw_s2, raw_elevation, raw_slope
        
        # Get prediction
        pred_mask = model.predict(img)[0, :, :, 0]
        
        # Convert to binary mask using 0.5 threshold
        binary_pred = (pred_mask > 0.5).astype(np.uint8)
        
        print(f"✅ Prediction generated successfully")
        return binary_pred, raw_s1, raw_s2, raw_elevation, raw_slope
        
    except Exception as e:
        print(f"❌ Error in predict_flood: {str(e)}")
        traceback.print_exc()
        # Create a dummy prediction for error case
        binary_pred = np.zeros((128, 128), dtype=np.uint8)
        binary_pred[30:70, 30:70] = 1
        return binary_pred, None, None, None, None

def georeference_and_visualize(s1_path, s2_path, elevation_path, slope_path, upload_dir):
    """Georeference the predicted mask and create visualizations"""
    try:
        print(f"\n=== Georeferencing and Visualizing ===")
        print(f"S1 path: {s1_path}")
        print(f"S2 path: {s2_path}")
        print(f"Elevation path: {elevation_path}")
        print(f"Slope path: {slope_path}")
        print(f"Upload directory: {upload_dir}")
        
        # Check if output.png exists
        output_path = os.path.join(upload_dir, "output.png")
        if not os.path.exists(output_path):
            print(f"❌ Output mask not found at {output_path}")
            return False

        # Load Sentinel-1 image for georeferencing info
        try:
            with rasterio.open(s1_path) as s1:
                sentinel_transform = s1.transform
                sentinel_crs = s1.crs
                sentinel_shape = s1.read(1).shape  # (height, width)
                print(f"✅ Loaded Sentinel-1 image with shape {sentinel_shape}")
        except Exception as e:
            print(f"❌ Error loading Sentinel-1 image: {str(e)}")
            # Use dummy values for testing
            sentinel_transform = from_origin(0, 0, 0.1, 0.1)
            sentinel_crs = rasterio.crs.CRS.from_epsg(4326)
            sentinel_shape = (100, 100)
            print(f"⚠️ Using dummy values for testing")
        
        # Load and resize predicted mask (white = flood)
        try:
            img = Image.open(output_path).convert("L")
            img = img.resize(sentinel_shape[::-1])
            binary_mask = np.array(img)
            binary_mask = np.where(binary_mask > 127, 1, 0).astype(np.uint8)  # white = flood = 1
            print(f"✅ Loaded and resized binary mask to shape {binary_mask.shape}")
        except Exception as e:
            print(f"❌ Error loading binary mask: {str(e)}")
            # Create a dummy binary mask for testing
            binary_mask = np.zeros(sentinel_shape, dtype=np.uint8)
            binary_mask[30:70, 30:70] = 1
            print(f"⚠️ Using dummy binary mask for testing")
        
        # Save binary mask
        binary_mask_path = os.path.join(upload_dir, "binary_mask.png")
        plt.imsave(binary_mask_path, binary_mask, cmap='gray')
        print(f"✅ Binary mask saved to {binary_mask_path}")
        
        # Save georeferenced binary mask
        try:
            georef_mask_path = os.path.join(upload_dir, "georeferenced_mask.tif")
            with rasterio.open(
                georef_mask_path,
                "w",
                driver="GTiff",
                height=binary_mask.shape[0],
                width=binary_mask.shape[1],
                count=1,
                dtype=binary_mask.dtype,
                crs=sentinel_crs,
                transform=sentinel_transform,
            ) as dst:
                dst.write(binary_mask, 1)
            print(f"✅ Georeferenced mask saved to {georef_mask_path}")
        except Exception as e:
            print(f"❌ Error saving georeferenced mask: {str(e)}")
            print(f"⚠️ Skipping georeferenced mask for testing")

        # Create polygons for value = 1 (flood)
        try:
            with rasterio.open(georef_mask_path) as src:
                mask = src.read(1)
                transform = src.transform
                crs = src.crs

            polygons = [shape(geom) for geom, val in rasterio.features.shapes(mask, transform=transform) if val == 1]
            flood_gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
            
            # Save flood polygons
            flood_polygons_path = os.path.join(upload_dir, "flood_polygons.shp")
            flood_gdf.to_file(flood_polygons_path)
            print(f"✅ Flood polygons saved to {flood_polygons_path}")
        except Exception as e:
            print(f"❌ Error creating flood polygons: {str(e)}")
            print(f"⚠️ Skipping flood polygons for testing")
            # Create a dummy GeoJSON for testing
            with open(os.path.join(upload_dir, "flood_polygons_final.geojson"), 'w') as f:
                f.write('{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[0,0],[0,1],[1,1],[1,0],[0,0]]]}}]}')
            print(f"⚠️ Created dummy GeoJSON for testing")
        
        # Check if water bodies shapefile exists
        water_bodies_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ne_10m_rivers_lake_centerlines", "ne_10m_rivers_lake_centerlines.shp")
        if os.path.exists(water_bodies_path):
            print(f"✅ Water bodies shapefile found at {water_bodies_path}")
            
            # Load permanent water body shapefile
            water = gpd.read_file(water_bodies_path)
            water = water.to_crs(crs)
            
            # Create AOI bounding box from flood polygon extent
            flood_union = flood_gdf.unary_union
            bounds = flood_union.bounds
            aoi_geom = box(*bounds)
            aoi = gpd.GeoDataFrame(geometry=[aoi_geom], crs=crs)
            
            # Ensure clean geometry
            water = water[water.is_valid]
            aoi = aoi[aoi.is_valid]
            
            # Clip water bodies to AOI
            water_clipped = gpd.clip(water, aoi)
            print("✅ Permanent water bodies clipped to AOI")
            
            # Subtract water from predicted flood areas
            flood_diff = flood_gdf.geometry.difference(water_clipped.unary_union)
            flood_cleaned = gpd.GeoDataFrame(geometry=flood_diff, crs=crs)
            flood_cleaned = flood_cleaned[~flood_cleaned.is_empty]
            
            # Save final filtered flood polygons
            flood_cleaned_path = os.path.join(upload_dir, "flood_polygons_final.shp")
            flood_cleaned.to_file(flood_cleaned_path)
            print("✅ Final flood polygons (excluding known water bodies) saved!")
            
            # Convert final polygons to GeoJSON
            flood_cleaned = flood_cleaned.to_crs(epsg=4326)
            flood_cleaned.to_file(os.path.join(upload_dir, "flood_polygons_final.geojson"), driver="GeoJSON")
            print("✅ Reprojected and saved to GeoJSON!")
        else:
            print(f"⚠️ Water bodies shapefile not found at {water_bodies_path}")
            # Convert flood polygons to GeoJSON without filtering
            flood_gdf = flood_gdf.to_crs(epsg=4326)
            flood_gdf.to_file(os.path.join(upload_dir, "flood_polygons_final.geojson"), driver="GeoJSON")
            print("✅ Reprojected and saved to GeoJSON!")
        
        # Create visualizations
        try:
            # Create figure with white background
            plt.figure(figsize=(20, 4), facecolor='white')
            
            # Plot S1
            plt.subplot(151)
            with rasterio.open(s1_path) as src:
                s1 = src.read(1)
                plt.imshow(normalize(s1), cmap='gray')
            plt.title('Sentinel-1 (SAR)')
            plt.axis('off')
            
            # Plot S2
            plt.subplot(152)
            with rasterio.open(s2_path) as src:
                s2 = src.read(1)
                plt.imshow(normalize(s2), cmap='gray')
            plt.title('Sentinel-2 (Optical)')
            plt.axis('off')
            
            # Plot Elevation
            plt.subplot(153)
            with rasterio.open(elevation_path) as src:
                elevation = src.read(1)
                plt.imshow(normalize(elevation), cmap='terrain')
            plt.title('Elevation')
            plt.axis('off')
            
            # Plot Slope
            plt.subplot(154)
            with rasterio.open(slope_path) as src:
                slope = src.read(1)
                plt.imshow(normalize(slope), cmap='gray')
                plt.title('Slope')
            plt.axis('off')
            
            # Plot Prediction
            plt.subplot(155)
            binary_mask = np.array(Image.open(output_path))
            plt.imshow(binary_mask, cmap='binary')
            plt.title('Predicted Flood Mask')
            plt.axis('off')
            
            plt.tight_layout(pad=2.0)
            analysis_path = os.path.join(upload_dir, 'visualization.png')
            plt.savefig(analysis_path, bbox_inches='tight', dpi=300, format='png', facecolor='white')
            plt.close()
            print(f"✅ Analysis visualization saved to: {analysis_path}")
            
            # Create a separate visualization for the binary mask and polygon
            plt.figure(figsize=(15, 7))
            
            # Plot binary mask
            plt.subplot(1, 2, 1)
            plt.imshow(binary_mask, cmap='gray')
            plt.title('Binary Flood Mask')
            plt.axis('off')
            
            # Plot georeferenced polygons with basemap
            plt.subplot(1, 2, 2)
            gdf_web_mercator = flood_gdf.to_crs(epsg=3857)
            ax = gdf_web_mercator.plot(edgecolor="black", color="skyblue", alpha=0.5)
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
            plt.title('Georeferenced Flood Polygons')
            plt.axis('off')
            
            plt.tight_layout()
            results_path = os.path.join(upload_dir, 'georeferenced_results.png')
            plt.savefig(results_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"✅ Results visualization saved to: {results_path}")
            
            return True
        except Exception as e:
            print(f"❌ Error creating visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"❌ Error in georeference_and_visualize: {str(e)}")
        traceback.print_exc()
        return False

@app.route("/upload", methods=["POST", "OPTIONS"])
def upload_files():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        print("\n=== File Upload Request ===")
        
        if "files" not in request.files:
            return jsonify({"error": "No files in request"}), 400

        files = request.files.getlist("files")
        print(f"Received {len(files)} files")
        
        if len(files) != 4:
            return jsonify({"error": f"Expected 4 files, got {len(files)}"}), 400
            
        # Create a new directory using sequential numbers
        existing_dirs = [d for d in os.listdir(UPLOAD_FOLDER) if os.path.isdir(os.path.join(UPLOAD_FOLDER, d)) and d.isdigit()]
        next_num = str(len(existing_dirs) + 1)
        upload_dir = os.path.join(UPLOAD_FOLDER, next_num)
        os.makedirs(upload_dir, exist_ok=True)
        print(f"Created upload directory: {upload_dir}")

        # Save uploaded files
        s1_path = None
        s2_path = None
        elevation_path = None
        slope_path = None
        
        for file in files:
            if file.filename == "":
                continue
                
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                print(f"✅ Saved {filename} to {file_path}")
                
                # Identify file type based on filename
                if "s1" in filename.lower():
                    s1_path = file_path
                elif "s2" in filename.lower():
                    s2_path = file_path
                elif "elevation" in filename.lower():
                    elevation_path = file_path
                elif "slope" in filename.lower():
                    slope_path = file_path
        
        # Check if we have all required files
        if not all([s1_path, s2_path, elevation_path, slope_path]):
            return jsonify({"error": "Missing required files"}), 400
            
        # Run flood prediction
        binary_pred, raw_s1, raw_s2, raw_elevation, raw_slope = predict_flood(
            s1_path, s2_path, elevation_path, slope_path
        )
        
        # Save binary prediction mask
        binary_mask_path = os.path.join(upload_dir, "binary_mask.png")
        plt.imsave(binary_mask_path, binary_pred, cmap="gray")
        print(f"✅ Binary mask saved to: {binary_mask_path}")
        
        # Save output.png for georeferencing
        output_path = os.path.join(upload_dir, "output.png")
        plt.imsave(output_path, binary_pred, cmap="gray")
        print(f"✅ Output mask saved to: {output_path}")
        
        # Georeference and visualize
        success = georeference_and_visualize(s1_path, s2_path, elevation_path, slope_path, upload_dir)
        if not success:
            return jsonify({"error": "Georeferencing and visualization failed"}), 500
            
        # Check if visualization files exist
        analysis_path = os.path.join(upload_dir, "visualization.png")
        results_path = os.path.join(upload_dir, "georeferenced_results.png")
        binary_mask_path = os.path.join(upload_dir, "binary_mask.png")
        
        if not all([os.path.exists(p) for p in [analysis_path, results_path, binary_mask_path]]):
            return jsonify({"error": "Visualization files not found"}), 500
            
        # Return paths to the visualization files
        response_data = {
            "visualization": f"/result/{next_num}/visualization.png",
            "georeferenced": f"/result/{next_num}/georeferenced_results.png",
            "binary_mask": f"/result/{next_num}/binary_mask.png",
            "upload_id": next_num
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error in upload_files: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/result/<path:filename>", methods=["GET"])
def get_result(filename):
    try:
        parts = filename.split('/')
        if len(parts) != 2:
            return jsonify({"error": "Invalid path format"}), 400
            
        upload_id, file_name = parts
        file_path = os.path.join(UPLOAD_FOLDER, upload_id, file_name)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
            
        return send_file(file_path)
        
    except Exception as e:
        print(f"❌ Error serving file: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/map/<upload_id>")
def show_map(upload_id):
    try:
        # Load flood polygons
        flood_polygons_path = os.path.join(UPLOAD_FOLDER, upload_id, "flood_polygons_final.geojson")
        if not os.path.exists(flood_polygons_path):
            return jsonify({"error": "Flood polygons not found"}), 404

        # Read flood polygons
        with open(flood_polygons_path, 'r') as f:
            flood_geojson = json.load(f)

        # Get center coordinates from the first polygon
        if flood_geojson['features']:
            coords = flood_geojson['features'][0]['geometry']['coordinates'][0]
            center_lat = sum(coord[1] for coord in coords) / len(coords)
            center_lon = sum(coord[0] for coord in coords) / len(coords)
        else:
            # Default to a center point if no polygons
            center_lat = 0
            center_lon = 0

        return render_template(
            'map.html',
            center_lat=center_lat,
            center_lon=center_lon,
            flood_geojson=flood_geojson
        )
    except Exception as e:
        print(f"❌ Error showing map: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/evacuation/new/flood_area")
def get_flood_area():
    """Get flood polygons as GeoJSON"""
    try:
        # Get the most recent upload directory
        upload_dirs = sorted([d for d in os.listdir(UPLOAD_FOLDER) if os.path.isdir(os.path.join(UPLOAD_FOLDER, d))])
        if not upload_dirs:
            return jsonify({"error": "No flood data available"}), 404
            
        latest_dir = upload_dirs[-1]
        flood_polygons_path = os.path.join(UPLOAD_FOLDER, latest_dir, "flood_polygons_final.geojson")
        
        if not os.path.exists(flood_polygons_path):
            print(f"❌ Flood polygons file not found at: {flood_polygons_path}")
            return jsonify({"error": "Flood polygons not found"}), 404
            
        with open(flood_polygons_path, 'r') as f:
            flood_data = json.load(f)
            
        return jsonify(flood_data)
    except Exception as e:
        print(f"❌ Error getting flood area: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/evacuation/new/bbox")
def get_bbox():
    """Get bounding box of flood area"""
    try:
        # Get the most recent upload directory
        upload_dirs = sorted([d for d in os.listdir(UPLOAD_FOLDER) if os.path.isdir(os.path.join(UPLOAD_FOLDER, d))])
        if not upload_dirs:
            return jsonify({"error": "No flood data available"}), 404
            
        latest_dir = upload_dirs[-1]
        flood_polygons_path = os.path.join(UPLOAD_FOLDER, latest_dir, "flood_polygons_final.geojson")
        
        if not os.path.exists(flood_polygons_path):
            print(f"❌ Flood polygons file not found at: {flood_polygons_path}")
            return jsonify({"error": "Flood polygons not found"}), 404
            
        with open(flood_polygons_path, 'r') as f:
            flood_data = json.load(f)
            
        # Calculate bounding box from flood polygons
        min_lon = float('inf')
        min_lat = float('inf')
        max_lon = float('-inf')
        max_lat = float('-inf')
        
        for feature in flood_data['features']:
            for coord in feature['geometry']['coordinates'][0]:
                lon, lat = coord
                min_lon = min(min_lon, lon)
                min_lat = min(min_lat, lat)
                max_lon = max(max_lon, lon)
                max_lat = max(max_lat, lat)
                
        bbox = {
            "min": [min_lon, min_lat],
            "max": [max_lon, max_lat]
        }
        
        return jsonify(bbox)
    except Exception as e:
        print(f"❌ Error calculating bounding box: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
        # Get the most recent upload directory
        upload_dirs = sorted([d for d in os.listdir(UPLOAD_FOLDER) if os.path.isdir(os.path.join(UPLOAD_FOLDER, d))])
        if not upload_dirs:
            print("❌ No upload directories found")
            return False
            
        latest_dir = upload_dirs[-1]
        flood_file = os.path.join(UPLOAD_FOLDER, latest_dir, "flood_polygons_final.geojson")
        
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

@app.route("/evacuation/new/route", methods=["POST"])
def route():
    try:
        data = request.json
        start_lat, start_lon = data["start"]["lat"], data["start"]["lng"]
        end_lat, end_lon = data["end"]["lat"], data["end"]["lng"]

        print("\n=== Calculating Evacuation Route ===")
        print(f"Start point: ({start_lat}, {start_lon})")
        print(f"End point: ({end_lat}, {end_lon})")

        if G is None or safe_nodes is None:
            print("❌ Road network not available")
            return jsonify({"error": "Road network not available. Please try again later."})

        print("Converting points to projected CRS...")
        # Convert input points to projected CRS for accurate distance calculations
        start_point = gpd.GeoSeries([Point(start_lon, start_lat)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
        end_point = gpd.GeoSeries([Point(end_lon, end_lat)], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]

        # Check if points are in flooded area
        print("Checking if points are in flooded areas...")
        if flood_union is not None:
            if start_point.intersects(flood_union):
                print("❌ Start point is in flooded area")
                return jsonify({"error": "Start point is in a flooded area. Please choose a safe location."})
            if end_point.intersects(flood_union):
                print("❌ End point is in flooded area")
                return jsonify({"error": "End point is in a flooded area. Please choose a safe location."})
        print("✅ Both points are in safe areas")

        # Find nearest safe nodes using projected coordinates
        print("Finding nearest safe nodes...")
        safe_nodes_proj["dist_start"] = safe_nodes_proj.distance(start_point)
        safe_nodes_proj["dist_end"] = safe_nodes_proj.distance(end_point)
        
        if len(safe_nodes_proj) == 0:
            print("❌ No safe nodes found in the network")
            return jsonify({"error": "No safe nodes found in the area. Please try different locations."})
            
        start_node = safe_nodes_proj["dist_start"].idxmin()
        end_node = safe_nodes_proj["dist_end"].idxmin()

        print(f"✅ Found nearest nodes: start={start_node}, end={end_node}")

        if start_node == end_node:
            print("❌ Start and end points are too close")
            return jsonify({"error": "Start and end points are too close. Select locations farther apart."})

        try:
            # Check if both nodes are in the graph
            if start_node not in G or end_node not in G:
                print(f"❌ One or both nodes not in graph: start={start_node in G}, end={end_node in G}")
                return jsonify({"error": "One or both selected points cannot be connected to the road network."})
                
            # Use A* algorithm with custom heuristic
            print("\n🔍 Calculating shortest path...")
            try:
                print("Trying A* algorithm...")
                path = nx.astar_path(G, start_node, end_node, weight="length", heuristic=geographic_distance)
                print(f"✅ A* path found with {len(path)} nodes")
            except nx.NetworkXNoPath:
                # Try Dijkstra's algorithm as fallback
                print("⚠️ A* failed, trying Dijkstra's algorithm...")
                path = nx.dijkstra_path(G, start_node, end_node, weight="length")
                print(f"✅ Dijkstra path found with {len(path)} nodes")
            
            print("Converting path to coordinates...")
            path_nodes = [G.nodes[n] for n in path]
            coords = [(node['x'], node['y']) for node in path_nodes]
                
            if len(coords) < 2:
                print("❌ Path is too short")
                return jsonify({"error": "Evacuation path is too short."})
                
            print("Creating GeoJSON response...")
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)