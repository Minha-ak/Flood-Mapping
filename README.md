# 🌊 Flood Mapping  
**Advanced Flood Detection and  Situational Awareness Using Sentinel-1 and Sentinel-2 Satellite Imagery:  A Deep Learning Approach for Enhanced Disaster Response**  


---

## 🔍 Project Overview

In response to the rising impact of climate-related disasters, especially floods, this project provides a deep learning–based solution for rapid and reliable flood detection. It combines data from **Sentinel-1 SAR** and **Sentinel-2 optical** imagery, processed through a **U-Net segmentation model**, with **geospatial tools** to extract flooded regions and identify safe evacuation paths.

The system leverages:
- Semantic segmentation via deep learning  
- Raster-to-vector geospatial transformation  
- OpenStreetMap-based road network analysis  
- Web-based visualization with interactive flood mapping  

This end-to-end pipeline supports both visual analysis and practical planning during flood events.

---

## 📌 Key Features

- 🛰️ **Multi-sensor input**: Sentinel-1 (SAR) & Sentinel-2 (optical RGB/NIR)  
- 🧠 **U-Net-based deep learning model** for binary flood segmentation  
- 🗺️ **Georeferenced flood mask conversion** to polygon shapefiles  
- 🛣️ **Evacuation route detection** using OSM road networks and NetworkX  
- 🌐 **Interactive web map frontend** using Leaflet + Flask backend  
- 🧪 Evaluation metrics: Dice, IoU, Accuracy (for segmentation)  

---

## 📁 Dataset

### 🔹 Sources

- **[S1S2-Water Dataset (Zenodo)]**  
  Global dataset for water-body semantic segmentation.

### 📦 Structure

Each dataset folder contains:

- **Sentinel-1 (C-band SAR)** — Radar imagery to detect water bodies  
- **Sentinel-2 (MSI Optical)** — High-res RGB/NIR images for mask generation  
- **Natural Earth** — River and lake shapefiles to remove permanent water bodies  
- **OpenStreetMap (via OSMnx)** — For road/path network extraction  

### 🔧 Preprocessing Steps

- Resampling and alignment of S1/S2 resolutions  
- Normalization of pixel values  
- Patch extraction (128×128)  
- Binary mask generation from S1 and S2
- Exclusion of permanent water bodies  

---

## 🧠 Model Architecture

| Component        | Details                              |
|------------------|--------------------------------------|
| Model Type       | U-Net (encoder-decoder)              |
| Input Channels   | Sentinel-1 (VV, VH) + Sentinel-2 (RGB/NIR) |
| Output           | Binary flood mask (0: non-flood, 1: flood) |
| Loss Function    | Dice Loss + Binary Crossentropy      |
| Framework        | Tensorflow                            |
| Metrics          | IoU, Dice Score, Accuracy            |

---

## 🗺️ Geospatial Processing

- **Georeferencing**: Align predicted flood masks (PNG) to original image coordinates using Sentinel-1 `.tif` metadata  
- **Vectorization**: Convert binary flood mask into shapefiles (polygons)  
- **Mask Filtering**: Remove areas overlapping known water bodies  
- **Coordinate Systems**: Uses EPSG:4326 (WGS84) for interoperability  

---

## 🛣️ Evacuation Pathfinding

- **Road Network Extraction**: Uses OSMnx to retrieve streets within the image extent  
- **Flood Overlay**: Flooded roads are identified by intersection with flood polygons  
- **Routing Engine**: NetworkX computes shortest path from start to destination while avoiding flooded nodes using A* and Djikshtra's Algoriy=thm  

> Note: Start/end locations are selected interactively on the frontend map.

---

## 🌐 Web Interface

Built with:
- **Frontend**: Vite + JavaScript + Leaflet.js  
- **Backend**: Flask (Python API)  

### Features:
- Folder selection from dropdown (e.g., "sample")  
- Real-time flood polygon overlay on interactive map  
- Click-to-select start and end points  
- Evacuation path displayed over the road network  
- Upload and visualize any new prediction masks  

---

## ⚙️ Installation

# 🔧 Frontend Setup
```bash
cd frontend
npm install        # Install all frontend dependencies
npm run dev        # Start the frontend development server
```
# 🔧 Backend Setup
```bash
cd ../backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py             # Start the Flask server

