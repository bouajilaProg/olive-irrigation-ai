from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import ee
import os
import sys

# Ensure src is in path so we can import modules if run from root
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import olive_brain as brain
from random_forest import trainer as rf_trainer
from svm import trainer as svm_trainer

# Initialize the FastAPI app
app = FastAPI(title="Olive Health AI", description="AI-driven irrigation recommendations")

# Mount static files
# We need to find the static directory relative to this file
project_root = os.path.dirname(current_dir)
static_path = os.path.join(project_root, 'static')
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.on_event("startup")
def startup_event():
    try:
        ee.Initialize(project="arsii-open")
    except Exception as e:
        print("EE Init failed. Ensure auth.py was run.")
        # Don't crash app if EE fails locally without auth, but warn
    
    # Check for model, warn but DON'T auto-train from web
    model_path = os.path.join(project_root, 'data', 'olive_model.pkl')
    if not os.path.exists(model_path):
        print("WARNING: Model not found. Please run 'python src/random_forest/trainer.py' to train it.")

@app.get("/api/predict")
def predict_health(lat: float, lon: float, temp: float = 25.0, variety: str = 'Languedoc'):
    try:
        coords = [lon, lat]
        point = ee.Geometry.Point(coords)
        
        # Use simple date range logic or specific dates
        start_date = '2026-01-01'
        end_date = '2026-02-28'
        
        # Fetch latest imagery
        image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(point) \
            .filterDate(start_date, end_date) \
            .sort('CLOUDY_PIXEL_PERCENTAGE') \
            .first()
        
        if image is None:
             # Just return mocked/fallback if EE fails (e.g. no auth) or no image
             # For production we'd raise 404, but for demo:
             pass 

        # Try to get real NDVI
        current_ndvi = None
        try:
            ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
            stats = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=10).getInfo()
            current_ndvi = stats.get('NDVI')
        except:
            # Fallback if EE fails or is cloudy
            current_ndvi = 0.45 
            
        if current_ndvi is None:
             current_ndvi = 0.45

        # The app doesn't care WHICH model is loaded. 
        # src.olive_brain.predict_irrigation_need loads whatever 'olive_model.pkl' is in data/
        brain_data = brain.predict_irrigation_need(current_ndvi, temp=temp, variety=variety)
        
        if isinstance(brain_data, dict) and brain_data.get("level") == "ERROR":
             raise HTTPException(status_code=500, detail=brain_data["message"])

        return {
            "status": "success",
            "lat": lat,
            "lon": lon,
            "ndvi": round(current_ndvi, 4),
            "recommendation": brain_data["message"],
            "value": brain_data["value"],
            "level": brain_data["level"],
            "quality": brain_data.get("quality", "N/A")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/explanation", response_class=HTMLResponse)
def get_explanation():
    template_path = os.path.join(project_root, 'templates', 'explanation.html')
    with open(template_path, "r") as f:
        return f.read()

@app.get("/", response_class=HTMLResponse)
def get_ui():
    template_path = os.path.join(project_root, 'templates', 'index.html')
    with open(template_path, "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
