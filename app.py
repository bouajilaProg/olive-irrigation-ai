from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import ee
import os
import src.olive_brain as brain


# Initialize the FastAPI app
app = FastAPI(title="Olive Health AI", description="AI-driven irrigation recommendations")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def startup_event():
    try:
        ee.Initialize(project="arsii-open")
    except Exception as e:
        print("EE Init failed. Ensure auth.py was run.")
        raise e
        
    model_path = os.path.join(os.path.dirname(__file__), 'data', 'olive_model.pkl')
    if not os.path.exists(model_path):
        brain.train_brain()


@app.get("/api/predict")
def predict_health(lat: float, lon: float, temp: float = 25.0, variety: str = 'Languedoc'):
    try:
        coords = [lon, lat]
        point = ee.Geometry.Point(coords)
        
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        # Fetch latest imagery
        image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(point) \
            .filterDate(start_date, end_date) \
            .sort('CLOUDY_PIXEL_PERCENTAGE') \
            .first()


        if image is None:
            raise HTTPException(status_code=404, detail="No satellite data found for this location.")

        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        current_ndvi = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10
        ).get('NDVI').getInfo()
        
        if current_ndvi is None:
            raise HTTPException(status_code=404, detail="NDVI could not be calculated (possibly too cloudy).")
            
        brain_data = brain.predict_irrigation_need(current_ndvi, temp=temp, variety=variety)
        
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
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'explanation.html')
    with open(template_path, "r") as f:
        return f.read()

@app.get("/", response_class=HTMLResponse)

def get_ui():
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
    with open(template_path, "r") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    # This allows you to run it directly with `python app.py`
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
