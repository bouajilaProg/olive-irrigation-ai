import ee
import os
# Fix import to use the module in the same directory
try:
    import olive_brain as brain
except ImportError:
    import src.olive_brain as brain

# Initialize Earth Engine
try:
    ee.Initialize(project="arsii-open")
except Exception as e:
    print("Please run auth.py first to authenticate Earth Engine!")
    raise e

# Ensure the model is trained before predicting
data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
if not os.path.exists(os.path.join(data_path, 'olive_model.pkl')) or not os.path.exists(os.path.join(data_path, 'variety_encoder.pkl')):
    print("Model not found. Training the AI brain first...")
    brain.train_brain()


# 1. Setup the location in Mahdia
# Remember: [Longitude, Latitude]
coords = [10.771795360694064, 35.697475786818984]
point = ee.Geometry.Point(coords)

# 2. Fetch and Process Data
def get_mahdia_health():
    print(f"Checking Mahdia Olive Grove at {coords}...")
    
    # Get the latest Sentinel-2 imagery (last 30 days)
    image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(point) \
        .filterDate('2026-02-01', '2026-02-28') \
        .sort('CLOUDY_PIXEL_PERCENTAGE') \
        .first()

    # Calculate NDVI
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # Get the numerical value
    value = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=10
    ).get('NDVI').getInfo()
    
    return value

# 3. Run the "Brain"
try:
    current_ndvi = get_mahdia_health()
    
    if current_ndvi is None:
        print("Could not get NDVI. It might be too cloudy or out of bounds. Using mock data for the demo!")
        current_ndvi = 0.45
        
    print(f"Real-time NDVI from Satellite: {current_ndvi:.4f}")
    
    # Call your module's logic
    # Fixed parameter name from 'temperature' to 'temp'
    recommendation = brain.predict_irrigation_need(current_ndvi, temp=22)
    
    print("-" * 30)
    print(f"RESULT: {recommendation}")
    print("-" * 30)

except Exception as e:
    print(f"Error fetching data: {e}")
    print("TIP: If you see 'None', it might be too cloudy. Use mock data for the demo!")
