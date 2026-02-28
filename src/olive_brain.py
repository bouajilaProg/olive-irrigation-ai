import os
import pandas as pd
import joblib

def predict_irrigation_need(ndvi, temp, variety='Languedoc'):
    """
    Predicts the irrigation regime based on satellite NDVI and local temperature.
    This function is MODEL-AGNOSTIC: It just loads 'olive_model.pkl' and calls .predict().
    
    NOTE: 'variety' parameter is kept for API compatibility but is NO LONGER USED in the model.
    """

    # Load the trained brain
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data')
    
    model_path = os.path.join(data_path, 'olive_model.pkl')

    if not os.path.exists(model_path):
        return {"level": "ERROR", "message": "AI Model not found. Please train a model first!"}

    # Load the artifact (could be just the model, or a dict with model+encoder)
    loaded_artifact = joblib.load(model_path)
    model = None
    encoder = None
    
    if isinstance(loaded_artifact, dict) and "model" in loaded_artifact:
        model = loaded_artifact["model"]
        encoder = loaded_artifact.get("encoder")
    else:
        model = loaded_artifact
        encoder = None
    
    # Map NDVI from satellite to field parameters the model expects:
    canopy_cover = ndvi * 100
    average_spad = ndvi * 50 
    
    # Reverted to base features only (No interaction terms)
    # Feature order MUST match trainer: ['Temperature', 'Average_SPAD', 'Canopy Cover ']
    feature_names = [
        'Temperature', 
        'Average_SPAD', 
        'Canopy Cover '
    ]
    
    input_df = pd.DataFrame([[
        temp, 
        average_spad, 
        canopy_cover
    ]], columns=feature_names)
    
    try:
        prediction = model.predict(input_df)[0]
        if encoder:
            # If an encoder was saved, we must decode the numeric prediction back to string label
            predicted_regime = encoder.inverse_transform([prediction])[0]
        else:
            predicted_regime = prediction
    except Exception as e:
        return {"level": "ERROR", "message": f"Prediction failed: {str(e)}"}
    
    # Map the predicted regime to a stress recommendation
    mapping = {
        '0%': {
            "level": "CRITICAL", 
            "value": 15, 
            "message": "🚨 EMERGENCY: Severe drought stress detected. Your trees are losing moisture faster than they can absorb it. If you don't irrigate today, you may see permanent branch dieback and fruit shriveling. Open all valves now.", 
            "quality": "CRITICAL"
        },
        '25%': {
            "level": "WARNING", 
            "value": 45, 
            "message": "⚠️ ACTION REQUIRED: Moderate stress. The trees are trying to survive by closing their leaf pores. This stops growth and oil production. Start an irrigation cycle within 12-24 hours to recover crop quality.", 
            "quality": "STRESSED"
        },
        '50%': {
            "level": "STATUS", 
            "value": 75, 
            "message": "💧 MONITORING: Your grove is currently fine, but moisture is becoming limited in the top 30cm of soil. Plan to irrigate in the next 2-3 days unless rain is forecast. Growth is still steady.", 
            "quality": "FAIR"
        },
        '100%': {
            "level": "STATUS", 
            "value": 100, 
            "message": "✅ OPTIMAL: Perfectly hydrated. The trees have high chlorophyll activity and are actively growing. Your water management is excellent. No irrigation needed; save your water!", 
            "quality": "OPTIMAL"
        }
    }
    
    res = mapping.get(predicted_regime, {"level": "UNKNOWN", "value": 0, "message": f"Unknown regime: {predicted_regime}"})
    return res
