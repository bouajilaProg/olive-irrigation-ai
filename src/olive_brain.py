import os
import pandas as pd
import joblib

def predict_irrigation_need(ndvi, temp, variety='Languedoc'):
    """
    Predicts the irrigation regime based on satellite NDVI and local temperature.
    This function is MODEL-AGNOSTIC: It just loads 'olive_model.pkl' and calls .predict().
    Whether the model is Random Forest, SVM, or XGBoost doesn't matter as long as it supports .predict().
    """

    # Load the trained brain
    # Using '..' to go up one level from src/ to root, then into data/
    # Correction: __file__ is inside src/, so we need to go up to project root, then to data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data')
    
    model_path = os.path.join(data_path, 'olive_model.pkl')
    encoder_path = os.path.join(data_path, 'variety_encoder.pkl')

    if not os.path.exists(model_path):
        return {"level": "ERROR", "message": "AI Model not found. Please train a model first!"}

    model = joblib.load(model_path)
    le = joblib.load(encoder_path)

    
    # Map NDVI from satellite to field parameters the model expects:
    # These heuristic mappings must match what was used during training (or be close approximations)
    canopy_cover = ndvi * 100
    average_spad = ndvi * 50 
    
    try:
        variety_encoded = le.transform([variety])[0]
    except:
        variety_encoded = 0 # Default if variety not found (e.g. 'unknown')
    
    # Model predicts the irrigation regime
    # The feature order MUST match training: ['Temperature', 'Average_SPAD', 'Canopy Cover ', 'Variety_Encoded']
    input_df = pd.DataFrame([[temp, average_spad, canopy_cover, variety_encoded]], 
                            columns=['Temperature', 'Average_SPAD', 'Canopy Cover ', 'Variety_Encoded'])
    
    try:
        predicted_regime = model.predict(input_df)[0]
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
