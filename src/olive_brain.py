import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def train_brain():
    # 1. Load the real data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    df = pd.read_excel(os.path.join(data_path, 'Growth.xlsx'))


    # 2. Pre-process the 'Variety' (Text to Numbers)
    le = LabelEncoder()
    df['Variety_Encoded'] = le.fit_transform(df['Variety'])
    
    # 3. Features (X) and Target (y)
    # We use Temp, SPAD, and Canopy as they are the strongest signals
    X = df[['Temperature', 'Average_SPAD', 'Canopy Cover ', 'Variety_Encoded']]
    y = df['Irrigation Regime'] # AI learns to predict the water level needed

    # 4. Train the Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 5. Save the Brain and the Encoder
    joblib.dump(model, os.path.join(data_path, 'olive_model.pkl'))
    joblib.dump(le, os.path.join(data_path, 'variety_encoder.pkl'))
    print("Success: AI Brain trained on real Mediterranean field data!")




def predict_irrigation_need(ndvi, temp, variety='Languedoc'):
    # Load the trained brain
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    model = joblib.load(os.path.join(data_path, 'olive_model.pkl'))
    le = joblib.load(os.path.join(data_path, 'variety_encoder.pkl'))

    
    # Map NDVI from satellite to field parameters the model expects:
    canopy_cover = ndvi * 100
    average_spad = ndvi * 50 # heuristic mapping
    
    try:
        variety_encoded = le.transform([variety])[0]
    except:
        variety_encoded = 0 # Default if variety not found
    
    # Model predicts the irrigation regime
    input_df = pd.DataFrame([[temp, average_spad, canopy_cover, variety_encoded]], 
                            columns=['Temperature', 'Average_SPAD', 'Canopy Cover ', 'Variety_Encoded'])
    predicted_regime = model.predict(input_df)[0]
    
    # Map the predicted regime to a stress recommendation
    mapping = {
        '0%': {
            "level": "CRITICAL", 
            "value": 15, 
            "message": "CRITICAL: Extreme water stress. Your trees are reaching the wilting point. Irrigate immediately to prevent leaf loss and fruit damage.", 
            "quality": "CRITICAL"
        },
        '25%': {
            "level": "WARNING", 
            "value": 45, 
            "message": "STRESSED: Significant water shortage. The trees are starting to shut down to save water. We recommend starting an irrigation cycle within 24 hours.", 
            "quality": "STRESSED"
        },
        '50%': {
            "level": "STATUS", 
            "value": 75, 
            "message": "AVERAGE: Light moisture stress. Your trees are healthy but starting to use up their reserves. Monitor closely and schedule your next irrigation soon.", 
            "quality": "AVERAGE"
        },
        '100%': {
            "level": "STATUS", 
            "value": 100, 
            "message": "EXCELLENT: Ideal hydration. Your trees have plenty of water to grow high-quality olives. No action is required at this time.", 
            "quality": "EXCELLENT"
        }
    }



    
    res = mapping.get(predicted_regime, {"level": "UNKNOWN", "value": 0, "message": f"Unknown regime: {predicted_regime}"})
    return res

