import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import joblib
import os

def train_brain():
    """Train a default RandomForest on Growth.xlsx and save model + encoder into data/.
    """
    # 1. Load the real data
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
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

    # 5. Save the Brain and the Encoder (Generic name so app doesn't care)
    joblib.dump(model, os.path.join(data_path, 'olive_model.pkl'))
    joblib.dump(le, os.path.join(data_path, 'variety_encoder.pkl'))
    print("Success: AI Brain (Random Forest) trained and saved!")

def optimize_and_train():
    """Run a grid search to find better hyperparameters then save the best model.
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    df = pd.read_excel(os.path.join(data_path, 'Growth.xlsx'))

    le = LabelEncoder()
    df['Variety_Encoded'] = le.fit_transform(df['Variety'])

    X = df[['Temperature', 'Average_SPAD', 'Canopy Cover ', 'Variety_Encoded']]
    y = df['Irrigation Regime']

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    print("🚀 Starting Hyperparameter Optimization (Grid Search) for Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    
    grid_search.fit(X, y)

    print("\n✅ Best Parameters Found:")
    print(grid_search.best_params_)
    
    # Save the Best Model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, os.path.join(data_path, 'olive_model.pkl'))
    joblib.dump(le, os.path.join(data_path, 'variety_encoder.pkl'))
    
    print("\n🏆 Optimized AI Brain (Random Forest) saved!")
