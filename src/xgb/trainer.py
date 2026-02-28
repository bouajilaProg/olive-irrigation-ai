import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import json
import datetime
import shutil

def save_run_artifacts(model, encoder, score, feature_names):
    """Saves model details, accuracy score, and feature importance to runs/ and updates best/ if needed."""
    
    model_name = type(model).__name__
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_{model_name}"
    
    # 1. Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(base_dir, 'data')
    runs_path = os.path.join(base_dir, 'runs')
    run_path = os.path.join(runs_path, run_id)
    best_path = os.path.join(runs_path, 'best')
    
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(best_path, exist_ok=True)

    # 2. Get Feature Importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importances = [float(x) for x in importances]
        importance_dict = dict(zip(feature_names, importances))
        importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
    else:
        importance_dict = {"error": "Model does not provide feature importances"}

    # 3. Build Report
    report = {
        "model_name": model_name,
        "date_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "score": score,
        "parameters": model.get_params(),
        "feature_importance": importance_dict
    }

    # 4. Save to run folder
    report_file = os.path.join(run_path, "model_report.json")
    model_file = os.path.join(run_path, "olive_model.pkl")
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    
    # Save dictionary with model and encoder
    artifact = {"model": model, "encoder": encoder}
    joblib.dump(artifact, model_file)

    # 5. Check if BEST
    is_best = False
    best_report_file = os.path.join(best_path, "model_report.json")
    if os.path.exists(best_report_file):
        with open(best_report_file, 'r') as f:
            best_data = json.load(f)
            if score > best_data.get("score", 0):
                is_best = True
    else:
        is_best = True

    if is_best:
        print("--- NEW BEST MODEL DETECTED ---")
        shutil.copy2(report_file, best_report_file)
        shutil.copy2(model_file, os.path.join(best_path, "olive_model.pkl"))
        # Also update the main model for the app
        shutil.copy2(model_file, os.path.join(data_path, "olive_model.pkl"))
        # Update the main report for the app
        shutil.copy2(report_file, os.path.join(data_path, "model_report.json"))
        print(f"Best model updated in: {best_path}")

    print(f"Run artifacts saved to: {run_path}")
    print("--- Feature Importance ---")
    for feat, val in importance_dict.items():
        print(f"{feat}: {val:.4f}")
    print("--------------------------")


def prepare_data(df):
    """Selects relevant features and encodes target."""
    feature_names = ['Temperature', 'Average_SPAD', 'Canopy Cover ']
    X = df[feature_names]
    y_raw = df['Irrigation Regime']
    
    # Encode target labels (e.g., '0%', '25%' -> 0, 1)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    return X, y, le, feature_names

def train_brain():
    """Train a default XGBoost on Growth.xlsx and save model into data/."""
    print("--- Starting XGBoost Training ---")
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    df = pd.read_excel(os.path.join(data_path, 'Growth.xlsx'))

    X, y, le, feature_names = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model Training Score: {accuracy:.2%}")
    print("---------------------------------------")

    # Re-train on full dataset
    model.fit(X, y)
    save_run_artifacts(model, le, accuracy, feature_names)
    print("Training process complete.")

def optimize_and_train():
    """Run a grid search for XGBoost then save the best model."""
    print("--- Starting Hyperparameter Optimization (XGBoost) ---")
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    df = pd.read_excel(os.path.join(data_path, 'Growth.xlsx'))

    X, y, le, feature_names = prepare_data(df)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 6, 10],
        'subsample': [0.8, 1.0]
    }

    xgb_clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid,  
                               cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
    
    grid_search.fit(X, y)

    print("Best Parameters Found:")
    print(grid_search.best_params_)
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    print("-----------------------------------------------------------")
    
    best_model = grid_search.best_estimator_
    save_run_artifacts(best_model, le, grid_search.best_score_, feature_names)
    print("Optimization process complete.")

if __name__ == "__main__":
    train_brain()
