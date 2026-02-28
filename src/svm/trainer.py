import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
import os

def train_brain():
    """Train a default SVM on Growth.xlsx, report accuracy, and save.
    """
    # 1. Load the real data
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    df = pd.read_excel(os.path.join(data_path, 'Growth.xlsx'))

    # 2. Pre-process the 'Variety' (Text to Numbers)
    le = LabelEncoder()
    df['Variety_Encoded'] = le.fit_transform(df['Variety'])
    
    # 3. Features (X) and Target (y)
    X = df[['Temperature', 'Average_SPAD', 'Canopy Cover ', 'Variety_Encoded']]
    y = df['Irrigation Regime']

    # 4. Split for Scoring
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Train and Score
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print("-" * 30)
    print(f"✅ AI Brain Training Score: {accuracy:.2%}")
    print("-" * 30)

    # 6. Re-train on FULL dataset for the clinical model and save
    model.fit(X, y)
    joblib.dump(model, os.path.join(data_path, 'olive_model.pkl'))
    joblib.dump(le, os.path.join(data_path, 'variety_encoder.pkl'))
    print("🏆 Optimized Model saved to data/olive_model.pkl")


def optimize_and_train():
    """Run a grid search to find better hyperparameters for SVM then save the best model.
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    df = pd.read_excel(os.path.join(data_path, 'Growth.xlsx'))

    le = LabelEncoder()
    df['Variety_Encoded'] = le.fit_transform(df['Variety'])

    X = df[['Temperature', 'Average_SPAD', 'Canopy Cover ', 'Variety_Encoded']]
    y = df['Irrigation Regime']

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 1]
    }

    print("🚀 Starting Hyperparameter Optimization (Grid Search) for SVM...")
    svc = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    
    grid_search.fit(X, y)

    print("\n✅ Best Parameters Found:")
    print(grid_search.best_params_)
    
    # Save the Best Model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, os.path.join(data_path, 'olive_model.pkl'))
    joblib.dump(le, os.path.join(data_path, 'variety_encoder.pkl'))
    
    print("\n🏆 Optimized AI Brain (SVM) saved!")

if __name__ == "__main__":
    train_brain()

