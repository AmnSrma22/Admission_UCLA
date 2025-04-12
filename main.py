import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.predict_model import evaluate_model



if __name__ == '__main__':
    df = load_and_preprocess_data("dataset.csv")
    X = build_features(df)
    y = df['Admit_Chance']
    model, scaler, X_test_scaled, y_test = train_model(X, y)
    rmse, r2 = evaluate_model(model, X_test_scaled, y_test)
    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")