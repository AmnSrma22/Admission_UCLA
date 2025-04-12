from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test_scaled, y_test):
    predictions = model.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)
    return rmse, r2