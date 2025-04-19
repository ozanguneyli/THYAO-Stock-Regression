import os
import pandas as pd
from src.model_utils import train_and_evaluate
from src.plot_utils import plot_predictions

def run_training_pipeline(preprocessed_path):
    df = pd.read_csv(preprocessed_path)
    target_col = "Highest Price"
    X = df.drop(columns=[target_col, "Month"])
    y = df[target_col]

    results_df, models, X_train, X_test, y_train, y_test = train_and_evaluate(X, y)

    # Klasörleri oluştur
    images_dir = os.path.join(os.path.dirname(preprocessed_path), "..", "images")
    models_dir = os.path.join(os.path.dirname(preprocessed_path), "..", "models")
    outputs_dir = os.path.join(os.path.dirname(preprocessed_path), "..", "outputs")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # Grafik sonuçlarını kaydet
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        image_path = os.path.join(images_dir, f"{name}_prediction.png")
        plot_predictions(y_test.values, y_pred, name, save_path=image_path)

    # Sonuçları Test_R2'ye göre sırala ve kaydet
    results_df = results_df.sort_values(by="Test_R2", ascending=False)
    results_csv_path = os.path.join(outputs_dir, "model_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(results_df)

    # En iyi modeli kaydet
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    import joblib
    best_model_path = os.path.join(models_dir, f"{best_model_name}.pkl")
    joblib.dump(best_model, best_model_path)