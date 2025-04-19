from src.data_preprocessing import load_and_preprocess_data
from src.train_model import run_training_pipeline
from src.plot_utils import plot_model_results
import os

def main():
    input_path = r"C:\Users\ozang\OneDrive\Masaüstü\THYAO_Project\data\THYAO_Regression_Data.csv"
    output_path = r"C:\Users\ozang\OneDrive\Masaüstü\THYAO_Project\data\preprocessed_THYAO.csv"
    load_and_preprocess_data(input_path, output_path)
    run_training_pipeline(output_path)

    # Save model results plot to images directory
    images_dir = os.path.join(os.path.dirname(output_path), "..", "images")
    os.makedirs(images_dir, exist_ok=True)
    results_csv_path = os.path.join(os.path.dirname(output_path), "..", "outputs", "model_results.csv")
    plot_path = os.path.join(images_dir, "model_performance_comparison.png")
    plot_model_results(results_csv_path, save_path=plot_path)

if __name__ == "__main__":
    main()