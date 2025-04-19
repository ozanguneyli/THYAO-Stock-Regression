import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_predictions(y_test, y_pred, model_name, save_path=None):
    sorted_idx = np.argsort(y_test)
    y_test_sorted = y_test[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_sorted, label="Actual Values", marker='o')
    plt.plot(y_pred_sorted, label="Predicted Values", marker='x')
    plt.title(f"{model_name} - Actual vs Predicted Values")
    plt.xlabel("Sample (Sorted by Actual Value)")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_model_results(csv_path, save_path=None):
    """
    Visualize model performances from the results CSV file.
    Plots Test_R2 and Test_RMSE for each model.
    """
    df = pd.read_csv(csv_path)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Test_R2', color=color)
    ax1.bar(df['Model'], df['Test_R2'], color=color, alpha=0.7, label='Test_R2')
    ax1.tick_params(axis='y', labelcolor=color)
    plt.xticks(rotation=30)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Test_RMSE', color=color)
    ax2.plot(df['Model'], df['Test_RMSE'], color=color, marker='o', label='Test_RMSE')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Model Performance Comparison')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()