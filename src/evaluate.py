""" 
Evaluate utilities for the salary prodiction model
Functions - Evaluate_model,Plot_predictions,print_observations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def evaluate_model(y_true, y_pred,title: str = 'Model Evalutaion') -> dict:
    """
     pass y_true, and y_pred to get back the dict ie
     Returns a dict with mae,rmse,r2score  
    """
    mae = mean_absolute_error(y_true,y_pred)
    rmse = np.sqrt(mean_squared_error(y_true,y_pred))
    r2 = r2_score(y_true, y_pred)

    return{'mae':mae,'rmse':rmse,'r2':r2}


def plot_predictions(y_true,y_pred, save_path: str=None):
  
    """
    plots  actual vs predicted salary scatter plot as a histogram for the residuals.

    """
    fig,axes = plt.subplots(1,2, figsize=(12,6))

  # actual vs predicted scatter plot.
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
    limit = max(y_true.max(), y_pred.max()) * 1.05
    axes[0].plot([0, limit], [0, limit], 'r--', linewidth=1.5, label='Perfect prediction')
    axes[0].set_xlabel('Actual salary (USD)')
    axes[0].set_ylabel('Predicted salary (USD)')
    axes[0].set_title('Actual vs predicted salary')
    axes[0].legend()

    # Hist for residuals (to see the distribution of the residuals)
    residuals = y_true - y_pred
    axes[1].hist(residuals, bins=60, color='coral', edgecolor='white')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('Residual (Actual - predicted)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Residual distribution')

    plt.suptitle('Model evaluation')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f'Plot saved to {save_path}')

    plt.show()

def print_observations(metrics: dict):
    """
        just prints observations based on model metrics, r2, mae, rmse.
    """
    r2 = metrics['r2']
    rmse = metrics['rmse']

    if r2 > 0.7:
        print(f"An r2 of {r2:.3f} is very strong, Good job!!!")
    elif r2 > 0.5:
        print(f"An r2 of {r2:.3f} is okay, there is room for improvement but this is nice, good job!!!")
    else: 
        print(f"An r2 of {r2:.3f} is relatively low however this is common for salary prediction\n")
        print(f"Note your model only explains {r2 * 100:.1f} of variance in salary.")

    print(f"RMSE of ${rmse:,.0f} means the model's prediction id off by ${rmse:,.0f} from the true salary")