import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from typing import Union, Dict, Any

def generate_shap_values(model, X_inf: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate SHAP values for multi-class model explanations based on predicted classes.
    
    Args:
        model: Trained multi-class model (should be compatible with TreeExplainer)
        X_inf: DataFrame with input features for explanation
        
    Returns:
        Dictionary containing SHAP values and feature importance data for predicted classes
    """
    try:
        # Validate inputs
        if X_inf is None or X_inf.empty:
            raise ValueError("Input data X_inf cannot be empty")
        
        # Get model predictions to determine which class to explain
        predictions = model.predict(X_inf)
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_inf)
        
        # Ensure we have multi-class output
        if not isinstance(shap_values, list):
            raise ValueError("Model does not appear to be multi-class")
        
        # Get SHAP values for each predicted class
        shap_values_for_predictions = []
        feature_importance_for_predictions = []
        
        for i, predicted_class in enumerate(predictions):
            # Get SHAP values for this prediction's class
            shap_values_instance = shap_values[predicted_class][i]
            shap_values_for_predictions.append(shap_values_instance.tolist())
            
        # Calculate overall feature importance across all predictions
        all_shap_values = np.vstack([shap_values[pred_class][i] for i, pred_class in enumerate(predictions)])
        feature_importance = np.abs(all_shap_values).mean(axis=0)
        
        # Prepare response data
        result = {
            "shap_values": shap_values_for_predictions,
            "predicted_classes": predictions.tolist(),
            "feature_names": X_inf.columns.tolist(),
            "feature_importance": feature_importance.tolist(),
            "expected_values": [float(explainer.expected_value[pred_class]) for pred_class in predictions],
            "num_classes": len(shap_values),
            "data_shape": X_inf.shape
        }
        
        return result
        
    except Exception as e:
        raise Exception(f"Error generating SHAP values: {str(e)}")

def generate_shap_plot(model, X_inf: pd.DataFrame, plot_type: str = "summary") -> str:
    """
    Generate SHAP plots for multi-class models based on predicted classes and return as base64 encoded image.
    
    Args:
        model: Trained multi-class model
        X_inf: DataFrame with input features
        plot_type: Type of plot ("summary", "waterfall")
        
    Returns:
        Base64 encoded string of the plot image
    """
    try:
        # Get model predictions
        predictions = model.predict(X_inf)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_inf)
        
        # Ensure we have multi-class output
        if not isinstance(shap_values, list):
            raise ValueError("Model does not appear to be multi-class")
        
        plt.figure(figsize=(12, 8))
        
        if plot_type == "summary":
            # For summary plot, show SHAP values for the most common predicted class
            most_common_class = max(set(predictions), key=list(predictions).count)
            shap.summary_plot(shap_values[most_common_class], X_inf, show=False)
            plt.title(f"SHAP Summary - Most Common Predicted Class: {most_common_class}")
            
        elif plot_type == "waterfall" and len(X_inf) > 0:
            # For waterfall, show the first instance with its predicted class
            predicted_class = predictions[0]
            shap.waterfall_plot(
                explainer.expected_value[predicted_class], 
                shap_values[predicted_class][0], 
                X_inf.iloc[0], 
                show=False
            )
            plt.title(f"SHAP Waterfall - Instance 0, Predicted Class: {predicted_class}")
            
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        plt.close()  # Clean up
        buffer.close()
        
        return plot_base64
        
    except Exception as e:
        plt.close()  # Ensure cleanup on error
        raise Exception(f"Error generating SHAP plot: {str(e)}")