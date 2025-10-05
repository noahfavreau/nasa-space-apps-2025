"""
SHAP Explanation Generator Module

This module provides intelligent SHAP explanations for exoplanet classification models.
Generates feature importance analysis and visualizations for model predictions.

Key Features:
- Multi-class SHAP value generation
- Automatic plot generation (summary, waterfall)
- API-friendly error handling
- Base64 encoded visualizations for web use
- Graceful fallback when SHAP is not available

Quick Usage:
from shap_generator import generate_shap_analysis

# Generate comprehensive SHAP analysis
result = generate_shap_analysis(model, input_data)

if result['success']:
    shap_values = result['shap_values']
    feature_importance = result['feature_importance']
    plot_base64 = result['summary_plot']

Output: Returns SHAP values, feature importance, and visualization plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from typing import Union, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP, provide fallback if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP library not available. Install with: pip install shap")

def generate_shap_analysis(model, X_inf, include_plots=True):
    """
    Generate comprehensive SHAP analysis for exoplanet classification.
    
    Parameters:
    -----------
    model : sklearn model
        Trained classification model
    X_inf : pd.DataFrame, dict, or list
        Input data for SHAP analysis
    include_plots : bool
        Whether to generate visualization plots
        
    Returns:
    --------
    dict
        Dictionary with SHAP analysis results including values, importance, and plots
    """
    try:
        # Check if SHAP is available
        if not SHAP_AVAILABLE:
            return {
                'success': False,
                'error': 'SHAP library not installed. Please install with: pip install shap',
                'shap_values': None,
                'feature_importance': None,
                'plots': None
            }
        
        # Convert input to DataFrame if needed
        if isinstance(X_inf, dict):
            X_inf = pd.DataFrame([X_inf])
        elif isinstance(X_inf, list):
            X_inf = pd.DataFrame(X_inf)
        elif not isinstance(X_inf, pd.DataFrame):
            raise ValueError("Input must be DataFrame, dict, or list")
        
        # Validate input
        if X_inf.empty:
            return {
                'success': False,
                'error': 'Input data is empty',
                'shap_values': None,
                'feature_importance': None,
                'plots': None
            }
        
        # Generate SHAP values
        shap_data = _generate_shap_values(model, X_inf)
        
        # Generate plots if requested
        plots = {}
        if include_plots:
            plots = _generate_all_plots(model, X_inf)
        
        return {
            'success': True,
            'error': None,
            'shap_values': shap_data['shap_values'],
            'predicted_classes': shap_data['predicted_classes'],
            'feature_names': shap_data['feature_names'],
            'feature_importance': shap_data['feature_importance'],
            'expected_values': shap_data['expected_values'],
            'num_classes': shap_data['num_classes'],
            'data_shape': shap_data['data_shape'],
            'plots': plots
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'SHAP analysis failed: {str(e)}',
            'shap_values': None,
            'feature_importance': None,
            'plots': None
        }

def _generate_shap_values(model, X_inf):
    """
    Internal function to generate SHAP values for multi-class model.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library not available")
    
    # Get model predictions
    predictions = model.predict(X_inf)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_inf)
    
    # Ensure we have multi-class output
    if not isinstance(shap_values, list):
        raise ValueError("Model does not appear to be multi-class")
    
    # Get SHAP values for each predicted class
    shap_values_for_predictions = []
    
    for i, predicted_class in enumerate(predictions):
        # Get SHAP values for this prediction's class
        shap_values_instance = shap_values[predicted_class][i]
        shap_values_for_predictions.append(shap_values_instance.tolist())
    
    # Calculate overall feature importance across all predictions
    all_shap_values = np.vstack([shap_values[pred_class][i] for i, pred_class in enumerate(predictions)])
    feature_importance = np.abs(all_shap_values).mean(axis=0)
    
    return {
        "shap_values": shap_values_for_predictions,
        "predicted_classes": predictions.tolist(),
        "feature_names": X_inf.columns.tolist(),
        "feature_importance": feature_importance.tolist(),
        "expected_values": [float(explainer.expected_value[pred_class]) for pred_class in predictions],
        "num_classes": len(shap_values),
        "data_shape": X_inf.shape
    }

def _generate_all_plots(model, X_inf):
    """
    Internal function to generate all SHAP visualization plots.
    """
    if not SHAP_AVAILABLE:
        return {
            'summary_plot': None,
            'waterfall_plot': None,
            'error': 'SHAP library not available'
        }
    
    plots = {}
    
    try:
        plots['summary_plot'] = generate_shap_plot(model, X_inf, plot_type="summary")
    except Exception as e:
        plots['summary_plot'] = None
        plots['summary_plot_error'] = str(e)
    
    try:
        plots['waterfall_plot'] = generate_shap_plot(model, X_inf, plot_type="waterfall")
    except Exception as e:
        plots['waterfall_plot'] = None
        plots['waterfall_plot_error'] = str(e)
    
    return plots

def generate_shap_values(model, X_inf):
    """
    Generate SHAP values for multi-class model explanations.
    
    Parameters:
    -----------
    model : sklearn model
        Trained multi-class model
    X_inf : pd.DataFrame
        Input features for explanation
        
    Returns:
    --------
    dict
        Dictionary containing SHAP values and feature importance data
    """
    try:
        # Validate inputs
        if X_inf is None or X_inf.empty:
            raise ValueError("Input data X_inf cannot be empty")
        
        return _generate_shap_values(model, X_inf)
        
    except Exception as e:
        raise Exception(f"Error generating SHAP values: {str(e)}")

def generate_shap_plot(model, X_inf, plot_type="summary"):
    """
    Generate SHAP visualization plots.
    
    Parameters:
    -----------
    model : sklearn model
        Trained multi-class model
    X_inf : pd.DataFrame
        Input features
    plot_type : str
        Type of plot ("summary", "waterfall")
        
    Returns:
    --------
    str
        Base64 encoded string of the plot image
    """
    try:
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not available")
        
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

def check_shap_availability():
    """
    Check if SHAP library is available and return status.
    
    Returns:
    --------
    dict
        Dictionary with availability status and installation instructions
    """
    return {
        'available': SHAP_AVAILABLE,
        'message': 'SHAP library is available' if SHAP_AVAILABLE else 'SHAP library not installed',
        'install_command': 'pip install shap' if not SHAP_AVAILABLE else None
    }

def get_feature_importance(model, X_inf, top_n=10):
    """
    Get top N most important features from SHAP analysis.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_inf : pd.DataFrame
        Input data
    top_n : int
        Number of top features to return
        
    Returns:
    --------
    dict
        Dictionary with top features and their importance scores
    """
    try:
        result = generate_shap_analysis(model, X_inf, include_plots=False)
        
        if not result['success']:
            return {
                'success': False,
                'error': result['error'],
                'top_features': None
            }
        
        # Get feature importance and names
        importance = np.array(result['feature_importance'])
        feature_names = result['feature_names']
        
        # Get top N features
        top_indices = np.argsort(importance)[-top_n:][::-1]
        top_features = [
            {
                'feature': feature_names[i],
                'importance': float(importance[i]),
                'rank': rank + 1
            }
            for rank, i in enumerate(top_indices)
        ]
        
        return {
            'success': True,
            'error': None,
            'top_features': top_features,
            'total_features': len(feature_names)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Feature importance analysis failed: {str(e)}',
            'top_features': None
        }