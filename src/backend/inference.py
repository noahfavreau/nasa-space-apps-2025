"""
Exoplanet Classification Inference Pipeline

This script provides a complete inference pipeline for the ensemble model
that classifies celestial objects as exoplanets or not.

The pipeline includes:
- 4 base models (CatBoost, LightGBM, XGBoost, TabNet) with 10-fold cross-validation
- A meta-model that combines predictions from base models
- Support for both single predictions and batch predictions
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from typing import Union, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Model imports
try:
    import catboost as cb
    from catboost import CatBoostClassifier
except ImportError:
    print("Warning! CatBoost has not been installed. Please install it with: pip install catboost")
    cb = None

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError:
    print("Warning! LightGBM has not been installed. Please install it with: pip install lightgbm")
    lgb = None

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    print("Warning! XGBoost has not been installed. Please install it with: pip install xgboost")
    xgb = None

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
except ImportError:
    print("Warning! PyTorch TabNet has not been installed. Please install it with: pip install pytorch-tabnet")
    TabNetClassifier = None

from sklearn.preprocessing import StandardScaler


class ExoplanetClassifier:
    """
    Ensemble classifier for exoplanet detection.
    This class loads all trained models and provides methods for inference.
    """
    
    def __init__(self, model_dir: str = "."):
        """
        Set up the classifier with all trained models.
        Args:
            model_dir: Directory where the trained models are stored
        """
        self.model_dir = model_dir
        self.best_params = self._load_best_params()
        self.meta_model = self._load_meta_model()
        self.base_models = self._load_base_models()
        self.scaler = None  # scaler will be set if needed for TabNet
        
        print(f"Loaded {len(self.base_models)} base models and meta-model") # print information about the loaded models
        print("Available model types:", list(self.base_models.keys()))
    
    def _load_best_params(self) -> Dict[str, Any]:
        """this method load the best parameters for each model type."""
        params_path = os.path.join(self.model_dir, "best_params.json")
        with open(params_path, 'r') as f:
            return json.load(f)
    
    def _load_meta_model(self):
        """loads the meta-model"""
        meta_model_path = os.path.join(self.model_dir, "meta_model.joblib")
        return joblib.load(meta_model_path) # joblib to load meta-model, apparently better than pickle
    
    def _load_base_models(self) -> Dict[str, List]:
        """Load all base models for each algorithm."""
        models = {}
        
        # load catboost models
        if cb is not None:
            models['catboost'] = []
            for i in range(1, 11):
                model_path = os.path.join(self.model_dir, "catboost", f"fold_{i}.cbm")
                if os.path.exists(model_path):
                    model = CatBoostClassifier()
                    model.load_model(model_path)
                    models['catboost'].append(model)
        
        # load luightgbm model
        if lgb is not None:
            models['lightgbm'] = []
            for i in range(1, 11):
                model_path = os.path.join(self.model_dir, "lightgbm", f"fold_{i}.txt")
                if os.path.exists(model_path):
                    model = LGBMClassifier()
                    model.booster_ = lgb.Booster(model_file=model_path)
                    models['lightgbm'].append(model)
        
        # load XGBoost model
        if xgb is not None:
            models['xgboost'] = []
            for i in range(1, 11):
                model_path = os.path.join(self.model_dir, "xgboost", f"fold_{i}.json")
                if os.path.exists(model_path):
                    model = XGBClassifier()
                    model.load_model(model_path)
                    models['xgboost'].append(model)
        
        # load tabnet modelsw
        if TabNetClassifier is not None:
            models['tabnet'] = []
            for i in range(1, 11):
                model_path = os.path.join(self.model_dir, "tabnet", f"fold_{i}.zip")
                if os.path.exists(model_path):
                    model = TabNetClassifier()
                    model.load_model(model_path)
                    models['tabnet'].append(model)
        
        return models
    
    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Gets predictions from all the base models and then returns an array with the predictions from each of them
        Args:
            X: Features

        Returns:
            Array of shape (n_samples, n_models) with predictions from each model
        """
        all_predictions = []
        

        for model_type, models in self.base_models.items(): # run all base models on input
            for model in models:
                if model_type == 'lightgbm':
                    # lightgbm
                    pred = model.predict(X)
                elif model_type == 'tabnet':
                    # tabnet
                    pred = model.predict(X)
                else:
                    # catboost and xgboost
                    pred = model.predict(X)
                
                all_predictions.append(pred)
        
        return np.column_stack(all_predictions)
    
    def _get_base_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Get probabilities and return array with prob for each base model
        
        Args:
            X: features
            
        Returns:
            Array of shape (n_samples, n_models * n_classes) with probabilities
        """
        all_probabilities = []
        
        for model_type, models in self.base_models.items():
            for model in models:
                if model_type == 'lightgbm':
                    # lightgbm
                    prob = model.predict_proba(X)
                elif model_type == 'tabnet':
                    # tabnet
                    prob = model.predict_proba(X)
                else:
                    # catboost and xgboost
                    prob = model.predict_proba(X)

                all_probabilities.append(prob)
        return np.hstack(all_probabilities)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Prediction using ensemble model

        Args:
            X: features
    
        Returns:
            Predicted classes
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Get base model predictions
        base_preds = self._get_base_predictions(X)
        
        # Get meta-model predictions
        meta_preds = self.meta_model.predict(base_preds)
        
        return meta_preds
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get prediction probabilities using the ensemble model.
        
        Args:
            X: features
            
        Returns:
            Probabilities for each class
        """

        if isinstance(X, pd.DataFrame): # convert if needed
            X = X.values
        base_probs = self._get_base_probabilities(X)
        meta_probs = self.meta_model.predict_proba(base_probs)
        
        return meta_probs
    
    def predict_single(self, features: Union[List, np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Make a prediction for a single celestial object.
        
        Args:
            features: Feature vector for a single object
        Returns:
            Dictionary with prediction results
        """
        if isinstance(features, (list, pd.Series)):
            features = np.array(features).reshape(1, -1)
        elif features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Get prediction and probabilities
        prediction = self.predict(features)[0]
        probabilities = self.predict_proba(features)[0]
        
        # Map class numbers to labels
        class_labels = {0: "Not Exoplanet", 1: "Exoplanet", 2: "Uncertain"}
        
        result = {
            "predicted_class": int(prediction),
            "predicted_label": class_labels[prediction],
            "probabilities": {
                class_labels[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            "confidence": float(max(probabilities))
        }
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded models."""
        info = {
            "base_models": {},
            "meta_model": str(type(self.meta_model).__name__),
            "total_models": sum(len(models) for models in self.base_models.values())
        }
        
        for model_type, models in self.base_models.items():
            info["base_models"][model_type] = {
                "count": len(models),
                "parameters": self.best_params.get(model_type, {})
            }
        return info


def load_classifier(model_dir: str = ".") -> ExoplanetClassifier:
    """
    Function to load the classifier directory containing the trained models(made for ease of use in other scripts)
    
    Args:
        model_dir: Directory where the trained models are stored
    Returns:
        ExoplanetClassifier instance
    """
    return ExoplanetClassifier(model_dir)

