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
from preprocess import preprocess_for_prediction

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
        
        # Define the original feature names
        self.feature_names = [
            'orbital_period', 'stellar_radius', 'rate_of_ascension', 'declination',
            'transit_duration', 'transit_depth', 'planet_radius', 'planet_temperature',
            'insolation_flux', 'stellar_temperature'
        ]
        
        print(f"Loaded {len(self.base_models)} base models and meta-model") # print information about the loaded models
        print("Available model types:", list(self.base_models.keys()))
        print("Original features:", self.feature_names)
    
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
                    # Load the booster directly
                    booster = lgb.Booster(model_file=model_path)
                    models['lightgbm'].append(booster)
        
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
            model_preds = []
            for model in models:
                if model_type == 'lightgbm':
                    # LightGBM booster
                    best_iter = model.best_iteration or model.current_iteration()
                    proba = model.predict(X, num_iteration=best_iter)
                    pred = np.argmax(proba, axis=1)
                elif model_type == 'tabnet':
                    # TabNet
                    pred = model.predict(X)
                else:
                    # CatBoost and XGBoost
                    pred = model.predict(X)
                
                model_preds.append(pred)
            
            # Average predictions for this model type across folds
            avg_pred = np.mean(model_preds, axis=0)
            all_predictions.append(avg_pred)
        
        return np.column_stack(all_predictions)
    
    def _get_base_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Get probabilities and return array with averaged probabilities for each model type
        
        Args:
            X: features
            
        Returns:
            Array of shape (n_samples, 12) with averaged probabilities from each model type
        """
        model_type_probabilities = {}
        
        for model_type, models in self.base_models.items():
            type_probs = []
            
            for model in models:
                if model_type == 'lightgbm':
                    # LightGBM booster - get probabilities directly
                    best_iter = model.best_iteration or model.current_iteration()
                    prob = model.predict(X, num_iteration=best_iter)
                    # LightGBM should already return 3-class probabilities for multiclass
                elif model_type == 'tabnet':
                    # TabNet
                    prob = model.predict_proba(X)
                else:
                    # CatBoost and XGBoost
                    prob = model.predict_proba(X)
                
                type_probs.append(prob)
            
            # Average probabilities across all folds for this model type (same as dividing by n_folds)
            model_type_probabilities[model_type] = np.mean(type_probs, axis=0)
        
        # Helper function to reshape for meta-model (same as in notebook)
        def reshape_for_meta(array: np.ndarray) -> np.ndarray:
            array = np.asarray(array)
            if array.ndim == 1:
                return array.reshape(-1, 1)
            return array
        
        # Concatenate averaged probabilities from all model types in the same order as training
        # 4 model types × 3 classes = 12 features
        meta_features = np.hstack([reshape_for_meta(model_type_probabilities[model_type]) 
                                  for model_type in ['catboost', 'lightgbm', 'xgboost', 'tabnet']])
        
        return meta_features
    
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
            features: Feature vector for a single object (10 raw features or 12 meta-features)
            
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
        class_labels = {0: "Exoplanet", 1: "Uncertain", 2: "Not Exoplanet"}
        
        result = {
            "predicted_class": int(prediction),
            "predicted_label": class_labels[prediction],
            "probabilities": {
                class_labels[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            "confidence": float(max(probabilities))
        }
        
        return result
    
    def predict_from_raw_features(self, raw_features: Union[List, np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Make prediction from the raw astronomical values.
        
        Args:
            raw_features: raw features in order:
                [orbital_period, stellar_radius, rate_of_ascension, declination,
                 transit_duration, transit_depth, planet_radius, planet_temperature,
                 insolation_flux, stellar_temperature]
        Returns:
            Dictionary with prediction results
        """
        if isinstance(raw_features, (list, pd.Series)):
            raw_features = np.array(raw_features).reshape(1, -1)
        elif raw_features.ndim == 1:
            raw_features = raw_features.reshape(1, -1)
        
        # checking there's 10 features
        if raw_features.shape[1] != 10:
            raise ValueError(f"Expected 10 raw features, got {raw_features.shape[1]}")
        
        # generate meta-features from base models
        meta_features = self._get_base_probabilities(raw_features)
        
        # final prediction from meta-model
        prediction = self.meta_model.predict(meta_features)[0]
        probabilities = self.meta_model.predict_proba(meta_features)[0]
        
        # class numbers to labels
        class_labels = {0: "Exoplanet", 1: "Uncertain", 2: "Not Exoplanet"}
        
        result = {
            "predicted_class": int(prediction),
            "predicted_label": class_labels[prediction],
            "probabilities": {
                class_labels[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            "confidence": float(max(probabilities)),
            "meta_features": meta_features[0].tolist()  # Include the generated meta-features
        }
        
        return result
    
    def predict_batch_from_raw_features(self, raw_features: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make batch predictions from raw astronomical features (10 features).
        
        Args:
            raw_features: Raw astronomical features array of shape (n_samples, 10)
            
        Returns:
            Array of predicted classes
        """
        if isinstance(raw_features, pd.DataFrame):
            raw_features = raw_features.values
        
        # make sure there are 10 features
        if raw_features.shape[1] != 10:
            raise ValueError(f"Expected 10 raw features, got {raw_features.shape[1]}")
        
        # meta features from base model
        meta_features = self._get_base_probabilities(raw_features)
        
        # final predictions from meta-model
        predictions = self.meta_model.predict(meta_features)
        
        return predictions
    
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


# example usage
if __name__ == "__main__":    
    # Load the classifier
    classifier = load_classifier()
    
    print("="*60)
    print("TESTING WITH RAW ASTRONOMICAL FEATURES")
    print("="*60)
    test_data = {
        'period': 365.25,           # orbital_period (days)
        'star_radius': 1.0,         # stellar_radius (solar radii)
        'ra': 291.93,               # rate_of_ascension (degrees)
        'dec': 48.14,               # declination (degrees)
        'duration': 13.0,           # transit_duration (hours)
        'depth': 0.01,              # transit_depth (fraction)
        'planet_radius': 1.0,       # planet_radius (Earth radii)
        'planet_temp': 288.0,       # planet_temperature (K)
        'insolation': 1361.0,       # insolation_flux (W/m²)
        'star_temp': 5778.0         # stellar_temperature (K)
    }
    
    print("Raw input data:")
    for key, value in test_data.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-"*40)
    print("Preprocessing data...")
    
    try:
        processed_data = preprocess_for_prediction(test_data)
        print("Preprocessed features:")
        print(processed_data)
        
        # Extract just the feature values (without disposition if present)
        feature_columns = [col for col in processed_data.columns if col != 'disposition']
        features = processed_data[feature_columns].values[0]
        
        print(f"\nExtracted features: {features}")
        print(f"Feature count: {len(features)}")
        
        print("\n" + "-"*40)
        print("Making prediction...")
        
        result = classifier.predict_from_raw_features(features)
        
        print(f"Prediction: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("Probabilities:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.3f}")
        
        print(f"\nGenerated meta-features: {len(result['meta_features'])} features")
        print("Meta-features:", [f"{x:.3f}" for x in result['meta_features']])
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
