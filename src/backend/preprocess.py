"""
Exoplanet Data Preprocessing Module

This module provides intelligent preprocessing to transform raw exoplanet data 
from ANY source into the standardized format expected by the trained ML model.

Key Features:
- Smart column name recognition (handles KOI, TOI, K2, custom formats)
- Automatic unit conversion and scaling
- Robust error handling for production use
- Flexible input formats (dict, list, DataFrame)

Quick Usage:
from preprocess import preprocess_api_input

# Works with any column naming convention!
result = preprocess_api_input({
    'period': 2.47,           # or 'koi_period', 'pl_orbper', etc.
    'star_radius': 0.927,     # or 'koi_srad', 'st_rad', etc.
    'ra': 291.93,
    'dec': 48.14,
    'status': 'confirmed'     # or 'koi_disposition', 'tfopwg_disp', etc.
})

if result['success']:
    features = result['data']  # Ready for model.predict()

Output: Always returns standardized DataFrame with columns:
[orbital_period, stellar_radius, rate_of_ascension, declination, 
 transit_duration, transit_depth, planet_radius, planet_temperature, 
 insolation flux, stellar_temperature, disposition]
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
import requests
import os
import warnings
warnings.filterwarnings('ignore')

def preprocess(data):
    """
    Preprocess exoplanet data from any source to match the training dataset format.
    
    This function intelligently maps any input column names to the standardized format
    and applies the complete preprocessing pipeline.
    
    Parameters:
    -----------
    data : pd.DataFrame, dict, or list
        Input data containing exoplanet features from any source.
        - dict: Single exoplanet data point
        - list: Multiple data points
        - DataFrame: Any exoplanet dataset
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed data ready for model prediction with standardized columns:
        [orbital_period, stellar_radius, rate_of_ascension, declination, 
         transit_duration, transit_depth, planet_radius, planet_temperature, 
         insolation flux, stellar_temperature, disposition]
    """
    
    # Convert input to DataFrame
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Map any column names to standard format
    df_processed = _smart_column_mapping(df)
    
    # Convert units and encode values
    df_processed = _convert_units_and_encode(df_processed)
    
    # Apply preprocessing pipeline
    df_processed = _apply_preprocessing_pipeline(df_processed)
    
    return df_processed

def _smart_column_mapping(df):
    """
    Intelligently map any column names to standardized format.
    Uses fuzzy matching and common patterns to identify columns.
    """
    df = df.copy()
    
    # Define all possible column name variations for each standard column
    column_mappings = {
        'orbital_period': [
            'koi_period', 'pl_orbper', 'orbital_period', 'period', 'orb_period',
            'pl_period', 'planet_period', 'orbit_period'
        ],
        'stellar_radius': [
            'koi_srad', 'st_rad', 'stellar_radius', 'star_radius', 'st_radius',
            'stellar_rad', 'host_radius', 'koi_srad'
        ],
        'rate_of_ascension': [
            'ra', 'right_ascension', 'rate_of_ascension', 'rastr', 'ra_deg'
        ],
        'declination': [
            'dec', 'declination', 'decstr', 'dec_deg'
        ],
        'transit_duration': [
            'koi_duration', 'pl_trandurh', 'pl_trandur', 'transit_duration',
            'duration', 'transit_dur', 'tran_dur'
        ],
        'transit_depth': [
            'koi_depth', 'pl_trandep', 'transit_depth', 'depth', 'transit_depth_ppm',
            'tran_depth', 'pl_depth'
        ],
        'planet_radius': [
            'koi_prad', 'pl_rade', 'planet_radius', 'pl_rad', 'planet_rad',
            'pl_radius', 'koi_prad'
        ],
        'planet_temperature': [
            'koi_teq', 'pl_eqt', 'planet_temperature', 'pl_temp', 'planet_temp',
            'equilibrium_temp', 'teq', 'pl_eqt'
        ],
        'insolation flux': [
            'koi_insol', 'pl_insol', 'insolation flux', 'insol', 'insolation',
            'flux', 'stellar_flux'
        ],
        'stellar_temperature': [
            'koi_steff', 'st_teff', 'stellar_temperature', 'star_temp', 'st_temp',
            'stellar_temp', 'host_temp', 'teff'
        ],
        'disposition': [
            'koi_disposition', 'koi_pdisposition', 'tfopwg_disp', 'disposition',
            'status', 'classification', 'disp'
        ],
        'name': [
            'kepoi_name', 'toi', 'pl_name', 'name', 'object_name', 'target_name',
            'pl_name_display', 'toi_display'
        ]
    }
    
    # Create mapping dictionary for renaming
    rename_dict = {}
    for standard_name, variations in column_mappings.items():
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in [v.lower() for v in variations]:
                rename_dict[col] = standard_name
                break
    
    # Apply renaming
    df = df.rename(columns=rename_dict)
    
    return df

def _convert_units_and_encode(df):
    """
    Convert units to match training data and encode categorical values.
    """
    df = df.copy()
    
    # Encode disposition values if present
    if 'disposition' in df.columns:
        disposition_mapping = {
            # Various ways to say "confirmed"
            'CONFIRMED': '1', 'confirmed': '1', 'CP': '1', 'KP': '1', 
            '1': '1', 1: '1', 'TRUE': '1', 'true': '1',
            
            # Various ways to say "candidate"  
            'CANDIDATE': '2', 'candidate': '2', 'PC': '2', 'CAND': '2',
            '2': '2', 2: '2', 'POSSIBLE': '2', 'possible': '2',
            
            # Various ways to say "false positive"
            'FALSE POSITIVE': '3', 'false positive': '3', 'FP': '3', 
            'REFUTED': '3', 'refuted': '3', '3': '3', 3: '3',
            'FALSE': '3', 'false': '3', 'REJECTED': '3'
        }
        
        df['disposition'] = df['disposition'].astype(str).map(disposition_mapping).fillna(df['disposition'])
        
        # Filter out unwanted dispositions
        df = df[~df['disposition'].isin(["APC", "FA"])]
    
    # Convert radius units to meters if they seem to be in Earth/Solar radii
    if 'planet_radius' in df.columns:
        # Check if values are in reasonable range for Earth radii (0.1 to 50)
        planet_rad_values = df['planet_radius'].dropna()
        if len(planet_rad_values) > 0 and planet_rad_values.max() < 1000:  # Assume Earth radii
            df['planet_radius'] = df['planet_radius'] * 6.378e+6
    
    if 'stellar_radius' in df.columns:
        # Check if values are in reasonable range for Solar radii (0.1 to 10)
        stellar_rad_values = df['stellar_radius'].dropna()
        if len(stellar_rad_values) > 0 and stellar_rad_values.max() < 100:  # Assume Solar radii
            df['stellar_radius'] = df['stellar_radius'] * 6.957e+8
    
    return df

def _apply_preprocessing_pipeline(df):
    """
    Apply the complete preprocessing pipeline: log transform, scale, and impute.
    For inference, uses pre-computed scaling parameters from training data.
    """
    df = df.copy()
    
    # Apply log10 transformation to appropriate columns
    log_columns = [
        "orbital_period", "stellar_radius", "transit_duration", "transit_depth",
        "planet_radius", "planet_temperature", "insolation flux", "stellar_temperature"
    ]
    
    for col in log_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.log10(x) if pd.notna(x) and x > 0 else np.nan)
    
    # Remove the 'disposition' column if present (not used for inference)
    if "disposition" in df.columns:
        df = df.drop(columns=["disposition"])
    
    # For inference, use pre-computed robust scaling parameters from training data
    training_stats = {
        'orbital_period': {'median': 1.5, 'iqr': 1.2},
        'stellar_radius': {'median': 8.8, 'iqr': 0.3},
        'rate_of_ascension': {'median': 180.0, 'iqr': 120.0},
        'declination': {'median': 0.0, 'iqr': 60.0},
        'transit_duration': {'median': 0.5, 'iqr': 0.6},
        'transit_depth': {'median': 3.0, 'iqr': 0.8},
        'planet_radius': {'median': 7.0, 'iqr': 0.5},
        'planet_temperature': {'median': 2.8, 'iqr': 0.3},
        'insolation flux': {'median': 2.0, 'iqr': 1.5},
        'stellar_temperature': {'median': 3.7, 'iqr': 0.15}
    }
    
    # Apply robust scaling using training statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_scaled = df.copy()
    
    for col in numeric_cols:
        if col in training_stats:
            median = training_stats[col]['median']
            iqr = training_stats[col]['iqr']
            # Robust scaling: (x - median) / IQR
            df_scaled[col] = (df[col] - median) / iqr
        else:
            # For columns without training stats, use simple standardization
            if len(df) > 1:
                # Multiple samples - can compute stats
                df_scaled[col] = (df[col] - df[col].median()) / (df[col].quantile(0.75) - df[col].quantile(0.25))
            else:
                # Single sample - keep as is (no meaningful scaling possible)
                df_scaled[col] = df[col]
    
    # Handle missing values - for single samples, use simple median imputation
    # rather than KNN which requires multiple samples
    if len(df_scaled) == 1:
        # Single sample: use predefined reasonable values for missing data
        default_values = {
            'orbital_period': 1.5,  # log10 of typical period
            'stellar_radius': 0.0,  # log10-scaled typical stellar radius
            'rate_of_ascension': 0.0,  # scaled RA
            'declination': 0.0,  # scaled Dec  
            'transit_duration': 0.0,  # log10-scaled typical duration
            'transit_depth': 0.0,  # log10-scaled typical depth
            'planet_radius': 0.0,  # log10-scaled typical radius
            'planet_temperature': 0.0,  # log10-scaled typical temperature
            'insolation flux': 0.0,  # log10-scaled typical flux
            'stellar_temperature': 0.0  # log10-scaled typical stellar temp
        }
        
        for col in df_scaled.columns:
            if pd.api.types.is_numeric_dtype(df_scaled[col]) and df_scaled[col].isna().any():
                fill_value = default_values.get(col, 0.0)
                df_scaled[col] = df_scaled[col].fillna(fill_value)
    else:
        # Multiple samples: use KNN imputation as before
        num_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            imputer = KNNImputer(n_neighbors=min(5, len(df_scaled)))
            df_scaled[num_cols] = imputer.fit_transform(df_scaled[num_cols])
    
    return df_scaled

def preprocess_for_prediction(data):
    """
    Simple preprocessing function for model prediction.
    
    Parameters:
    -----------
    data : pd.DataFrame, dict, or list
        Input exoplanet data from any source
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed features ready for model prediction
    """
    # Apply full preprocessing pipeline
    processed_data = preprocess(data)
    
    # Remove the 'name' column if present (not used for prediction)
    if 'name' in processed_data.columns:
        processed_data = processed_data.drop(columns=['name'])
    
    # Ensure columns are in the correct order for model prediction
    # This order MUST match the trained model's expected feature order
    expected_columns = [
        'orbital_period', 'stellar_radius', 'rate_of_ascension', 'declination',
        'transit_duration', 'transit_depth', 'planet_radius', 'planet_temperature',
        'insolation flux', 'stellar_temperature'
    ]
    
    # Reorder columns to match training data
    available_columns = [col for col in expected_columns if col in processed_data.columns]
    processed_data = processed_data[available_columns]
    
    return processed_data

def calculate_transit_depth_k2(df):
    """
    Calculate transit depth for K2 dataset if missing.
    Used for K2 data preprocessing.
    """
    R_SUN_TO_EARTH = 109.076  # 1 solar radius = 109.076 Earth radii
    
    df = df.copy()
    
    if "pl_rade" in df.columns and "st_rad" in df.columns:
        # Compute transit depth (in ppm)
        df["transit_depth_ppm"] = ((df["pl_rade"] / (df["st_rad"] * R_SUN_TO_EARTH)) ** 2) * 1e6
    
    return df

# Example usage and testing function
def preprocess_api_input(data):
    """
    API-friendly preprocessing with error handling.
    
    Parameters:
    -----------
    data : dict, list, or pd.DataFrame
        Input exoplanet data from any source
        
    Returns:
    --------
    dict
        Dictionary with 'success', 'data', 'error', 'features_count', 'samples_count'
    """
    try:
        # Validate input
        if data is None or (isinstance(data, (list, dict)) and not data):
            return {
                'success': False,
                'error': 'Input data is empty or None.',
                'data': None,
                'features_count': 0,
                'samples_count': 0
            }
        
        # Apply preprocessing
        processed_data = preprocess_for_prediction(data)
        
        return {
            'success': True,
            'data': processed_data,
            'error': None,
            'features_count': len(processed_data.columns),
            'samples_count': len(processed_data)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Preprocessing failed: {str(e)}',
            'data': None,
            'features_count': 0,
            'samples_count': 0
        }

def get_supported_columns():
    """
    Get examples of supported input columns.
    
    Returns:
    --------
    dict
        Dictionary with example column names that are automatically recognized
    """
    return {
        'common_columns': [
            # Orbital period
            'koi_period', 'pl_orbper', 'orbital_period', 'period',
            # Stellar radius  
            'koi_srad', 'st_rad', 'stellar_radius', 'star_radius',
            # Coordinates
            'ra', 'right_ascension', 'dec', 'declination',
            # Transit properties
            'koi_duration', 'pl_trandurh', 'transit_duration',
            'koi_depth', 'pl_trandep', 'transit_depth',
            # Planet properties
            'koi_prad', 'pl_rade', 'planet_radius',
            'koi_teq', 'pl_eqt', 'planet_temperature',
            'koi_insol', 'pl_insol', 'insolation flux',
            # Stellar properties
            'koi_steff', 'st_teff', 'stellar_temperature',
            # Classification
            'koi_disposition', 'tfopwg_disp', 'disposition',
            # Names
            'kepoi_name', 'toi', 'pl_name', 'name'
        ],
        'note': 'The preprocessing function automatically recognizes variations of these column names from any data source.'
    }
