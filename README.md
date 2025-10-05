# EXOSCAN - Advanced Exoplanet Classification System

**NASA Space Apps Challenge 2025**

EXOSCAN is an advanced AI-powered web application that uses multi-layer machine learning ensembles to analyze tabular light curve data for precise and reliable exoplanet detection and classification.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)](https://flask.palletsprojects.com/)

## Project Overview

EXOSCAN combines cutting-edge machine learning techniques with an intuitive web interface to democratize exoplanet discovery. Our system processes astronomical data from various sources (Kepler, TESS, K2) and provides accurate classifications with explainable AI insights.

### NASA Space Apps Challenge Solution

This project addresses the challenge of **making exoplanet detection accessible** by:
- Processing diverse astronomical data formats automatically
- Providing real-time predictions with confidence scores
- Offering explainable AI through SHAP analysis
- Creating an intuitive interface for researchers and enthusiasts

## Key Features

### Advanced Machine Learning
- **Multi-Layer AI Ensembles**: XGBoost, LightGBM, CatBoost, and TabNet models
- **Intelligent Preprocessing**: Automatic column mapping and unit conversion
- **Robust Prediction**: Handles data from KOI, TOI, K2, and custom formats
- **High Accuracy**: Trained on comprehensive exoplanet datasets

### Explainable AI
- **SHAP Analysis**: Feature importance and contribution analysis
- **Visual Explanations**: Interactive plots and charts
- **Model Interpretability**: Understand why predictions are made
- **Confidence Scoring**: Reliability assessment for each prediction

### User-Friendly Interface
- **Responsive Web App**: Works on desktop and mobile devices
- **Drag & Drop Upload**: Easy file handling for batch processing
- **Real-time Results**: Instant predictions and visualizations
- **Interactive Data Cards**: Manage multiple exoplanet candidates

### Data Processing
- **Smart Column Recognition**: Automatically maps various naming conventions
- **Batch Processing**: Handle multiple candidates simultaneously
- **Data Validation**: Comprehensive input checking and error handling
- **API Integration**: RESTful endpoints for programmatic access

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js (for frontend development)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/noahfavreau/nasa-space-apps-2025.git
   cd nasa-space-apps-2025
   ```

2. **Set up Python environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Start the application**
   ```bash
   # Terminal 1: Start frontend server
   cd public
   python -m http.server 8002
   
   # Terminal 2: Start backend API
   cd ..
   python src/backend/wsgi.py
   ```

4. **Access the application**
   - Open your browser and navigate to `http://localhost:8002`
   - The API will be available at `http://localhost:5000`

## Project Structure

```
nasa-space-apps-2025/
├── public/                      # Frontend web application
│   ├── index.html              # Homepage
│   ├── prediction.html         # Main prediction interface
│   ├── architecture.html       # System architecture docs
│   ├── equipe.html            # Team information
│   ├── css/                   # Stylesheets
│   ├── js/                    # JavaScript modules
│   │   ├── api-client.js      # API communication layer
│   │   ├── prediction.js      # UI interaction logic
│   │   └── smooth-scroll.js   # Navigation utilities
│   └── assets/                # Images and icons
├── src/                        # Backend source code
│   └── backend/               # Python API server
│       ├── wsgi.py            # Flask application entry point
│       ├── preprocess.py      # Data preprocessing pipeline
│       ├── shap_generator.py  # SHAP analysis module
│       ├── inference.py       # ML model inference
│       ├── pdf_generator.py   # Report generation
│       └── model/             # ML models and training
│           ├── model_architecture2.ipynb
│           ├── preprocessing.ipynb
│           └── dataset/       # Training datasets
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## API Documentation

### Endpoints

#### Prediction Endpoints

**Single Prediction**
```http
POST /api/prediction/preditiondata
Content-Type: application/json

{
  "name": "Kepler-227 b",
  "orbital_period": 9.48803557,
  "stellar_radius": 0.927,
  "rate_of_ascension": 291.934230,
  "declination": 48.141651,
  "transit_duration": 2.9575,
  "transit_depth": 615.8,
  "planet_radius": 2.26,
  "planet_temperature": 793,
  "insolation_flux": 93.59,
  "stellar_temperature": 5455
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "CONFIRMED",
  "confidence": 0.892,
  "processed_data": {...}
}
```

**Batch Processing**
```http
POST /api/prediction/graph
Content-Type: application/json

{
  "batch_data": [
    { /* candidate 1 data */ },
    { /* candidate 2 data */ }
  ]
}
```

#### SHAP Analysis
```http
POST /api/report/shap
Content-Type: application/json

{
  "name": "Kepler-227 b",
  /* ... exoplanet data ... */
}
```

**Response:**
```json
{
  "success": true,
  "shap_values": [0.23, -0.15, 0.34, ...],
  "feature_names": ["orbital_period", "stellar_radius", ...],
  "explanation": "Feature importance analysis"
}
```

#### Utility Endpoints
- `GET /api/prediction/fillexample` - Get sample data for testing
- `GET /api/prediction/graph` - Get sample visualization data

## Technical Details

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Intelligent column mapping across different astronomical surveys
   - Automatic unit conversion and normalization
   - Missing value imputation using KNN
   - Robust scaling for numerical stability

2. **Model Architecture**
   - **XGBoost**: Gradient boosting for tabular data excellence
   - **LightGBM**: Fast gradient boosting with optimal memory usage
   - **CatBoost**: Categorical feature handling without preprocessing
   - **TabNet**: Deep learning for tabular data with attention mechanism

3. **Ensemble Strategy**
   - Weighted voting based on validation performance
   - Cross-validation for robust model selection
   - Confidence estimation through prediction variance

### SHAP Explainability

- **Global Explanations**: Overall feature importance across the dataset
- **Local Explanations**: Individual prediction breakdowns
- **Interactive Visualizations**: Summary plots, waterfall charts, force plots
- **Feature Contribution**: Positive/negative impact analysis

### Frontend Architecture

- **Modular Design**: Separation of API logic and UI interactions
- **Progressive Enhancement**: Works without JavaScript for basic functionality
- **Responsive Design**: Mobile-first approach with flexible layouts
- **Accessibility**: WCAG compliant with keyboard navigation support

## Testing

### Running Tests

```bash
# Test the complete pipeline
python -c "
import sys
sys.path.append('src/backend')
from preprocess import preprocess_api_input

# Test with Kepler-227 b data
result = preprocess_api_input({
    'orbital_period': 9.48803557,
    'stellar_radius': 0.927,
    'rate_of_ascension': 291.934230,
    'declination': 48.141651,
    'transit_duration': 2.9575,
    'transit_depth': 615.8,
    'planet_radius': 2.26,
    'planet_temperature': 793,
    'insolation_flux': 93.59,
    'stellar_temperature': 5455
})

print('Test passed!' if result.get('success') else 'Test failed!')
print(f'Prediction: {result.get(\"prediction\", \"Unknown\")}')
"
```

### Manual Testing

1. Start both servers (frontend and backend)
2. Navigate to `http://localhost:8002/prediction.html`
3. Select "Kepler-227 b" from the object library
4. Click "Run Classification"
5. Verify prediction results and SHAP analysis

## Supported Data Formats

### Input Data Sources
- **Kepler Object of Interest (KOI)**: `koi_period`, `koi_srad`, etc.
- **TESS Object of Interest (TOI)**: `toi_period`, `toi_srad`, etc.
- **K2 Campaign Data**: `k2_period`, `k2_srad`, etc.
- **Custom Formats**: Flexible column mapping

### Required Features
| Feature | Description | Units |
|---------|-------------|-------|
| Orbital Period | Time for one orbit | Days |
| Stellar Radius | Host star radius | Solar radii |
| Rate of Ascension | Right ascension | Degrees |
| Declination | Declination coordinate | Degrees |
| Transit Duration | Duration of transit | Hours |
| Transit Depth | Depth of light curve dip | PPM |
| Planet Radius | Estimated planet radius | Earth radii |
| Planet Temperature | Estimated equilibrium temperature | Kelvin |
| Insolation Flux | Stellar flux received | Earth flux |
| Stellar Temperature | Host star temperature | Kelvin |

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 jupyter

# Run code formatting
black src/
flake8 src/

# Run tests
pytest tests/
```

## Acknowledgments

- **NASA** for the Space Apps Challenge and providing astronomical datasets
- **ESA/Kepler Team** for the comprehensive exoplanet catalogs
- **TESS Mission** for continuous sky monitoring data
- **Scikit-learn Community** for excellent machine learning tools
- **SHAP Developers** for explainable AI capabilities

## Links

- **Live Demo**: [Add deployment URL when available]
- **NASA Space Apps**: [Challenge Page URL]
- **Documentation**: [Additional docs if available]
- **Dataset Sources**: 
  - [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
  - [Kepler Data](https://www.nasa.gov/mission_pages/kepler/main/index.html)
  - [TESS Data](https://tess.mit.edu/)
---

**Made with care for NASA Space Apps Challenge 2025**

*"Bringing distant worlds within reach through the power of AI"*