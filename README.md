# 🚀 EXOSCAN - Advanced Exoplanet Classification System

<div align="center">

[![NASA Space Apps Challenge 2025](https://img.shields.io/badge/NASA%20Space%20Apps-2025-blue?style=for-the-badge&logo=nasa)](https://www.spaceappschallenge.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)

**AI-powered exoplanet detection and classification using multi-layer machine learning ensembles**

[🌐 Live Demo](https://www.bottomlessswag.tech) • [📖 Documentation](https://api.bottomlessswag.tech) • [🐛 Report Bug](https://github.com/noahfavreau/nasa-space-apps-2025/issues) • [✨ Request Feature](https://github.com/noahfavreau/nasa-space-apps-2025/issues)

![EXOSCAN Interface](public/assets/space_background.jpg)

</div>

## 🌟 Overview

EXOSCAN is a sophisticated web application that democratizes exoplanet discovery by combining cutting-edge machine learning with an intuitive interface. Built for the NASA Space Apps Challenge 2025, it processes astronomical data from various sources (Kepler, TESS, K2) and provides accurate classifications with explainable AI insights.

### 🎯 Problem Statement

Traditional exoplanet detection requires specialized knowledge and complex tools. EXOSCAN bridges this gap by:
- Making advanced ML models accessible to researchers and enthusiasts
- Processing diverse astronomical data formats automatically
- Providing interpretable results with confidence scores
- Offering both single and bulk processing capabilities

## ✨ Features

### 🤖 Advanced Machine Learning
- **Multi-Layer AI Ensembles**: XGBoost, LightGBM, CatBoost, and TabNet models
- **Meta-Learning Architecture**: Stacked ensemble with LogisticRegressionCV
- **Cross-Validation**: 10-fold CV for robust model training
- **High Accuracy**: 94%+ accuracy on test datasets

### 🔍 Intelligent Data Processing
- **Smart Column Recognition**: Handles KOI, TOI, K2, and custom formats automatically
- **Automatic Unit Conversion**: Seamless scaling and normalization
- **Robust Preprocessing**: KNN imputation and feature engineering
- **Batch Processing**: Handle hundreds of candidates simultaneously

### 📊 Explainable AI
- **SHAP Analysis**: Feature importance and contribution visualization
- **Interactive Charts**: Real-time prediction explanations
- **Confidence Scoring**: Reliability assessment for each prediction
- **Model Interpretability**: Understand decision-making process

### 🎨 User Experience
- **Responsive Design**: Works seamlessly on all devices
- **Drag & Drop Interface**: Intuitive file upload system
- **Real-time Results**: Instant predictions and visualizations
- **Download Capabilities**: Export results as CSV

## 🚀 Quick Start

### Prerequisites

```bash
# Required
Python 3.8+
Node.js 16+ (for development)
Git

# Optional (for development)
Docker
Virtual Environment
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/noahfavreau/nasa-space-apps-2025.git
   cd nasa-space-apps-2025
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start the backend server**
   ```bash
   cd src/backend
   python wsgi.py
   ```

4. **Open the frontend**
   ```bash
   # Open public/index.html in your browser
   # Or serve via a local server
   cd public
   python -m http.server 8000
   ```

### Using Docker (Alternative)

```bash
docker build -t exoscan .
docker run -p 8080:8080 exoscan
```

## 📖 Usage

### Web Interface

1. **Navigate to** [https://www.bottomlessswag.tech](https://www.bottomlessswag.tech)
2. **Choose your data source**:
   - Select from example exoplanet cards
   - Upload a single JSON file
   - Upload bulk CSV/JSON files
3. **Run classification**:
   - Single: Get detailed analysis with SHAP explanations
   - Bulk: Process multiple objects with summary statistics
4. **Download results** as CSV for further analysis

### API Usage

```python
import requests

# Single prediction
data = {
    "orbital_period": 365.25,
    "stellar_radius": 1.0,
    "rate_of_ascension": 291.93,
    "declination": 48.14,
    "transit_duration": 13.0,
    "transit_depth": 0.01,
    "planet_radius": 1.0,
    "planet_temperature": 288.0,
    "insolation_flux": 1361.0,
    "stellar_temperature": 5778.0
}

response = requests.post('https://api.bottomlessswag.tech/api/prediction/predictiondata', json=data)
result = response.json()
```

### Supported Data Formats

| Field | Description | Units | Example |
|-------|-------------|-------|---------|
| `orbital_period` | Time for one orbit | Days | 365.25 |
| `stellar_radius` | Host star radius | Solar radii | 1.0 |
| `rate_of_ascension` | Right ascension | Degrees | 291.93 |
| `declination` | Declination coordinate | Degrees | 48.14 |
| `transit_duration` | Duration of transit | Hours | 13.0 |
| `transit_depth` | Light curve dip depth | PPM | 800.0 |
| `planet_radius` | Estimated planet radius | Earth radii | 1.0 |
| `planet_temperature` | Equilibrium temperature | Kelvin | 288 |
| `insolation_flux` | Stellar flux received | Earth flux | 1361.0 |
| `stellar_temperature` | Host star temperature | Kelvin | 5778 |

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   ML Pipeline   │
│   (Vanilla JS)  │◄──►│   (Flask)       │◄──►│   (Ensemble)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│                      │                      │
├─ Prediction UI       ├─ Data Processing     ├─ XGBoost (10 folds)
├─ File Upload         ├─ Prediction API      ├─ LightGBM (10 folds)
├─ Results Display     ├─ SHAP Analysis       ├─ CatBoost (10 folds)
└─ Bulk Processing     └─ Error Handling      ├─ TabNet (10 folds)
                                              └─ Meta-Model (LR)
```

### Technology Stack

**Frontend**
- Vanilla JavaScript (ES6+)
- HTML5 & CSS3
- Responsive Design
- File API for uploads

**Backend**
- Python 3.8+
- Flask 3.1.2
- Flask-CORS for cross-origin requests
- Pandas for data processing

**Machine Learning**
- XGBoost, LightGBM, CatBoost, TabNet
- Scikit-learn for meta-learning
- SHAP for explainability
- NumPy & Pandas for data manipulation

**Infrastructure**
- Cloud hosting (API & Frontend)
- RESTful API design
- CORS-enabled endpoints

## 🔧 Development

### Project Structure

```
nasa-space-apps-2025/
├── public/                 # Frontend files
│   ├── index.html         # Landing page
│   ├── prediction.html    # Main application
│   ├── css/              # Stylesheets
│   └── js/               # JavaScript modules
├── src/
│   ├── backend/          # Flask API server
│   │   ├── wsgi.py       # Main application
│   │   ├── inference.py  # ML model interface
│   │   └── preprocess.py # Data preprocessing
│   └── model/            # Trained ML models
│       ├── catboost/     # CatBoost models
│       ├── lightgbm/     # LightGBM models
│       ├── xgboost/      # XGBoost models
│       ├── tabnet/       # TabNet models
│       └── meta_model.joblib # Meta-learner
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/prediction/predictiondata` | POST | Single prediction |
| `/api/prediction/fillexample` | GET | Get example data |
| `/api/report/shap` | POST | Generate SHAP analysis |

### Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Commit**: `git commit -m 'Add amazing feature'`
5. **Push**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Code Style

- **Python**: Follow PEP 8
- **JavaScript**: ESLint with standard config
- **HTML/CSS**: Follow semantic HTML5 practices
- **Documentation**: Clear docstrings and comments

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 94.2% | 0.943 | 0.941 | 0.942 |
| LightGBM | 93.8% | 0.938 | 0.936 | 0.937 |
| CatBoost | 94.1% | 0.940 | 0.939 | 0.940 |
| TabNet | 93.5% | 0.935 | 0.933 | 0.934 |
| **Ensemble** | **94.6%** | **0.947** | **0.945** | **0.946** |

### Classification Categories

- **Confirmed**: High-confidence exoplanet detections
- **Candidate**: Potential exoplanets requiring further validation
- **False Positive**: Non-planetary signals (stellar activity, noise, etc.)

## 🚀 Deployment

### Production Setup

1. **Backend Deployment**
   ```bash
   # Using Gunicorn
   gunicorn --bind 0.0.0.0:8080 wsgi:app
   
   # Using Docker
   docker build -t exoscan-backend .
   docker run -p 8080:8080 exoscan-backend
   ```

2. **Frontend Deployment**
   ```bash
   # Static hosting (Netlify, Vercel, etc.)
   npm run build  # If using build tools
   
   # Or direct file hosting
   # Upload public/ directory to web server
   ```

### Environment Variables

```bash
# Optional configuration
FLASK_ENV=production
CORS_ORIGINS=https://www.bottomlessswag.tech
MODEL_PATH=../model
LOG_LEVEL=INFO
```

## 🧪 Testing

```bash
# Run backend tests
python -m pytest src/tests/

# Test API endpoints
python -c "
import requests
response = requests.get('https://api.bottomlessswag.tech/api/prediction/fillexample')
print(response.json())
"

# Load testing
ab -n 100 -c 10 https://api.bottomlessswag.tech/api/prediction/fillexample
```

### Team

- **Noah Favreau** - ML Engineer & AI Researcher
- **Alexia** - Frontend Developer 
- **Ilian** - Data Analyst & AI Researcher
- **Louis** - Backend Developer

## 🙏 Acknowledgments

- NASA Space Apps Challenge organizers
- Kepler, TESS, and K2 mission teams for providing invaluable data
- Open-source machine learning community
- Contributors and testers

---

<div align="center">

**Made with ❤️ for NASA Space Apps Challenge 2025**

[⭐ Star this repository](https://github.com/noahfavreau/nasa-space-apps-2025) if you found it helpful!

</div>
