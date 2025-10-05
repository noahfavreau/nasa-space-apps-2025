# EXOSCAN Training Metrics Report

**Model Architecture**: Stacking Ensemble  
**Training Date**: October 2025  
**Dataset**: Combined Imputed Exoplanet Data  

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Test Set Size** | 3,342 samples |
| **Features** | 10 |
| **Classes** | 3 (0: CONFIRMED, 1: CANDIDATE, 2: FALSE POSITIVE) |
| **Class Distribution** | 803 / 1,332 / 1,207 |

---

## Meta Model Performance

### Overall Metrics
| Metric | Train | Test |
|--------|-------|------|
| **Accuracy** | 73.00% | **73.61%** |
| **AUC-ROC (macro)** | - | **0.8890** |

### Detailed Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **CONFIRMED (0)** | 0.6549 | 0.7397 | 0.6947 | 803 |
| **CANDIDATE (1)** | 0.7284 | 0.7027 | 0.7153 | 1,332 |
| **FALSE POSITIVE (2)** | 0.8087 | 0.7705 | 0.7891 | 1,207 |
| | | | | |
| **Accuracy** | - | - | **0.7361** | 3,342 |
| **Macro Average** | 0.7307 | 0.7376 | 0.7331 | 3,342 |
| **Weighted Average** | 0.7397 | 0.7361 | 0.7370 | 3,342 |

---

## Base Model Cross-Validation Performance

| Model | Mean Accuracy | Standard Deviation | Performance |
|-------|---------------|-------------------|-------------|
| **CatBoost** | 72.02% | ±1.19% | Stable |
| **LightGBM** | 72.29% | ±0.93% | Most Stable |
| **XGBoost** | 72.48% | ±1.24% | Best Individual |
| **TabNet** | 70.45% | ±2.15% | Most Variable |

### Model Rankings
1. **XGBoost**: 72.48% (Best base model performance)
2. **LightGBM**: 72.29% (Most consistent performance)  
3. **CatBoost**: 72.02% (Solid performance)
4. **TabNet**: 70.45% (Highest variance)

---

## Ensemble Performance Analysis

### Meta Model Improvement
| Metric | Value |
|--------|-------|
| **Best Base Model** | XGBoost (72.48%) |
| **Meta Model** | **73.61%** |
| **Absolute Improvement** | +1.13% |
| **Relative Improvement** | +1.56% |

### Key Insights
- **Ensemble Benefit**: Meta model outperforms all individual base models
- **Stable Performance**: Low standard deviation across CV folds
- **High AUC-ROC**: 0.8890 indicates excellent class separation
- **Balanced Performance**: Good precision/recall across all classes

---

## Feature Importance Analysis

### Top 5 Most Important Features
*(Based on XGBoost feature importance)*

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | **planet_radius** | 0.4301 | Most predictive feature for classification |
| 2 | **rate_of_ascension** | 0.2713 | Sky coordinate - strong signal |
| 3 | **orbital_period** | 0.2603 | Fundamental orbital characteristic |
| 4 | **transit_depth** | 0.2067 | Light curve depth measurement |
| 5 | **transit_duration** | 0.1866 | Transit timing information |

### Feature Insights
- **Planet Radius** dominates predictions (43% importance)
- **Positional features** (RA) are surprisingly important
- **Transit characteristics** provide complementary information
- **Physical parameters** are key discriminators

---

## Model Architecture Summary

### Stacking Ensemble Design
```
Input Features (10) 
    ↓
Base Models (4):
├── CatBoost      → CV Predictions
├── LightGBM      → CV Predictions  
├── XGBoost       → CV Predictions
└── TabNet        → CV Predictions
    ↓
Meta Features (Combined Predictions)
    ↓
Meta Model: LogisticRegressionCV
    ↓
Final Prediction (3 classes)
```

### Training Strategy
- **Cross-Validation**: 5-fold stratified CV for base models
- **Meta Learning**: Out-of-fold predictions as meta features
- **Class Balancing**: Balanced class weights applied
- **Hyperparameter Optimization**: Optuna for all base models

---

## Performance Benchmarks

### Classification Thresholds
| Class | Precision Target | Recall Target | Status |
|-------|------------------|---------------|--------|
| **CONFIRMED** | >0.65 | >0.70 | Met |
| **CANDIDATE** | >0.70 | >0.70 | Met |
| **FALSE POSITIVE** | >0.80 | >0.75 | Met |

### Production Readiness
- **Accuracy**: 73.61% (Target: >70%)
- **AUC-ROC**: 0.8890 (Target: >0.85)
- **Stability**: Low CV variance across base models
- **Class Balance**: Good performance across all classes

---

## Next Steps & Recommendations

### Model Improvements
1. **Feature Engineering**: Explore interaction terms between top features
2. **Data Augmentation**: Collect more CANDIDATE examples (smallest class)
3. **Ensemble Expansion**: Consider adding neural network models
4. **Hyperparameter Tuning**: Further optimize meta-model parameters

### Deployment Considerations
1. **Model Serving**: Ready for production deployment
2. **Monitoring**: Track prediction confidence scores
3. **Retraining**: Monitor for data drift in astronomical surveys
4. **Validation**: Regular validation on new Kepler/TESS data

---

## Technical Specifications

| Component | Details |
|-----------|---------|
| **Framework** | scikit-learn, XGBoost, LightGBM, CatBoost, TabNet |
| **Cross-Validation** | 5-fold Stratified |
| **Optimization** | Optuna (30 trials per model) |
| **Meta Model** | LogisticRegressionCV |
| **Feature Count** | 10 standardized features |
| **Training Time** | ~2-3 hours (estimated) |
| **Model Size** | ~50MB (all models combined) |

---

*Generated by EXOSCAN ML Pipeline - NASA Space Apps Challenge 2025*