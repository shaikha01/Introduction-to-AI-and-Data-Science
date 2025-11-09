# Crop Recommendation System

## CS316: Introduction to AI and Data Science
**Instructor:** Dr. Heyfa Ammar
**Date:** October 2025

## Project Overview

This project develops a machine learning-based crop recommendation system to support sustainable agriculture practices. By analyzing soil nutrient levels and environmental conditions, the system recommends optimal crops for cultivation, helping farmers make data-driven decisions that can reduce resource waste and improve agricultural productivity.

## Sustainability Impact

This project aligns with **UN Sustainable Development Goal 2: Zero Hunger** by:

- Optimizing crop selection based on existing soil and environmental conditions
- Reducing unnecessary use of fertilizers and pesticides through better crop-soil matching
- Helping farmers maximize yield potential with appropriate crop choices
- Supporting sustainable agricultural practices through data-driven decision making
- Minimizing resource waste and environmental impact

## Dataset

**Source:** [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

**Description:**
- Total samples: 2,200
- Number of crops: 22 (rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee)
- Features: 7 input features
- Target: Crop label (classification)

**Features:**
1. **N** - Nitrogen content ratio in soil (kg/ha)
2. **P** - Phosphorus content ratio in soil (kg/ha)
3. **K** - Potassium content ratio in soil (kg/ha)
4. **temperature** - Temperature in degrees Celsius
5. **humidity** - Relative humidity in percentage
6. **ph** - pH value of the soil
7. **rainfall** - Rainfall in mm

**Dataset Characteristics:**
- Perfectly balanced: 100 samples per crop
- No missing values
- No duplicate entries
- Clean and ready for model training

## Project Structure

```
Crop Recommendation Models/
│
├── 01_Data_Exploration.ipynb       # Exploratory Data Analysis
├── 02_Model_Training.ipynb         # Model training and comparison
├── 03_Evaluation.ipynb             # Model evaluation and testing
├── crop_recommendation.csv         # Dataset
├── README.md                       # Project documentation
│
├── images/                         # Generated visualizations
│   ├── crop_distribution.png
│   ├── feature_distributions.png
│   ├── boxplots_outliers.png
│   ├── correlation_heatmap.png
│   ├── features_by_crop.png
│   ├── pairplot_features.png
│   ├── feature_variability.png
│   ├── model_comparison.png
│   ├── comprehensive_metrics.png
│   ├── confusion_matrix_best_model.png
│   ├── cross_validation_results.png
│   ├── feature_importance_rf.png
│   ├── confusion_matrix.png
│   ├── prediction_confidence.png
│   └── feature_importance.png
│
└── files/                          # Output files and saved models
    ├── model_comparison_results.csv
    ├── comprehensive_metrics.csv
    ├── feature_importance_rf.csv
    └── crop_recommendation_model.pkl
```

## Methodology

### Phase 1: Data Exploration

**Notebook:** `01_Data_Exploration.ipynb`

Key activities:
- Dataset loading and initial inspection
- Data quality assessment (missing values, duplicates)
- Statistical analysis of features
- Distribution analysis and outlier detection
- Correlation analysis between features
- Feature relationship visualization
- Feature variability analysis

**Key Findings:**
- All features show appropriate ranges for agricultural data
- Strong correlation between Phosphorus (P) and Potassium (K): 0.74
- Potassium (K) shows highest variability (CV: 105%)
- No data quality issues requiring preprocessing

### Phase 2: Model Training

**Notebook:** `02_Model_Training.ipynb`

**Data Split:**
- Training set: 80% (1,760 samples)
- Test set: 20% (440 samples)
- Stratified split to maintain class balance

**Models Trained:**

1. **Logistic Regression**
   - Training Accuracy: 97.39%
   - Test Accuracy: 97.27%
   - Features: Scaled using StandardScaler

2. **Decision Tree Classifier**
   - Training Accuracy: 100.00%
   - Test Accuracy: 97.95%
   - Features: Original features (no scaling required)

3. **Random Forest Classifier** (Selected Model)
   - Training Accuracy: 100.00%
   - Test Accuracy: 99.55%
   - Number of estimators: 100
   - Features: Original features (no scaling required)

4. **XGBoost** (Attempted)
   - Not available in the execution environment
   - Skipped due to library installation constraints

**Model Selection Criteria:**
- Highest test accuracy: Random Forest (99.55%)
- Excellent cross-validation performance: 99.32% (±0.43%)
- Low overfitting: 0.45% difference between train and test
- Stable performance across folds

### Phase 3: Model Evaluation

**Notebook:** `03_Evaluation.ipynb`

**Final Model Performance:**
- Accuracy: 99.55%
- Precision: 99.57%
- Recall: 99.55%
- F1-Score: 99.55%

**Error Analysis:**
- Total test samples: 440
- Correctly classified: 438
- Misclassified: 2 (0.45% error rate)
- Misclassification pairs:
  - Blackgram predicted as Maize (1 instance)
  - Rice predicted as Jute (1 instance)

**Feature Importance (Random Forest):**
1. Rainfall (highest importance)
2. Nitrogen (N)
3. Potassium (K)
4. Phosphorus (P)
5. Humidity
6. Temperature
7. pH

**Prediction Confidence:**
- Mean confidence: 95.79%
- Demonstrates high reliability in predictions

## Installation and Requirements

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Library Versions Used

- pandas: 2.0.3
- numpy: 1.24.3
- scikit-learn: latest
- matplotlib: latest
- seaborn: latest

## How to Run

### Step 1: Data Exploration

```bash
jupyter notebook 01_Data_Exploration.ipynb
```

This notebook will:
- Load and analyze the dataset
- Generate statistical summaries
- Create visualizations in the `images/` folder
- Provide insights into feature distributions and relationships

### Step 2: Model Training

```bash
jupyter notebook 02_Model_Training.ipynb
```

This notebook will:
- Split data into training and test sets
- Train multiple classification models
- Compare model performance
- Perform cross-validation
- Save results to `files/` folder

### Step 3: Model Evaluation

```bash
jupyter notebook 03_Evaluation.ipynb
```

This notebook will:
- Evaluate the Random Forest model in detail
- Generate confusion matrix and classification report
- Analyze prediction confidence
- Save the trained model as `crop_recommendation_model.pkl`

### Making Predictions

After running all notebooks, you can load the saved model and make predictions:

```python
import pickle
import pandas as pd

# Load the trained model
with open('files/crop_recommendation_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Prepare input data
input_data = pd.DataFrame({
    'N': [90],
    'P': [42],
    'K': [43],
    'temperature': [21],
    'humidity': [82],
    'ph': [6.5],
    'rainfall': [202]
})

# Make prediction
prediction = model.predict(input_data)
print(f"Recommended Crop: {prediction[0]}")
```

## Key Results

### Model Comparison

| Model | Training Accuracy | Test Accuracy | Overfitting |
|-------|------------------|---------------|-------------|
| Random Forest | 100.00% | 99.55% | 0.45% |
| Decision Tree | 100.00% | 97.95% | 2.05% |
| Logistic Regression | 97.39% | 97.27% | 0.11% |

### Cross-Validation Results (5-Fold)

| Model | Mean CV Accuracy | Std Dev |
|-------|-----------------|---------|
| Random Forest | 99.32% | 0.43% |
| Decision Tree | 98.52% | 0.75% |
| Logistic Regression | 96.76% | 1.13% |

### Feature Importance

| Feature | Importance Score |
|---------|-----------------|
| Rainfall | 0.2847 |
| N (Nitrogen) | 0.1771 |
| K (Potassium) | 0.1690 |
| P (Phosphorus) | 0.1317 |
| Humidity | 0.1228 |
| Temperature | 0.0792 |
| pH | 0.0354 |

## Usage Example

```python
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predict crop based on soil and environmental conditions

    Parameters:
    -----------
    N : int - Nitrogen content in soil
    P : int - Phosphorus content in soil
    K : int - Potassium content in soil
    temperature : float - Temperature in Celsius
    humidity : float - Humidity percentage
    ph : float - Soil pH value
    rainfall : float - Rainfall in mm

    Returns:
    --------
    prediction : str - Recommended crop
    confidence : float - Prediction confidence
    """
    import pickle
    import pandas as pd
    import numpy as np

    with open('files/crop_recommendation_model.pkl', 'rb') as file:
        model = pickle.load(file)

    input_data = pd.DataFrame({
        'N': [N], 'P': [P], 'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = np.max(probabilities)

    return prediction, confidence

# Example usage
crop, conf = predict_crop(N=90, P=42, K=43, temperature=21,
                          humidity=82, ph=6.5, rainfall=202)
print(f"Recommended Crop: {crop}")
print(f"Confidence: {conf:.2%}")
```

## Conclusions

### Key Findings

1. **High Accuracy Achievement:** The Random Forest model achieves 99.55% accuracy in crop recommendation, demonstrating excellent performance for agricultural decision support.

2. **Feature Insights:** Rainfall emerges as the most important feature (28.5% importance), followed by soil nutrients (N, P, K), indicating that water availability is the primary driver of crop suitability.

3. **Balanced Performance:** The model shows minimal overfitting (0.45%) and consistent cross-validation results, indicating robust generalization to new data.

4. **Practical Applicability:** With only 2 misclassifications out of 440 test samples, the system demonstrates high reliability for real-world deployment.

### Limitations

1. **Dataset Scope:** The model is trained on 22 crop types with balanced data, which may not reflect real-world agricultural distribution or regional crop preferences.

2. **Static Features:** The system uses static measurements and does not account for temporal variations in weather patterns or seasonal changes.

3. **Geographic Limitations:** The dataset does not include geographic or regional information, which can significantly impact crop suitability.

4. **Environmental Factors:** Other important factors such as soil texture, drainage, altitude, and local climate patterns are not considered.

### Future Work

1. **Enhanced Features:** Incorporate additional environmental factors such as soil texture, drainage capacity, elevation, and historical weather patterns.

2. **Temporal Analysis:** Develop time-series models to account for seasonal variations and climate trends.

3. **Regional Customization:** Create region-specific models that account for local agricultural practices and market demands.

4. **Economic Integration:** Include economic factors such as crop prices, market demand, and cultivation costs for comprehensive recommendations.

5. **Multi-crop Recommendations:** Extend the system to suggest crop rotation strategies and complementary crop combinations.

6. **Mobile Application:** Develop a user-friendly mobile application for farmers to easily access recommendations in the field.

7. **Real-time Updates:** Integrate with weather APIs and soil sensors for real-time, dynamic recommendations.

## Contributors

This project was developed as part of the CS316: Introduction to AI and Data Science course under the supervision of Dr. Heyfa Ammar.

## References

1. Crop Recommendation Dataset - Kaggle: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
2. Scikit-learn Documentation: https://scikit-learn.org/
3. UN Sustainable Development Goals: https://sdgs.un.org/goals
4. Random Forest Algorithm - Breiman, L. (2001). "Random Forests". Machine Learning. 45 (1): 5-32.

## License

This project is developed for educational purposes as part of the CS316 course curriculum.

