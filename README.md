# Rainfall Prediction using Weather Data (2015-2025)

## Overview

This machine learning project develops predictive models to forecast rainfall for the Osun State region in Nigeria. The system addresses two complementary prediction tasks:

1. **Classification**: Predicting whether rain will occur tomorrow (binary outcome)
2. **Regression**: Estimating the amount of rainfall expected tomorrow (in millimeters)

By combining historical weather data spanning a decade with advanced machine learning techniques, this project demonstrates a practical and scalable approach to local weather forecasting with real-world validation.

## Table of Contents

- [Motivation](#motivation)
- [Project Structure](#project-structure)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Motivation

Accurate rainfall prediction is essential for informed decision-making across multiple sectors:

- **Agriculture**: Farmers require precise rainfall forecasts to optimize irrigation schedules, fertilizer application, and harvest planning
- **Logistics and Transportation**: Companies need weather predictions to adjust delivery schedules and ensure operational safety
- **Urban Infrastructure**: City planners and emergency services rely on rainfall predictions to manage drainage systems, prevent flooding, and coordinate emergency response
- **Daily Planning**: Individual users benefit from reliable weather forecasts for personal decision-making

The lack of accessible, localized weather prediction systems in many regions creates an opportunity for developing targeted solutions. This project addresses that gap by creating a region-specific model that can serve as a foundation for broader weather forecasting applications.

## Project Structure

```
rainfall-prediction-main/
├── rainfall_preditor.ipynb        # Main analysis and modeling notebook
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
├── README.md                      # This file
└── data/
    ├── osun_rainfall_data.csv     # Weather data 2024-2025
    ├── osun_rainfall_data2.csv    # Weather data 2023
    ├── osun_rainfall_data3.csv    # Weather data 2022
    ├── osun_rainfall_data4.csv    # Weather data 2021
    ├── osun_rainfall_data5.csv    # Weather data 2020
    ├── osun_rainfall_data6.csv    # Weather data 2019
    ├── osun_rainfall_data7.csv    # Weather data 2018
    ├── osun_rainfall_data8.csv    # Weather data 2017
    ├── osun_rainfall_data9.csv    # Weather data 2016
    ├── osun_rainfall_data10.csv   # Weather data 2015
    └── osun_weather_full_data.csv # Merged and cleaned dataset
```

## Data

### Data Collection

Weather data was collected from the Visual Crossing Weather API, covering a ten-year period from 2015 to 2025. The data collection process involved:

- Querying historical weather records for Osun State, Nigeria
- Organizing data into annual CSV files for easier management
- Implementing caching mechanisms to avoid redundant API calls
- Consolidating all data into a unified dataset

The final merged dataset contains 3,761 observations with 33 weather-related features.

### Features

The dataset includes the following meteorological variables:

- **Temperature Metrics**: Maximum temperature, minimum temperature, apparent temperature (feels-like)
- **Humidity and Pressure**: Relative humidity, atmospheric pressure
- **Precipitation**: Rainfall occurrence, rainfall amount (target variables)
- **Wind**: Wind speed
- **Cloud Cover and Visibility**: Cloud coverage percentage, visibility distance
- **Solar Radiation**: Solar energy, UV index
- **Temporal**: Year, month, day of week, day of year

### Data Processing

Data cleaning and preprocessing steps included:

- Removing duplicate entries based on datetime
- Handling missing values
- Converting data types to appropriate formats
- Feature engineering to extract temporal patterns
- Log transformation of the regression target to address skewness (original skewness: 2.42, post-transformation: 1.65)

## Methodology

### Exploratory Data Analysis

Initial analysis revealed:

- **Target Variable Distribution**: 64.14% rainy days vs. 35.86% non-rainy days (slight class imbalance)
- **Temporal Patterns**: Rainfall peaks during mid-year months and declines toward year-end
- **Feature Relationships**: Strong correlations between humidity, temperature, and rainfall
- **Distribution Characteristics**: Temperature and pressure follow approximately normal distributions; cloud cover and humidity are left-skewed

### Feature Engineering

Key features were engineered to capture temporal and cyclical patterns:

- Day of week adjustment (1-7 scale)
- Monthly and yearly indicators
- Day of year to capture seasonal effects
- Lagged features for rainfall history (rain yesterday, rain last 7 days)

### Model Development

Two complementary models were trained:

1. **Classification Model**: Random Forest classifier
   - Task: Predicting rain occurrence (binary)
   - Architecture: Ensemble of decision trees
   - Hyperparameter tuning performed via grid search

2. **Regression Model**: Gradient Boosting regressor
   - Task: Predicting rainfall amount
   - Architecture: Sequential ensemble of weak learners
   - Target scaled using log transformation

### Model Training

- Dataset split: 80% training, 20% testing
- Cross-validation applied to assess generalization
- Class weights balanced to handle target imbalance in classification task
- Models evaluated on multiple metrics for robustness

## Results

### Classification Performance

**Random Forest Classifier**:
- Accuracy: 90%
- F1-Score: 91%
- Confusion Matrix:
  - True Negatives: 255
  - False Positives: 38
  - True Positives: 418
  - False Negatives: 40

### Regression Performance

**Gradient Boosting Regressor**:
- R-squared (R²): 0.43
- Mean Absolute Error (MAE): 0.08
- Mean Squared Error (MSE): 0.02
- Residual Analysis: Residuals display approximate normality with skewness of 1.3

### Rainfall Intensity Classification

Predicted rainfall amounts are categorized as follows:

- **0.0-0.9 mm**: Very Light Rain
- **1.0-2.0 mm**: Light Rain
- **>2.0 mm**: Moderate Rain

### Real-World Validation

The models were tested on April 22, 2025, with independent weather data. The system successfully predicted rainfall occurrence, confirming practical utility beyond the test set.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/themrandroid/rainfall-prediction.git
cd rainfall-prediction
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import pandas; import sklearn; print('Installation successful')"
```

## Usage

### Running the Analysis

1. Open the Jupyter notebook:
```bash
jupyter notebook rainfall_preditor.ipynb
```

2. Execute cells sequentially to:
   - Load and explore the data
   - Perform feature engineering
   - Train classification and regression models
   - Evaluate model performance
   - Generate predictions

### Making Predictions

To generate predictions for custom weather conditions:

```python
# Prepare input features with weather data
prediction_df = pd.DataFrame({
    'Year': [2025],
    'Month': [4],
    'day_of_week': [2],
    'temperature': [80.4],
    'humidity': [71.5],
    'pressure': [1007.8],
    'windspeed': [9.2],
    'cloudcover': [74.5],
    'visibility': [15],
    'rain_yesterday': [0],
    'rain_last_7_days': [0.97]
    # Include all required features
})

# Generate predictions
rain_prediction = random_forest_model.predict(prediction_df)
rainfall_amount = np.expm1(gradient_boosting_model.predict(prediction_df))

# Interpret results
if rain_prediction[0] == 1:
    print("Yes, it will rain tomorrow")
    if 0 < rainfall_amount[0] < 1:
        print("Intensity: Very Light Rain")
    elif 1 <= rainfall_amount[0] <= 2:
        print("Intensity: Light Rain")
    else:
        print("Intensity: Moderate Rain")
```

## Model Performance

### Key Findings

- **Random Forest** outperformed alternative classification algorithms (Logistic Regression, SVM) achieving 90% accuracy
- **Gradient Boosting** provided the best regression performance with R² of 0.43, demonstrating moderate predictive power for rainfall amount
- Models show consistent performance across cross-validation folds, indicating good generalization
- The slight class imbalance in the target variable was effectively managed through weighted loss functions

### Performance Comparison

Models were evaluated against baseline approaches:

- Baseline (predicting majority class): 64% accuracy
- Random Forest: 90% accuracy (+26% improvement)
- Gradient Boosting MAE: 0.08 mm (compared to mean baseline error of 0.23 mm)

## Future Work

### Immediate Improvements

- Expand temporal coverage with data prior to 2015 and beyond 2025
- Incorporate satellite imagery and atmospheric pressure patterns
- Implement real-time data ingestion from weather APIs for continuous model updating

### Scalability

- Extend model to cover all states in Nigeria and other regions
- Develop region-specific models to capture local climate variations
- Create an ensemble approach combining multiple regional models

### Deployment and Integration

- Deploy model on cloud platforms (AWS, Google Cloud, Azure) for accessibility
- Develop REST API endpoints for programmatic predictions
- Create web interface for end-user access
- Implement automated daily predictions with email/SMS alert system
- Integrate with agricultural and logistics planning platforms

### Advanced Modeling

- Explore deep learning architectures (LSTM, GRU) for temporal sequence modeling
- Investigate ensemble methods combining multiple model types
- Implement transfer learning approaches from related climate datasets
- Develop uncertainty quantification for prediction confidence intervals

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

Please ensure code follows PEP 8 standards and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

The MIT License permits:
- Commercial use
- Modification
- Distribution
- Private use

With the requirement of:
- License and copyright notice

## Author

**Muhammed Abdulrasheed** – Data Science Practitioner

For inquiries or collaboration opportunities, please reach out or check out additional projects:
- [Ramadan Length Prediction (1940-2024)](https://github.com/themrandroid/ramadan-length-prediction)

## Acknowledgments

- Visual Crossing Weather API for historical weather data
- Open-source community for machine learning libraries (scikit-learn, pandas, matplotlib)
- Contributors and reviewers who provided feedback

## References

For additional context on rainfall prediction and weather forecasting methodologies, consider:
- Time series analysis in meteorological data
- Machine learning approaches to precipitation forecasting
- Regional climate modeling techniques

---

**Last Updated**: 2025

For issues, questions, or suggestions, please open an issue in the repository.
