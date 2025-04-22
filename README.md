# Rainfall Prediction using Weather Data (2015–2025)

This machine learning project predicts:
1. Whether it will rain tomorrow (classification).
2. How much it will rain (regression).

Built using weather data from 2015 to 2025, this model is designed to support local weather forecasting with real-world testing and validation.

---

## Project Overview

The project aims to answer two key questions:
- **Will it rain tomorrow?**
- **If yes, how much rain is expected?**

By training classification and regression models on historical weather data, the project demonstrates a practical approach to daily rainfall prediction, currently tested and functioning in **Osun State, Nigeria**.

---

## Data Collection

- Weather data was collected and organized into **11 CSV files**.
- Data includes features like temperature, humidity, pressure, wind speed, and rainfall.
- All CSV files were merged to create the final dataset for modeling.

You’ll find these files in the [`data/`](./data) folder.

---

## Exploratory Data Analysis (EDA)

- Checked dataset overview using `.info()` and `.describe()`.
- Plotted histograms, KDEs, and correlation heatmaps.
- Analyzed rainfall patterns and relationships between features.
- Visualized the balance in classification target (`RainTomorrow`) and the skew in regression target (`AmountRainTomorrow`).

---

## Feature Engineering

- Handled missing values.
- Applied label encoding to categorical features.
- Transformed skewed features using `log1p` for regression.
- Feature scaling with `StandardScaler`.
- Created train/test splits for both tasks.

---

## Modeling

Two separate ML tasks:
1. **Classification (RainTomorrow)**  
2. **Regression (AmountRainTomorrow)**

### Classification Results (Accuracy | F1 Score | ROC-AUC in %)

| Model               | Accuracy | F1 Score | ROC-AUC |
|--------------------|----------|----------|---------|
| Logistic Regression| 89%      | 91%      | 89%     |
| Decision Tree      | 89%      | 91%      | 88%     |
| Random Forest      | 90%      | 91%      | 89%     |
| Gradient Boosting  | 89%      | 91%      | 88%     |
| XGBoost            | 89%      | 91%      | 89%     |

### Regression Results (R² | MAE | MSE)

| Model               | R² Score | MAE  | MSE  |
|--------------------|----------|------|------|
| Linear Regression  | 0.39     | 0.09 | 0.02 |
| Decision Tree      | -0.5     | 0.12 | 0.04 |
| Random Forest      | 0.41     | 0.08 | 0.02 |
| Gradient Boosting  | 0.43     | 0.08 | 0.02 |
| XGBoost            | 0.42     | 0.08 | 0.02 |

**Final Models Chosen:**
- Classification: **Random Forest**
- Regression: **Gradient Boosting**

---

## Evaluation

- **Confusion Matrix** visualized classification performance:
  - Correct No Rain: 255
  - False Rain predictions: 38
  - Correct Rain: 418
  - Missed Rain: 40
- **Prediction vs Actual** and **Residual Plots** show the regression model performance, with residual skewness of 1.3.

---

## Real-World Testing

The model was tested in real-time for April 22, 2025, and accurately predicted **rainfall**, confirming its usefulness under real-world conditions.

---

## Prediction Example

After fitting the model, a sample input can generate:
- A **classification prediction**: "Will it rain?"
- A **regression prediction**: "How much?"

Rainfall categories:
- `0.0–0.9 mm`: Very Light Rain
- `1–2 mm`: Light Rain
- `>2 mm`: Moderate Rain

Example output:
> **Yes, it will rain tomorrow. It will be light rainfall.**

---

## How to Use

1. Clone the repo or download the notebook.
2. Run the `.ipynb` file with dependencies installed.
3. Input custom weather values to make predictions.

---

## Future Improvements

- Expand the model to work for **all states and countries**.
- Use live weather APIs to automatically scrape real-time data.
- Deploy with PythonAnywhere to **run predictions daily and send email alerts**.
- Build a frontend interface for wider usability.

---

## Author

**Muhammed Abdulrasheed** – Aspiring Data Scientist  
If you enjoyed this project, check out my other work: [Ramadan Length Prediction (1940–2024)](add-link-here)

---

## License

This project is open-source under the [MIT License](LICENSE).
