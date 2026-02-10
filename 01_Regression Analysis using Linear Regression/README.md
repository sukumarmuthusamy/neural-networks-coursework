# Regression Analysis using Linear Regression – Salary Prediction

This assignment implements a linear regression model to predict employee salaries based on years of experience. The implementation includes both manual calculations and scikit-learn model fitting, with comprehensive analysis using evaluation metrics and visualization.

## Files Included

- **salary_prediction_linear_regression.ipynb** — Complete Jupyter Notebook with code, outputs, and analysis for the regression task.
- **Assignment1_SalaryData.csv** — Dataset containing years of experience and corresponding salary information.

## Assignment Overview

This assignment covers fundamental concepts in supervised learning through linear regression:
- **Dataset:** Assignment1_SalaryData.csv (Years of Experience vs. Salary)
- **Objective:** Build a predictive model for salary based on years of experience
- **Framework:** Python with Pandas, NumPy, Scikit-Learn, and Matplotlib

## Tasks Implemented

### Task 1: Load Dataset and Create DataFrame
- Upload and import Assignment1_SalaryData.csv into a pandas DataFrame
- Display first few rows and dataset summary statistics
- Verify data types and dataset dimensions

### Task 2: Manual Calculation of Coefficient and Intercept
- Implement linear regression from scratch without using scikit-learn
- Calculate the slope (coefficient) and y-intercept manually using mathematical formulas
- Validate manual implementation against scikit-learn results

### Task 3: Fit Linear Regression Model Using Scikit-Learn
- Train a LinearRegression model from scikit-learn
- Generate predictions for the dataset
- Output the learned coefficient and intercept values

### Task 4: Model Evaluation Using Metrics
- **Mean Absolute Error (MAE):** Average absolute prediction error
- **Mean Squared Error (MSE):** Penalizes larger errors more heavily
- **R² Score:** Measures proportion of variance explained by the model

### Task 5: Visualization
- Scatter plot of actual data points (Years of Experience vs. Salary)
- Fitted regression line overlaid on the scatter plot
- Clear labeling of axes and legend for interpretation

## Model Analysis

The linear regression model demonstrates a strong relationship between years of experience and salary. The manual calculation and scikit-learn results align almost exactly, validating the mathematical implementation.

**Key Findings:**
- **Coefficient (Slope):** ~$9,449 per year of experience — each additional year of experience is associated with approximately $9,449 increase in salary
- **Intercept:** ~$25,792 — the model predicts a base salary level even at zero years of experience, which aligns with entry-level salary expectations
- **Mean Absolute Error (MAE):** ~$4,644 — predictions deviate by about $4.6k on average, which is reasonable relative to the salary range
- **R² Score:** ~0.96 — the model explains approximately 96% of the variation in salary, indicating a very strong fit for a single-feature model

**Interpretation:**
The regression line fits the data trend very well, as visualized in the scatter plot. With an R² score of 0.96, the model captures the relationship effectively. While this simple model performs exceptionally well within this dataset, real-world salary prediction would benefit from additional features such as education level, job role, industry, or location. Nevertheless, the analysis demonstrates that years of experience alone is a strong predictor of salary in this dataset, with low prediction errors and excellent model fit.

## How to View Results

Open the Jupyter Notebook file to see:
- All Python code with syntax highlighting
- Execution outputs for each task
- Data visualizations (plots and graphs)
- Summary statistics and metrics
- Detailed model interpretation

## Requirements

- Python 3.x
- Pandas (data manipulation)
- NumPy (numerical computations)
- Scikit-Learn (machine learning)
- Matplotlib (data visualization)

## Notes

This notebook was developed in Google Colab and includes all required outputs and analysis as per assignment specifications.
