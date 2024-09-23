# Wine Quality Analysis using Random Forest

## Overview

This project aims to analyze the quality of wine using machine learning techniques, specifically Random Forest Classifier and Random Forest Regressor. The dataset used is the Wine Quality dataset, which contains various chemical properties of wines and their corresponding quality ratings.

## Dataset

The dataset can be found in the `../wines/wine-quality.csv` file. It includes the following features:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (target variable)

## Requirements

- Python 3.x
- pandas
- scikit-learn

You can install the required packages using pip:

```bash
pip install pandas scikit-learn
```

## Code Structure

### Imports

The necessary libraries for data manipulation and modeling are imported:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
```

### Data Preparation

The dataset is read and preprocessed:

```python
# Read data
read_file = pd.read_csv("../wines/wine-quality.csv")

# Fill null values
read_file.fillna(read_file.mean(), inplace=True)

# Format data
wine_data = pd.DataFrame(read_file)
```

### Classification Model

A Random Forest Classifier is used to predict wine quality:

```python
# Define features and target variable
x = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model_random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
model_random_forest_classifier.fit(x_train, y_train)

# Make predictions
y_pred = model_random_forest_classifier.predict(x_test)

# Evaluate model
precision = accuracy_score(y_test, y_pred)
print("Precision model:", precision)
print("Ranking Report:\n", classification_report(y_test, y_pred, zero_division=1))
```

### Regression Model

A Random Forest Regressor is used to predict the quality of wine:

```python
# Define features and target variable
x = read_file[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol']]
y = read_file['quality']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
model_random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
model_random_forest.fit(x_train, y_train)

# Make predictions
y_pred = model_random_forest.predict(x_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}, R^2: {r2}')
```

## Results

- The classification model outputs precision and a classification report.
- The regression model provides Mean Squared Error (MSE) and R² score to evaluate performance.

## Conclusion

This analysis provides insights into how different chemical properties of wine influence its quality. The Random Forest models effectively capture the underlying patterns in the data for both classification and regression tasks.

## Future Work

Consider experimenting with hyperparameter tuning and feature selection to improve model performance. Additionally, testing other machine learning algorithms may yield interesting comparisons.

---
