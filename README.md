# Heart Disease Prediction

## Project Overview

This project is focused on predicting heart disease using machine learning techniques. By processing a dataset of medical features, the project aims to build and evaluate models that can accurately classify the likelihood of heart disease.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries and Tools](#libraries-and-tools)
3. [Data Preprocessing](#data-preprocessing)
4. [Modeling](#modeling)
5. [Results](#results)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Conclusion](#conclusion)
9. [Contact](#contact)

## Introduction

Heart disease is a major health issue worldwide. This project utilizes machine learning to predict the presence of heart disease based on various medical parameters. The objective is to provide an accurate and efficient diagnostic tool.

## Libraries and Tools

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning modeling and evaluation

## Data Preprocessing

1. **Data Loading**:
    - Imported the dataset and performed an initial inspection using `data.head()` and `data.info()`.

2. **Exploratory Data Analysis**:
    - Used `describe()`, `boxplot()`, and `hist()` for data exploration.
    - Visualized feature correlations with heatmaps.

3. **Outlier Detection**:
    - Identified and removed outliers in the `chol` feature where values exceeded 400.

4. **Data Splitting**:
    - Split the dataset into training (80%) and testing (20%) sets using `train_test_split`.

## Modeling

1. **Logistic Regression**:
    - Configured with `max_iter=2000` and `C=1`.
    - Trained and evaluated using accuracy scores.

2. **Random Forest**:
    - Set with `max_depth=3`, `min_samples_leaf=4`, and `n_estimators=30`.
    - Assessed feature importance and evaluated model performance.

3. **Hyperparameter Tuning for Random Forest**:
    - Used `GridSearchCV` to optimize parameters such as `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `ccp_alpha`.

4. **K-Nearest Neighbors (KNN)**:
    - Configured with `n_neighbors=6`, `p=1`, `leaf_size=10`, `weights="uniform"`, and `algorithm="brute"`.
    - Evaluated performance on training and testing data.

## Results

- **Logistic Regression**:
    - Training Accuracy: 0.8594
    - Testing Accuracy: 0.8688

- **Random Forest**:
    - Training Accuracy: 0.8574
    - Testing Accuracy: 0.8812

- **K-Nearest Neighbors**:
    - Training Accuracy: 0.8613
    - Testing Accuracy: 0.8699

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/heart-disease-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd heart-disease-prediction
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter notebook or script to view the results.

## Usage

1. **Prepare Data**:
    - Ensure the dataset is available at the specified path.

2. **Train Models**:
    - Run the provided script or Jupyter notebook to train models and evaluate performance.

3. **Predict Outcomes**:
    - Use the trained model to predict heart disease by providing new input data.

## Conclusion

This project successfully demonstrates the application of machine learning in predicting heart disease. The models achieved high accuracy and can serve as a foundation for further development in healthcare diagnostics.

---

### Sample Code (for reference)

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Read data from CSV
data = pd.read_csv('path_to_dataset.csv')

# Explore data
data.info()

# Visualize data
data.boxplot(figsize=(20,10))
plt.show()
data.hist(figsize=(30, 15))
plt.show()

# Remove outliers
data = data[data["chol"] <= 400]

# Visualize correlations
plt.figure(figsize=(30, 20))
sns.heatmap(data.corr(), annot=True, cmap="Blues")
plt.show()

# Split data
x_input = data.drop(columns="target", axis=1)
y_output = data["target"]
x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, train_size=0.8, random_state=42)

# Model 1: Logistic Regression
model_lo = LogisticRegression(max_iter=2000, C=1)
model_lo.fit(x_train, y_train)
train_predict = model_lo.predict(x_train)
training_model = accuracy_score(train_predict, y_train)
print(f"Logistic Regression - Train Accuracy: {training_model}")
test_predict = model_lo.predict(x_test)
testing_model = accuracy_score(test_predict, y_test)
print(f"Logistic Regression - Test Accuracy: {testing_model}")

# Model 2: Random Forest
model_RF = RandomForestClassifier(max_depth=3, min_samples_leaf=4, n_estimators=30)
model_RF.fit(x_train, y_train)
train_predict = model_RF.predict(x_train)
training_model = accuracy_score(train_predict, y_train)
print(f"Random Forest - Train Accuracy: {training_model}")
test_predict = model_RF.predict(x_test)
testing_model = accuracy_score(test_predict, y_test)
print(f"Random Forest - Test Accuracy: {testing_model}")

# Feature importance
features = list(model_RF.feature_importances_)
feature_dict = {data.columns[i]: features[i] for i in range(len(features))}
df_feature = pd.DataFrame({"Feature": feature_dict.keys(), "Score": feature_dict.values()})
plt.figure(figsize=(10, 5))
plt.bar(x=df_feature["Feature"], height=df_feature["Score"])
plt.grid()
plt.show()

# Hyperparameter tuning for Random Forest
param2 = {
    "n_estimators": np.arange(22, 27, 1),
    "max_depth": np.arange(8, 12, 1),
    "min_samples_split": np.arange(1, 3),
    "min_samples_leaf": np.arange(2, 4),
    "ccp_alpha": np.arange(0, 1, 0.2)
}
new_model_RF = GridSearchCV(estimator=model_RF, param_grid=param2, verbose=6, cv=5, n_jobs=-1)
new_model_RF.fit(x_train, y_train)
print(new_model_RF.best_estimator_, new_model_RF.best_score_)

# Model 3: K-Nearest Neighbors
model_KNN = KNeighborsClassifier(n_neighbors=6, p=1, leaf_size=10, weights="uniform", algorithm="brute")
model_KNN.fit(x_train, y_train)
train_predict = model_KNN.predict(x_train)
training_model = accuracy_score(train_predict, y_train)
print(f"KNN - Train Accuracy: {training_model}")
test_predict = model_KNN.predict(x_test)
testing_model = accuracy_score(test_predict, y_test)
print(f"KNN - Test Accuracy: {testing_model}")

# Prediction example
input_data_prediction = np.asarray(list(map(float, input().split(",")))).reshape(1, -1)
prediction = model_RF.predict(input_data_prediction)
print("--" * 20)
if prediction[0] == 0:
    print("This Person Does Not Have Heart Disease.")
else:
    print("This Person Has Heart Disease.")
print("--" * 20)
```

Feel free to reach out for further details or discussions!

---
